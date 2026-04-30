"""Scope-aware Good-Turing / Chao1 estimator.

This module generalises :mod:`src.adaptive.cell_gt` by treating the
unit of analysis as a *scope* (e.g. one column, one column-pair, one
CSV) rather than the whole run. Within each scope every error type is
its own arm with its own per-arm Bernoulli sampling experiment, and
every arm has its own stopping rule. There is **no global rollup**:
column experiments are independent of CSV experiments are independent
of cross-CSV experiments. If a caller wants an overall figure they
must aggregate arm-level estimates explicitly.

The motivation - and the math - are documented in
``docs/per-arm-good-turing.md`` and the conversation summarised in
the design doc; the short version is that pooling species across
heterogeneous arms (e.g. ``typo`` on a numeric column vs. on a name
column) violates the i.i.d. assumption Good-Turing needs and produces
a missing-mass estimate that hides exactly the heterogeneity the user
wants to see.

Public surface:

* :data:`ScopeKind`, :data:`ScopeKey`, :data:`ArmKey` - typed keys.
* :func:`iter_species_keys` - dispatch a detection record to a
  ``(scope_key, locator, error_type)`` triple per implied species.
* :func:`aggregate_runs_by_scope` - K perturbation runs ->
  ``{ArmKey: SpeciesRecord}``.
* :class:`ArmEstimate`, :func:`estimate_by_arm` - per-arm Chao1 + GT
  with a status enum (``saturated`` / ``insufficient`` / ``active`` /
  ``exhausted`` / ``not_applicable``).
* :class:`ArmConfig`, :func:`resolve_arm_config` - hierarchical config
  resolution (arm -> error_type -> scope_kind -> global).
* :func:`evaluate_stop` - evaluate the stop predicate for one arm.
* :func:`is_applicable` - applicability whitelist lookup.
* :func:`next_pass_targets` - scheduler that returns the active scope
  -> error_type map for the next perturbation pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ._chao1 import (
    compute_core as _compute_core,
    frequency_of_frequencies as _shared_freq_of_freq,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


ScopeKind = Literal[
    "dataset",
    "column",
    "column_pair",
    "row",
    "csv",
    "workbook",
    "cross_csv",
    "multi_table",
]

#: Recognised scope kinds. Listed explicitly so ``is_applicable`` and
#: config validators can reject typos quickly.
ALL_SCOPE_KINDS: Tuple[ScopeKind, ...] = (
    "dataset",
    "column",
    "column_pair",
    "row",
    "csv",
    "workbook",
    "cross_csv",
    "multi_table",
)


#: ``(scope_kind, scope_id_tuple)``. ``scope_id_tuple`` is a tuple of
#: strings whose meaning depends on ``scope_kind``; see module docs.
ScopeKey = Tuple[ScopeKind, Tuple[str, ...]]


#: ``(scope_key, error_type)``. The full identifier of one arm of the
#: detection experiment.
ArmKey = Tuple[ScopeKey, str]


#: Scope-specific defect locator. Hashable + canonical-across-runs.
Locator = Tuple[Any, ...]


StopWhen = Literal["gt", "chao1", "both"]
"""Which estimator(s) gate the per-arm saturation predicate.

* ``"gt"`` (default, back-compat) - stop when the Good-Turing point
  estimate of next-emission missing mass ``p_0 = n_1 / N`` falls
  below ``epsilon``. Bounds the *probability mass* on undetected
  species.
* ``"chao1"`` - stop when the Chao1 *upper* 95% CI on the missing
  count, normalised by ``S_obs``, falls below ``epsilon_chao1``.
  Bounds the *count* of undetected species (using the conservative
  upper end of the CI because Chao1 itself is a lower bound on S).
* ``"both"`` - require both criteria simultaneously. Strictly
  stronger than either alone; the recommended setting when a real
  FN guarantee on the dataset is needed.
"""


ArmStatus = Literal[
    "active",
    "saturated",
    "insufficient",
    "exhausted",
    "not_applicable",
]


# ---------------------------------------------------------------------------
# Species iteration: detection record -> (scope_key, locator, error_type)
# ---------------------------------------------------------------------------


def _dataset_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Per-cell species under the singleton ``scope_kind="dataset"``.

    There is exactly one scope_id - the empty tuple - so arms become
    ``(("dataset", ()), error_type)``. Each species is the full cell
    key ``(source, column, row)`` so duplicate detections of the same
    cell across runs collapse into a single species (as with the
    column scope, but pooled across columns + sources).

    Row-level / column-less detections fall back to ``(source, row)``;
    cell-less detections (only a source given) yield nothing.
    """
    source = str(detection.get("source", ""))
    error_type = str(detection.get("error_type", "unknown"))
    if not source:
        return

    rows: Set[int] = set()
    primary_row = detection.get("row")
    if isinstance(primary_row, int):
        rows.add(primary_row)
    for r in detection.get("affected_rows") or ():
        if isinstance(r, int):
            rows.add(r)
    if not rows:
        return

    raw_cols = detection.get("columns") or []
    cols = [str(c) for c in raw_cols]
    scope_key: ScopeKey = ("dataset", ())
    if cols:
        for col in cols:
            for row in rows:
                yield scope_key, (source, col, int(row)), error_type
    else:
        for row in rows:
            yield scope_key, (source, int(row)), error_type


def _column_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Per-cell species expansion for ``scope_kind="column"``.

    One detection that lists ``columns=[c1, c2]`` and
    ``affected_rows=[r1, r2]`` yields four species - one per (column,
    row) pair - each living in its own ``(source, column)`` arm. This
    keeps each arm's Bernoulli model self-contained.
    """
    source = str(detection.get("source", ""))
    error_type = str(detection.get("error_type", "unknown"))

    rows: Set[int] = set()
    primary_row = detection.get("row")
    if isinstance(primary_row, int):
        rows.add(primary_row)
    for r in detection.get("affected_rows") or ():
        if isinstance(r, int):
            rows.add(r)
    if not rows:
        return

    raw_cols = detection.get("columns") or []
    if not raw_cols:
        # Row-level detection is not a column-scope species. Skip.
        return
    cols = [str(c) for c in raw_cols]

    for col in cols:
        scope_key: ScopeKey = ("column", (source, col))
        for row in rows:
            yield scope_key, (int(row),), error_type


def _column_pair_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Species for ``scope_kind="column_pair"``.

    Detections with two or more columns map to the (canonicalised,
    sorted) pair ``(source, c_a, c_b)``. Detections with three+ columns
    fan out to all C(n, 2) pairs - the per-pair singleton count is
    well-defined this way and matches what a multi-column detector
    would produce when probing pairs explicitly.
    """
    source = str(detection.get("source", ""))
    error_type = str(detection.get("error_type", "unknown"))

    raw_cols = detection.get("columns") or []
    cols = sorted({str(c) for c in raw_cols})
    if len(cols) < 2:
        return

    rows: Set[int] = set()
    primary_row = detection.get("row")
    if isinstance(primary_row, int):
        rows.add(primary_row)
    for r in detection.get("affected_rows") or ():
        if isinstance(r, int):
            rows.add(r)
    if not rows:
        return

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            scope_key: ScopeKey = (
                "column_pair",
                (source, cols[i], cols[j]),
            )
            for row in rows:
                yield scope_key, (int(row),), error_type


def _row_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Species for ``scope_kind="row"`` (whole-row findings)."""
    source = str(detection.get("source", ""))
    error_type = str(detection.get("error_type", "unknown"))

    rows: Set[int] = set()
    primary_row = detection.get("row")
    if isinstance(primary_row, int):
        rows.add(primary_row)
    for r in detection.get("affected_rows") or ():
        if isinstance(r, int):
            rows.add(r)
    if not rows:
        return

    scope_key: ScopeKey = ("row", (source,))
    for row in rows:
        yield scope_key, (int(row),), error_type


def _csv_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Species for ``scope_kind="csv"`` (whole-CSV findings).

    Two sub-shapes today:

    * Duplicate / dedup findings: locator is the unordered pair of row
      indices ``frozenset({row, other_row})``. The detection record is
      expected to carry ``other_row`` (or a list ``related_rows``);
      missing -> we fall back to ``(row,)`` so the species is still
      well-defined, just less informative.
    * Schema / uniqueness findings: locator is the column name (single
      tuple element).
    """
    source = str(detection.get("source", ""))
    error_type = str(detection.get("error_type", "unknown"))
    scope_key: ScopeKey = ("csv", (source,))

    # Pair locator: prefer related_rows, then other_row, then row+row.
    related = detection.get("related_rows")
    primary = detection.get("row")
    other = detection.get("other_row")

    pair_rows: List[int] = []
    if isinstance(primary, int):
        pair_rows.append(primary)
    if isinstance(other, int):
        pair_rows.append(other)
    if isinstance(related, (list, tuple)):
        for r in related:
            if isinstance(r, int):
                pair_rows.append(r)

    if len(pair_rows) >= 2:
        unique_pair = tuple(sorted(set(pair_rows)))
        yield scope_key, unique_pair, error_type
        return

    cols = detection.get("columns") or []
    if cols:
        for col in cols:
            yield scope_key, (str(col),), error_type
        return

    if isinstance(primary, int):
        yield scope_key, (primary,), error_type


def _cross_csv_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Species for ``scope_kind="cross_csv"`` (FK / join findings).

    The detection is expected to carry ``other_source`` (and optionally
    ``other_row``) identifying the related-table side of the violation.
    Sources are sorted to canonicalise the unordered pair.
    """
    source = str(detection.get("source", ""))
    other_source = str(detection.get("other_source", ""))
    if not source or not other_source:
        return
    error_type = str(detection.get("error_type", "unknown"))

    pair = tuple(sorted([source, other_source]))
    scope_key: ScopeKey = ("cross_csv", pair)

    primary = detection.get("row")
    other = detection.get("other_row")
    if isinstance(primary, int) and isinstance(other, int):
        # Pair the rows in the same order as the (sorted) sources so
        # the locator is canonical regardless of which side the
        # detector calls "primary".
        if source == pair[0]:
            locator: Locator = (primary, other)
        else:
            locator = (other, primary)
        yield scope_key, locator, error_type
    elif isinstance(primary, int):
        yield scope_key, (primary,), error_type


def _workbook_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Species for ``scope_kind="workbook"`` (cross-sheet, single .xlsx).

    Expects ``workbook`` (str) and ``sheet`` (str) on the detection.
    Locator is ``(sheet, row)`` if a row is given, else ``(sheet,)``.
    Cross-sheet detections (carrying ``other_sheet``) yield a sorted
    sheet pair.
    """
    workbook = detection.get("workbook")
    sheet = detection.get("sheet")
    other_sheet = detection.get("other_sheet")
    if not isinstance(workbook, str) or not isinstance(sheet, str):
        return
    error_type = str(detection.get("error_type", "unknown"))
    scope_key: ScopeKey = ("workbook", (workbook,))

    primary = detection.get("row")
    if isinstance(other_sheet, str):
        sheets = tuple(sorted([sheet, other_sheet]))
        yield scope_key, sheets, error_type
        return
    if isinstance(primary, int):
        yield scope_key, (sheet, primary), error_type
        return
    yield scope_key, (sheet,), error_type


def _multi_table_species(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Species for ``scope_kind="multi_table"`` (3+ tables).

    Expects ``sources`` = list of source names participating in the
    finding. Locator is the sorted tuple of (source, row) pairs that
    were implicated; if rows are missing we fall back to the bare
    sources tuple.
    """
    sources = detection.get("sources")
    if not isinstance(sources, (list, tuple)) or len(sources) < 3:
        return
    error_type = str(detection.get("error_type", "unknown"))
    sources_canon = tuple(sorted(str(s) for s in sources))
    scope_key: ScopeKey = ("multi_table", sources_canon)

    related = detection.get("source_rows")
    if isinstance(related, (list, tuple)) and related:
        # Expect list of [source, row] pairs.
        items: List[Tuple[str, int]] = []
        for entry in related:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and isinstance(entry[1], int)
            ):
                items.append((str(entry[0]), int(entry[1])))
        if items:
            locator: Locator = tuple(sorted(items))
            yield scope_key, locator, error_type
            return

    yield scope_key, sources_canon, error_type


_DISPATCH: Dict[
    ScopeKind,
    Any,  # Callable[[Mapping], Iterable[Tuple[ScopeKey, Locator, str]]]
] = {
    "dataset": _dataset_species,
    "column": _column_species,
    "column_pair": _column_pair_species,
    "row": _row_species,
    "csv": _csv_species,
    "cross_csv": _cross_csv_species,
    "workbook": _workbook_species,
    "multi_table": _multi_table_species,
}


def iter_species_keys(
    detection: Mapping[str, Any],
    *,
    scope_kind: ScopeKind,
) -> Iterable[Tuple[ScopeKey, Locator, str]]:
    """Yield ``(scope_key, locator, error_type)`` triples for one
    detection record under the requested scope kind.

    Detections that don't carry the fields a given scope_kind needs
    (e.g. a column-only detection asked for ``cross_csv``) yield
    nothing - the species is silently skipped, which is the desired
    behaviour: scopes are independent and an arm without observations
    will simply have ``S_obs = 0`` and ``status = active``.
    """
    fn = _DISPATCH.get(scope_kind)
    if fn is None:
        raise ValueError(f"unknown scope_kind: {scope_kind!r}")
    return fn(detection)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class SpeciesRecord:
    """One unique species under a fixed arm.

    Mirrors :class:`src.adaptive.cell_gt.SpeciesRecord` but keys
    on a scope-aware ``(scope, locator)`` pair instead of
    ``(source, column, row)``.
    """

    scope: ScopeKey
    locator: Locator
    runs_found_in: List[int] = field(default_factory=list)
    # Total raw emissions for this species across all runs, with NO
    # within-run de-duplication. See the matching field on
    # ``cell_gt.SpeciesRecord`` for the rationale: it lets the GT
    # denominator track raw detection volume instead of K * arms.
    raw_hits: int = 0

    @property
    def frequency(self) -> int:
        return len(self.runs_found_in)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_kind": self.scope[0],
            "scope_id": list(self.scope[1]),
            "locator": list(self.locator),
            "runs_found_in": sorted(self.runs_found_in),
            "frequency": self.frequency,
            "raw_hits": self.raw_hits,
        }


def aggregate_runs_by_scope(
    runs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    scope_kind: ScopeKind,
) -> Dict[ArmKey, Dict[Locator, SpeciesRecord]]:
    """Merge K perturbation runs into a per-arm species table.

    Returns ``{(scope_key, error_type): {locator: SpeciesRecord}}``.
    A given (arm, locator) pair is counted at most once per run, so
    the per-arm frequency-of-frequencies vector is well-defined with
    ``r in [1, K_a]``.
    """
    table: Dict[ArmKey, Dict[Locator, SpeciesRecord]] = {}
    for run_id, detections in enumerate(runs):
        seen_this_run: Set[Tuple[ArmKey, Locator]] = set()
        for det in detections:
            for scope_key, locator, etype in iter_species_keys(
                det, scope_kind=scope_kind
            ):
                arm: ArmKey = (scope_key, etype)
                key = (arm, locator)
                arm_table = table.setdefault(arm, {})
                rec = arm_table.get(locator)
                if rec is None:
                    rec = SpeciesRecord(scope=scope_key, locator=locator)
                    arm_table[locator] = rec
                # Raw-emission counter (no dedup) feeds the
                # textbook GT denominator (R_x); the
                # ``runs_found_in`` list remains deduped per
                # (arm, locator, run) for diagnostic display only.
                rec.raw_hits += 1
                if key in seen_this_run:
                    continue
                seen_this_run.add(key)
                rec.runs_found_in.append(run_id)
    return table


def frequency_of_frequencies(
    species: Iterable[SpeciesRecord], *, num_runs: int
) -> Dict[int, int]:
    """Bucket per-arm species into ``{r: n_r}`` for ``r`` in ``[1, K_a]``.

    Thin wrapper over :func:`src.adaptive._chao1.frequency_of_frequencies`
    that pulls the ``frequency`` attribute off each record.
    """
    return _shared_freq_of_freq(
        (rec.frequency for rec in species), num_runs=num_runs
    )


# ---------------------------------------------------------------------------
# Per-arm estimator
# ---------------------------------------------------------------------------


@dataclass
class ArmEstimate:
    """Per-arm Chao1 / GT bundle.

    ``status`` is set by :func:`evaluate_stop` (or stays ``"active"``
    if no config is supplied). ``num_runs`` is the per-arm K, NOT the
    global run count.
    """

    arm: ArmKey
    num_runs: int
    n_observed: int
    frequency_counts: Dict[int, int]
    total_hits: int
    good_turing_missing_mass: float
    chao1_estimated_missed: float
    chao1_estimated_total: float
    chao1_variance: float
    chao1_ci_95_low: float
    chao1_ci_95_high: float
    fn_rate_point: float
    fn_rate_ci_95_low: float
    fn_rate_ci_95_high: float
    status: ArmStatus = "active"
    # Filled by stop predicate to keep two-consecutive logic stateless.
    saturated_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        scope_kind, scope_id = self.arm[0]
        return {
            "scope_kind": scope_kind,
            "scope_id": list(scope_id),
            "error_type": self.arm[1],
            "num_runs": self.num_runs,
            "n_observed": self.n_observed,
            "frequency_counts": {
                str(k): v for k, v in self.frequency_counts.items()
            },
            "total_hits": self.total_hits,
            "good_turing_missing_mass": self.good_turing_missing_mass,
            "chao1": {
                "estimated_missed": self.chao1_estimated_missed,
                "estimated_total": self.chao1_estimated_total,
                "variance": self.chao1_variance,
                "ci_95": [
                    self.chao1_ci_95_low,
                    self.chao1_ci_95_high,
                ],
            },
            "estimated_fn_rate": {
                "point": self.fn_rate_point,
                "ci_95": [
                    self.fn_rate_ci_95_low,
                    self.fn_rate_ci_95_high,
                ],
            },
            "status": self.status,
            "saturated_streak": self.saturated_streak,
        }


def _empty_estimate(arm: ArmKey, num_runs: int) -> ArmEstimate:
    return ArmEstimate(
        arm=arm,
        num_runs=num_runs,
        n_observed=0,
        frequency_counts={r: 0 for r in range(1, max(num_runs, 1) + 1)},
        total_hits=0,
        good_turing_missing_mass=0.0,
        chao1_estimated_missed=0.0,
        chao1_estimated_total=0.0,
        chao1_variance=0.0,
        chao1_ci_95_low=0.0,
        chao1_ci_95_high=0.0,
        fn_rate_point=0.0,
        fn_rate_ci_95_low=0.0,
        fn_rate_ci_95_high=0.0,
    )


def estimate_arm(
    arm: ArmKey,
    species: Iterable[SpeciesRecord],
    *,
    num_runs: int,
) -> ArmEstimate:
    """Compute Chao1 + GT for a single arm.

    Same closed-form math as :func:`src.adaptive.cell_gt.estimate`
    but parameterised by ``num_runs = K_a`` (per-arm) so the
    frequency-of-frequencies vector is correctly bounded for arms that
    were probed in fewer passes than the global K.
    """
    species_list = list(species)
    core = _compute_core(
        (int(getattr(rec, "raw_hits", 0)) for rec in species_list),
        num_runs=num_runs,
    )
    if core.n_observed == 0:
        return _empty_estimate(arm, num_runs)
    return ArmEstimate(
        arm=arm,
        num_runs=num_runs,
        n_observed=core.n_observed,
        frequency_counts=core.frequency_counts,
        total_hits=core.total_hits,
        good_turing_missing_mass=core.good_turing_missing_mass,
        chao1_estimated_missed=core.chao1_estimated_missed,
        chao1_estimated_total=core.chao1_estimated_total,
        chao1_variance=core.chao1_variance,
        chao1_ci_95_low=core.chao1_ci_95_low,
        chao1_ci_95_high=core.chao1_ci_95_high,
        fn_rate_point=core.fn_rate_point,
        fn_rate_ci_95_low=core.fn_rate_ci_95_low,
        fn_rate_ci_95_high=core.fn_rate_ci_95_high,
    )


def estimate_by_arm(
    species_table: Mapping[ArmKey, Mapping[Locator, SpeciesRecord]],
    *,
    num_runs_by_arm: Mapping[ArmKey, int],
) -> Dict[ArmKey, ArmEstimate]:
    """Per-arm estimates, no pooling.

    ``num_runs_by_arm`` MUST cover every arm in ``species_table``;
    arms not present in the map fall back to ``0`` (which produces
    an empty estimate). This is intentional - silently defaulting to
    a global K would reintroduce the pooling bug this module exists to
    avoid.
    """
    return {
        arm: estimate_arm(
            arm,
            species_dict.values(),
            num_runs=int(num_runs_by_arm.get(arm, 0)),
        )
        for arm, species_dict in species_table.items()
    }


# ---------------------------------------------------------------------------
# Hierarchical config + applicability
# ---------------------------------------------------------------------------


@dataclass
class ArmConfig:
    """Per-arm stopping knobs.

    Defaults match the recommendations in the design doc. Any field
    can be ``None`` to defer to the next layer in the resolution chain
    (see :func:`resolve_arm_config`).

    Note: a separate ``s_min`` (minimum distinct species before an
    arm could saturate) used to live here. For data-cleaning runs we
    do not require any minimum number of detections per error type -
    legitimately rare error types should be allowed to saturate at
    very low S_obs - so the field has been removed. The ``num_runs``
    floor is enforced by ``k_min`` instead.
    """

    epsilon: Optional[float] = None
    epsilon_chao1: Optional[float] = None
    stop_when: Optional[StopWhen] = None
    k_min: Optional[int] = None
    k_max: Optional[int] = None
    require_two_consecutive: Optional[bool] = None

    @staticmethod
    def default() -> "ArmConfig":
        return ArmConfig(
            epsilon=0.05,
            epsilon_chao1=0.05,
            stop_when="gt",
            k_min=3,
            k_max=10,
            require_two_consecutive=True,
        )

    def merge_into(self, other: "ArmConfig") -> "ArmConfig":
        """Return ``other`` with this object's set fields layered on top.

        ``self`` is the *higher-priority* layer; only its non-None
        fields override.
        """
        return ArmConfig(
            epsilon=self.epsilon if self.epsilon is not None else other.epsilon,
            epsilon_chao1=(
                self.epsilon_chao1
                if self.epsilon_chao1 is not None
                else other.epsilon_chao1
            ),
            stop_when=(
                self.stop_when
                if self.stop_when is not None
                else other.stop_when
            ),
            k_min=self.k_min if self.k_min is not None else other.k_min,
            k_max=self.k_max if self.k_max is not None else other.k_max,
            require_two_consecutive=(
                self.require_two_consecutive
                if self.require_two_consecutive is not None
                else other.require_two_consecutive
            ),
        )


def resolve_arm_config(
    arm: ArmKey,
    *,
    global_default: ArmConfig,
    by_scope_kind: Optional[Mapping[ScopeKind, ArmConfig]] = None,
    by_error_type: Optional[Mapping[str, ArmConfig]] = None,
    by_arm: Optional[Mapping[ArmKey, ArmConfig]] = None,
) -> ArmConfig:
    """Hierarchical resolution: per-arm > per-error-type > per-scope-kind > global.

    Higher layers override lower; a layer with ``None`` for a field
    falls through to the next. The returned :class:`ArmConfig` always
    has every field populated (callers can rely on this).
    """
    cfg = global_default.merge_into(ArmConfig.default())
    scope_kind: ScopeKind = arm[0][0]
    error_type = arm[1]
    if by_scope_kind and scope_kind in by_scope_kind:
        cfg = by_scope_kind[scope_kind].merge_into(cfg)
    if by_error_type and error_type in by_error_type:
        cfg = by_error_type[error_type].merge_into(cfg)
    if by_arm and arm in by_arm:
        cfg = by_arm[arm].merge_into(cfg)
    # All fields must be set after merging into default().default has them.
    return cfg


def is_applicable(
    arm: ArmKey,
    *,
    whitelist: Optional[Mapping[ScopeKind, Sequence[str]]] = None,
) -> bool:
    """Return True iff this arm is allowed to be probed.

    ``whitelist`` maps each ``scope_kind`` to the list of error types
    permitted in that scope. ``None`` (no whitelist) means "everything
    is applicable" - useful for tests and for the legacy default
    behaviour. An empty list means "no error types apply" for that
    scope_kind, which is how callers turn off a whole scope.
    """
    if whitelist is None:
        return True
    scope_kind: ScopeKind = arm[0][0]
    error_type = arm[1]
    allowed = whitelist.get(scope_kind)
    if allowed is None:
        # scope_kind not mentioned -> default-allow.
        return True
    return error_type in allowed


# ---------------------------------------------------------------------------
# Stop predicate
# ---------------------------------------------------------------------------


def evaluate_stop(
    estimate: ArmEstimate,
    cfg: ArmConfig,
    *,
    previous_status: ArmStatus = "active",
    previous_streak: int = 0,
) -> Tuple[ArmStatus, int]:
    """Evaluate the per-arm stop predicate.

    Returns ``(new_status, new_saturated_streak)``. Pure function -
    callers persist the returned streak alongside the estimate.

    Status transitions:

    * ``saturated`` is sticky (already-saturated arms stay saturated).
    * ``exhausted`` (K >= K_max) is sticky.
    * ``not_applicable`` is sticky (set externally).
    * Otherwise: arm is ``saturated`` once the configured criterion
      (``cfg.stop_when``) AND ``K >= k_min`` have both held for
      ``require_two_consecutive`` ? 2 : 1 consecutive passes.
      Available criteria:

      * ``"gt"``    -- ``p_0 <= epsilon`` (GT missing-mass point est.)
      * ``"chao1"`` -- ``(chao1_ci_95_high - S_obs) / S_obs <= epsilon_chao1``
        (upper-95 CI on missing-count fraction)
      * ``"both"``  -- both of the above hold simultaneously.

      ``exhausted`` if ``K >= k_max`` without saturation; else
      ``active``.

    Note: there is no per-arm minimum-species (``s_min``) gate.
    Data-cleaning runs need to allow legitimately rare error types
    to saturate at very low S_obs.
    """
    if previous_status in ("saturated", "exhausted", "not_applicable"):
        return previous_status, previous_streak

    # Field invariants - default() guarantees these are non-None, but
    # belt-and-suspenders so a malformed external config blows up here
    # rather than silently producing wrong answers downstream.
    epsilon = cfg.epsilon
    epsilon_chao1 = cfg.epsilon_chao1
    stop_when: StopWhen = cfg.stop_when or "gt"
    k_min = cfg.k_min
    k_max = cfg.k_max
    two = bool(cfg.require_two_consecutive)
    if (
        epsilon is None
        or epsilon_chao1 is None
        or k_min is None
        or k_max is None
    ):
        raise ValueError(
            "ArmConfig is missing required fields after resolution; "
            "ensure global_default is fully populated."
        )

    # Good-Turing criterion: point estimate of next-emission missing
    # mass below ``epsilon``. GT is upward-biased for the true missing
    # mass so the point estimate is already conservative; no CI wrap.
    gt_ok = estimate.good_turing_missing_mass <= epsilon
    # Chao1 criterion: upper 95% CI on missing count, normalised by
    # S_obs, below ``epsilon_chao1``. Chao1 is a *lower* bound on S, so
    # ``chao1_ci_95_high - n_observed`` is the conservative upper end
    # of the missing count.
    if estimate.n_observed > 0:
        f0_upper = max(
            0.0, estimate.chao1_ci_95_high - estimate.n_observed
        )
        chao1_ratio = f0_upper / estimate.n_observed
    else:
        # No species observed yet: estimator is degenerate. Treat as
        # "satisfied" only if pure-GT mode is in use (which itself
        # returns p_miss=0 in this regime); otherwise fail safe.
        chao1_ratio = float("inf")
    chao1_ok = chao1_ratio <= epsilon_chao1

    if stop_when == "gt":
        criterion_ok = gt_ok
    elif stop_when == "chao1":
        criterion_ok = chao1_ok
    elif stop_when == "both":
        criterion_ok = gt_ok and chao1_ok
    else:
        raise ValueError(
            f"ArmConfig.stop_when has unknown value {stop_when!r}; "
            "expected one of 'gt', 'chao1', 'both'."
        )

    sat_now = criterion_ok and estimate.num_runs >= k_min
    new_streak = previous_streak + 1 if sat_now else 0
    threshold = 2 if two else 1
    if new_streak >= threshold:
        return "saturated", new_streak

    if estimate.num_runs >= k_max:
        # K_max reached without saturation: mark ``exhausted`` so the
        # report flags it as a "needs more runs" arm even though we
        # won't spend any. (We used to also produce ``insufficient``
        # when S_obs was below an ``s_min`` floor, but that gate has
        # been removed - rare error types must be allowed to saturate
        # at very low S_obs in data-cleaning runs.)
        return "exhausted", new_streak

    return "active", new_streak


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def next_pass_targets(
    estimates: Mapping[ArmKey, ArmEstimate],
) -> Dict[ScopeKey, List[str]]:
    """Group still-active arms by scope.

    Returns ``{scope_key: [error_types still active in that scope]}``.
    Empty dict means the experiment is finished. Lists are
    deterministically sorted so the run-driver gets stable prompt
    composition across passes.
    """
    by_scope: Dict[ScopeKey, List[str]] = {}
    for arm, est in estimates.items():
        if est.status != "active":
            continue
        scope_key, error_type = arm
        by_scope.setdefault(scope_key, []).append(error_type)
    return {k: sorted(v) for k, v in by_scope.items()}


# ---------------------------------------------------------------------------
# Markdown report (additive sibling to the legacy global report)
# ---------------------------------------------------------------------------


def _fmt_pct(p: float) -> str:
    return f"{100.0 * p:6.2f}%"


def _scope_label(scope: ScopeKey) -> str:
    """Human-friendly identifier for a scope, used as a row header."""
    kind, sid = scope
    if not sid:
        return kind
    return f"{kind}: {' / '.join(sid)}"


_STATUS_GLYPH: Dict[ArmStatus, str] = {
    "saturated":      "✓",
    "exhausted":      "✗",
    "insufficient":   "?",
    "active":         "·",
    "not_applicable": "—",
}


def render_per_arm_markdown(
    estimates: Mapping[ArmKey, ArmEstimate],
    *,
    title: str = "Per-arm Good-Turing / Chao1 report",
) -> str:
    """Render a per-`scope_kind` matrix Markdown report.

    One section per ``scope_kind``; within each, one row per
    ``(scope_id, error_type)`` arm with K, S_obs, n1/n2, p_miss,
    FN-hat + 95% CI, and status glyph. The whole document is purely
    descriptive - no global rollup row, since callers asked for each
    scope to be an independent experiment.
    """
    # Group by scope_kind for a per-section table.
    by_kind: Dict[ScopeKind, List[ArmEstimate]] = {}
    for est in estimates.values():
        kind: ScopeKind = est.arm[0][0]
        by_kind.setdefault(kind, []).append(est)

    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        "Each scope is an independent experiment; arms within a scope "
        "(error types) stop independently.\n"
    )
    lines.append(
        "Status legend: "
        "`✓` saturated, `✗` exhausted, `?` insufficient, `·` active, "
        "`—` not applicable.\n"
    )

    if not by_kind:
        lines.append("_no arms observed_\n")
        return "\n".join(lines)

    for kind in sorted(by_kind.keys()):
        section = sorted(
            by_kind[kind],
            key=lambda e: (e.arm[0][1], e.arm[1]),
        )
        lines.append(f"## scope_kind = `{kind}`\n")
        lines.append(
            "| scope | error_type | K | S_obs | n1 | n2 | p_miss "
            "| FN_hat | 95% CI | status |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|---|:---:|"
        )
        for est in section:
            scope_id = " / ".join(est.arm[0][1]) or "(empty)"
            etype = est.arm[1]
            n1 = est.frequency_counts.get(1, 0)
            n2 = est.frequency_counts.get(2, 0)
            ci = (
                f"[{_fmt_pct(est.fn_rate_ci_95_low).strip()}, "
                f"{_fmt_pct(est.fn_rate_ci_95_high).strip()}]"
                if est.n_observed > 0
                else "—"
            )
            lines.append(
                f"| {scope_id} | {etype} | {est.num_runs} | "
                f"{est.n_observed} | {n1} | {n2} | "
                f"{_fmt_pct(est.good_turing_missing_mass).strip()} | "
                f"{_fmt_pct(est.fn_rate_point).strip()} | "
                f"{ci} | {_STATUS_GLYPH.get(est.status, est.status)} |"
            )
        lines.append("")

    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Convenience: end-to-end pipeline for one scope_kind
# ---------------------------------------------------------------------------


def run_estimator(
    runs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    scope_kind: ScopeKind,
    num_runs_by_arm: Optional[Mapping[ArmKey, int]] = None,
    global_default: Optional[ArmConfig] = None,
    by_scope_kind: Optional[Mapping[ScopeKind, ArmConfig]] = None,
    by_error_type: Optional[Mapping[str, ArmConfig]] = None,
    by_arm: Optional[Mapping[ArmKey, ArmConfig]] = None,
    applicability: Optional[Mapping[ScopeKind, Sequence[str]]] = None,
    previous_streaks: Optional[Mapping[ArmKey, int]] = None,
    previous_statuses: Optional[Mapping[ArmKey, ArmStatus]] = None,
) -> Dict[ArmKey, ArmEstimate]:
    """End-to-end: aggregate runs, estimate, evaluate stop predicate.

    ``num_runs_by_arm`` defaults to ``len(runs)`` for every arm
    discovered, which is the right answer when every pass probed every
    active arm (today's typical case). Callers that prune saturated
    arms mid-loop should pass a per-arm K explicitly.

    ``previous_streaks`` / ``previous_statuses`` let the caller
    preserve the two-consecutive-pass guard across loop iterations.
    """
    table = aggregate_runs_by_scope(runs, scope_kind=scope_kind)

    if num_runs_by_arm is None:
        default_k = len(runs)
        num_runs_by_arm = {arm: default_k for arm in table.keys()}

    estimates = estimate_by_arm(table, num_runs_by_arm=num_runs_by_arm)

    gd = global_default or ArmConfig.default()
    streaks = previous_streaks or {}
    statuses = previous_statuses or {}

    out: Dict[ArmKey, ArmEstimate] = {}
    for arm, est in estimates.items():
        if not is_applicable(arm, whitelist=applicability):
            est.status = "not_applicable"
            out[arm] = est
            continue
        cfg = resolve_arm_config(
            arm,
            global_default=gd,
            by_scope_kind=by_scope_kind,
            by_error_type=by_error_type,
            by_arm=by_arm,
        )
        new_status, new_streak = evaluate_stop(
            est,
            cfg,
            previous_status=statuses.get(arm, "active"),
            previous_streak=streaks.get(arm, 0),
        )
        est.status = new_status
        est.saturated_streak = new_streak
        out[arm] = est
    return out
