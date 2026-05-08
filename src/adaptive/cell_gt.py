"""Good-Turing + Chao1 false-negative estimators for multi-run detections.

Context
-------

Running the LLM-based detector once produces a set of flagged defects,
but we have no statistical handle on the MISSED defects ("false
negatives"). A classical trick is to rerun the detector on the same
data under small perturbations (see :mod:`src.perturbations` - typically
row / column shuffles) and treat each distinct real defect as a
"species". A defect that is found by most runs is abundant; one found
by only a single run is a "singleton". Good-Turing's insight is that
the abundance of singletons tells you something about how much mass
has NOT yet been seen.

Two complementary estimators are exposed:

* **Good-Turing missing mass** (Good, 1953): the *probability* that a
  freshly perturbed run discovers a previously-unseen defect. Formally
  ``p_miss = n_1 / N`` where ``n_1`` is the number of species observed
  in exactly one of the K runs and ``N`` is the total number of
  (species, run) hits. Equivalent intuition: "how often will the next
  run surface something new?"

* **Chao1 lower bound on total species count** (Chao, 1984):
  ``S_hat = S_obs + n_1^2 / (2 * n_2)`` (with the usual edge-case
  fallback when ``n_2 == 0``). Rearranged this yields a lower-bound
  estimate of the FALSE-NEGATIVE RATE: ``fn_rate_hat =
  (S_hat - S_obs) / S_hat``. Chao1 also admits a log-normal 95 %
  confidence interval, which is what we surface to the user as the
  "statistical guarantee".

Both estimators operate on the same input: a frequency-of-frequencies
vector ``{r: n_r}`` plus the raw species count ``S_obs``.

Canonical species definition
----------------------------

A *species* is one CELL, identified by::

    (source, column, row)

where:

* ``source`` is the CSV file name,
* ``column`` is the single affected column (or ``None`` for
  multi-column / row-level findings),
* ``row`` is the 0-indexed CSV row.

Note that ``error_type`` is intentionally NOT part of the species
identity. The same cell flagged with ``typo`` in one run and
``inconsistency`` in another counts as ONE species (a single
disputed cell) detected in two runs - not as two singletons. This
matches the practical question Good-Turing answers: "how many DATA
DEFECTS are we missing?", where a defect lives in a cell, not in a
label space.

Group-level detections with ``affected_rows = [r1, r2, ...]`` are
EXPANDED into one species per affected row, which makes the estimator
row-level and comparable across runs that disagree about group
boundaries.

Per-column fanout: if a per-cell detection lists several columns, we
emit one species per column. This means the estimator scores *cell*
stability across runs.

Per-error-type breakdowns are still available via
:func:`estimate_by_group` - they are computed by restricting each cell
to the runs that detected it under a given error_type, so per-type
frequencies are well-defined and disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SpeciesKey = Tuple[str, Optional[str], int]
"""(source, column, row) - see module docstring."""


# ---------------------------------------------------------------------------
# Species canonicalisation
# ---------------------------------------------------------------------------


def iter_species_keys(
    detection: Mapping[str, Any],
) -> Iterable[Tuple[SpeciesKey, str]]:
    """Yield ``(cell_key, error_type)`` pairs implied by a detection.

    ``detection`` is a dict with the keys produced by
    :func:`src.report_builder.flatten_issues`:

    * ``source``        : CSV file name (required).
    * ``error_type``    : taxonomy key (required).
    * ``row``           : primary row (required).
    * ``columns``       : list of columns; may be empty for row-level
                          findings.
    * ``affected_rows`` : optional list of rows for group-level
                          findings. When present, ALL listed rows are
                          expanded (plus ``row`` as a safety net).

    Rows present in neither ``row`` nor ``affected_rows`` are ignored.
    Columns default to ``[None]`` when the detection is row-level.

    The cell key intentionally omits ``error_type`` - per-type
    bookkeeping is returned alongside the key so callers can keep both
    a cell-level union (for missing-mass) and a per-type slice (for
    :func:`estimate_by_group`).
    """
    source = str(detection.get("source", ""))
    error_type = str(detection.get("error_type", "unknown"))

    # Rows - union of primary row and affected_rows.
    rows: set = set()
    primary_row = detection.get("row")
    if isinstance(primary_row, int):
        rows.add(primary_row)
    for r in detection.get("affected_rows") or ():
        if isinstance(r, int):
            rows.add(r)
    if not rows:
        return

    # Columns - one species per column. None = row-level / multi-col.
    raw_cols = detection.get("columns") or []
    cols: List[Optional[str]]
    if not raw_cols:
        cols = [None]
    else:
        cols = [str(c) for c in raw_cols]

    for col in cols:
        for row in rows:
            yield (source, col, int(row)), error_type


# ---------------------------------------------------------------------------
# Cross-run aggregation
# ---------------------------------------------------------------------------


@dataclass
class SpeciesRecord:
    """Per-cell record: which runs found it, with per-error-type detail.

    A *species* here is one CELL ``(source, column, row)``. Cell-level
    identity makes the singleton count what users intuitively expect:
    a cell flagged by run 0 as ``typo`` and by run 1 as
    ``inconsistency`` is ONE species detected in two runs, not
    two singletons. ``runs_by_error_type`` keeps per-type breakdown for
    :func:`estimate_by_group`.
    """

    source: str
    column: Optional[str]
    row: int
    runs_found_in: List[int] = field(default_factory=list)
    runs_by_error_type: Dict[str, List[int]] = field(default_factory=dict)
    # Total raw cell-level emissions for this species across all
    # runs, with NO within-run de-duplication. The textbook GT
    # denominator (sum_r r * n_r) caps each species at one hit per
    # run, which makes the missing-mass ratio collapse to zero just
    # by re-running the loop more times. Tracking raw emissions
    # lets the estimator use a denominator that grows with the
    # model's actual detection volume rather than with K alone.
    raw_hits: int = 0
    # Per-error-type raw-hit breakdown, used by
    # :func:`estimate_by_group` so that the per-type GT denominator
    # also tracks raw emissions (rather than the deduped per-(cell,
    # type, run) count) when the estimator is sliced.
    raw_hits_by_error_type: Dict[str, int] = field(default_factory=dict)

    @property
    def frequency(self) -> int:
        return len(self.runs_found_in)

    @property
    def error_types(self) -> List[str]:
        """Sorted list of error types this cell was tagged with (any run)."""
        return sorted(self.runs_by_error_type.keys())

    @property
    def error_type(self) -> str:
        """Convenience: comma-joined error_types (kept for back-compat)."""
        return ",".join(self.error_types) or "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "column": self.column,
            "row": self.row,
            "error_types": self.error_types,
            "runs_found_in": sorted(self.runs_found_in),
            "runs_by_error_type": {
                t: sorted(rs) for t, rs in self.runs_by_error_type.items()
            },
            "frequency": self.frequency,
            "raw_hits": self.raw_hits,
            "raw_hits_by_error_type": dict(self.raw_hits_by_error_type),
        }


def aggregate_runs(
    runs: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[SpeciesKey, SpeciesRecord]:
    """Merge detections from K runs into one cell-indexed table.

    ``runs[i]`` is the flat detection list from run ``i`` (0-indexed).
    The returned mapping keys each cell by its canonical tuple. A cell
    is counted AT MOST ONCE per run for the cell-level union (so the
    Good-Turing frequency-of-frequencies is well-defined with
    ``r in [1, K]``); error_types are unioned per (cell, run).
    """
    table: Dict[SpeciesKey, SpeciesRecord] = {}
    for run_id, detections in enumerate(runs):
        seen_cells_this_run: set = set()
        seen_pairs_this_run: set = set()  # (cell_key, error_type)
        for det in detections:
            for cell_key, etype in iter_species_keys(det):
                rec = table.get(cell_key)
                if rec is None:
                    rec = SpeciesRecord(
                        source=cell_key[0],
                        column=cell_key[1],
                        row=cell_key[2],
                    )
                    table[cell_key] = rec
                # Raw-emission counter: every yield from
                # ``iter_species_keys`` counts, including repeats
                # within the same run. This is what feeds the
                # ``R_x`` frequency in ``compute_core`` (textbook
                # Good-Turing under Interpretation B).
                rec.raw_hits += 1
                rec.raw_hits_by_error_type[etype] = (
                    rec.raw_hits_by_error_type.get(etype, 0) + 1
                )
                if cell_key not in seen_cells_this_run:
                    seen_cells_this_run.add(cell_key)
                    rec.runs_found_in.append(run_id)
                pair = (cell_key, etype)
                if pair in seen_pairs_this_run:
                    continue
                seen_pairs_this_run.add(pair)
                rec.runs_by_error_type.setdefault(etype, []).append(run_id)
    return table


def frequency_of_frequencies(
    species: Iterable[SpeciesRecord],
    *,
    num_runs: int,
) -> Dict[int, int]:
    """Return ``{r: n_r}`` for ``r`` in ``[1, num_runs]``.

    Thin wrapper over :func:`src.adaptive._chao1.frequency_of_frequencies`
    that pulls the ``frequency`` attribute off each record.
    """
    return _shared_freq_of_freq(
        (rec.frequency for rec in species), num_runs=num_runs
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@dataclass
class Estimate:
    """Bundle of cross-run estimators for one slice of detections."""

    num_runs: int
    n_observed: int                     # S_obs: distinct species seen
    frequency_counts: Dict[int, int]    # n_r
    total_hits: int                     # N = sum_r r * n_r = sum_x R_x
    # Good-Turing
    good_turing_missing_mass: float     # n_1 / N (in [0, 1])
    # Chao1 (bias-corrected when n_2 == 0)
    chao1_estimated_missed: float       # n_1^2 / (2 n_2) (or fallback)
    chao1_estimated_total: float        # S_obs + missed
    chao1_variance: float
    chao1_ci_95_low: float
    chao1_ci_95_high: float
    fn_rate_point: float                # missed / chao1_total
    fn_rate_ci_95_low: float
    fn_rate_ci_95_high: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "n_observed": self.n_observed,
            "frequency_counts": {str(k): v for k, v in self.frequency_counts.items()},
            "total_hits": self.total_hits,
            "good_turing_missing_mass": self.good_turing_missing_mass,
            "chao1": {
                "estimated_missed": self.chao1_estimated_missed,
                "estimated_total": self.chao1_estimated_total,
                "variance": self.chao1_variance,
                "ci_95": [self.chao1_ci_95_low, self.chao1_ci_95_high],
            },
            "estimated_fn_rate": {
                "point": self.fn_rate_point,
                "ci_95": [self.fn_rate_ci_95_low, self.fn_rate_ci_95_high],
            },
        }


# Closed-form Chao1 / GT core lives in the shared ``_chao1`` module so
# the cell-level and per-arm estimators stay numerically identical.
from ._chao1 import (  # noqa: E402
    compute_core as _compute_core,
    frequency_of_frequencies as _shared_freq_of_freq,
)


def estimate(
    species_records: Iterable[SpeciesRecord],
    *,
    num_runs: int,
) -> Estimate:
    """Compute both estimators from a list of :class:`SpeciesRecord`.

    Each species' ``raw_hits`` (total detection emissions, NO
    within-pass de-duplication) is fed to ``compute_core`` as its
    ``R_x``. The textbook GT identity ``N = sum_x R_x`` then holds,
    and ``good_turing_missing_mass = n_1 / N`` is the canonical
    estimate of the probability that the next emission belongs to
    a previously unseen species.

    Edge cases handled:

    * No detections at all -> returns zeroed estimators (FN rate is
      reported as 0.0 because we have no evidence either way).
    * ``n_2 == 0`` -> use the bias-corrected Chao1 formula
      ``n_1 * (n_1 - 1) / 2`` which is the standard recommendation when
      doubletons are absent.
    * ``n_1 == 0`` -> estimated missed species is 0 (every species was
      observed at least twice); FN rate 0.
    """
    species_list = list(species_records)
    core = _compute_core(
        (int(getattr(rec, "raw_hits", 0)) for rec in species_list),
        num_runs=num_runs,
    )
    return Estimate(
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


# ---------------------------------------------------------------------------
# Convenience: slice by (error_type, source) for per-group estimates
# ---------------------------------------------------------------------------


def estimate_by_group(
    species: Iterable[SpeciesRecord],
    *,
    num_runs: int,
    key: str = "error_type",
    num_runs_by_key: Optional[Mapping[str, int]] = None,
) -> Dict[str, Estimate]:
    """Group species by ``key`` in {"error_type", "source"} and estimate.

    For ``key="source"`` each cell falls into exactly one bucket (the
    cell key already includes its source).

    For ``key="error_type"`` a cell may be tagged with multiple types
    across runs. Each type T's bucket contains a *projected* record:
    same cell, but ``runs_found_in`` restricted to the runs that tagged
    the cell with T. Per-type singletons are therefore cells flagged as
    T in exactly one run, which is the natural per-type quantity.

    ``num_runs_by_key`` lets the caller override ``num_runs`` per
    bucket. This matters when types were probed in different numbers
    of passes (e.g. the adaptive loop prunes saturated types from later
    passes - their effective K is the number of passes they were
    actually evaluated in, NOT the total number of passes that
    completed). Buckets without an entry fall back to the global
    ``num_runs``.
    """
    if key not in {"error_type", "source"}:
        raise ValueError("key must be 'error_type' or 'source'")
    buckets: Dict[str, List[SpeciesRecord]] = {}
    if key == "source":
        for rec in species:
            buckets.setdefault(rec.source, []).append(rec)
    else:
        for rec in species:
            for etype, runs in rec.runs_by_error_type.items():
                # Project the per-type raw-hit count so the
                # downstream ``estimate`` denominator is the per-
                # type raw emissions for this cell, not the cell's
                # total raw emissions across every type.
                per_type_raw = int(
                    rec.raw_hits_by_error_type.get(etype, 0)
                )
                projected = SpeciesRecord(
                    source=rec.source,
                    column=rec.column,
                    row=rec.row,
                    runs_found_in=list(runs),
                    runs_by_error_type={etype: list(runs)},
                    raw_hits=per_type_raw,
                    raw_hits_by_error_type={etype: per_type_raw},
                )
                buckets.setdefault(etype, []).append(projected)
    overrides = num_runs_by_key or {}
    return {
        k: estimate(v, num_runs=max(1, int(overrides.get(k, num_runs))))
        for k, v in buckets.items()
    }


def _fmt_pct(p: float) -> str:
    return f"{100.0 * p:6.2f}%"


def _fmt_range(a: float, b: float) -> str:
    return f"[{100.0 * a:6.2f}%, {100.0 * b:6.2f}%]"


def render_markdown_report(
    *,
    overall: Estimate,
    per_type: Dict[str, Estimate],
    per_source: Dict[str, Estimate],
    run_info: List[Dict[str, Any]],
    num_runs: int,
    mode: str = "fixed",
    num_runs_max: Optional[int] = None,
    min_runs: Optional[int] = None,
    target_missing_mass: Optional[float] = None,
    stop_reason: Optional[str] = None,
    progression: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render a human-readable Markdown report for the GT estimates.

    Used by both the (legacy) standalone GT entry point and the
    adaptive mode of ``run_cleaning_detection``.
    """
    lines: List[str] = []
    lines.append("# Good-Turing / Chao1 false-negative report\n")
    lines.append(f"- Mode: **{mode}**")
    if mode == "adaptive":
        lines.append(
            f"- Runs completed: **{num_runs}** / max **{num_runs_max}** "
            f"(min {min_runs}, target missing mass <= "
            f"{target_missing_mass:.4f})"
        )
    else:
        lines.append(f"- Runs: **{num_runs}**")
    if stop_reason:
        lines.append(f"- Stop reason: {stop_reason}")
    lines.append(f"- Total distinct defects observed: **{overall.n_observed}**")
    lines.append(
        f"- Good-Turing missing mass: **{_fmt_pct(overall.good_turing_missing_mass)}** "
        "(chance that another perturbed run finds a new defect)"
    )
    lines.append(
        "- Chao1 estimated total defects: "
        f"**{overall.chao1_estimated_total:.1f}** "
        f"(95% CI {overall.chao1_ci_95_low:.1f}\u2013{overall.chao1_ci_95_high:.1f})"
    )
    lines.append(
        "- Estimated false-negative rate: "
        f"**{_fmt_pct(overall.fn_rate_point)}**, 95% CI "
        f"{_fmt_range(overall.fn_rate_ci_95_low, overall.fn_rate_ci_95_high)}"
    )
    lines.append("")

    lines.append("## Frequency of frequencies (overall)")
    lines.append("| detected_in_r_runs | num_defects |")
    lines.append("|---:|---:|")
    for r in sorted(overall.frequency_counts):
        lines.append(f"| {r} | {overall.frequency_counts[r]} |")
    lines.append("")

    def _section(title: str, d: Dict[str, Estimate]) -> None:
        if not d:
            return
        lines.append(f"## {title}")
        lines.append(
            "| group | S_obs | n_1 | n_2 | GT missing | Chao1 total | "
            "FN rate (point) | FN rate 95% CI |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key in sorted(d):
            e = d[key]
            lines.append(
                f"| {key} | {e.n_observed} | "
                f"{e.frequency_counts.get(1, 0)} | "
                f"{e.frequency_counts.get(2, 0)} | "
                f"{_fmt_pct(e.good_turing_missing_mass)} | "
                f"{e.chao1_estimated_total:.1f} | "
                f"{_fmt_pct(e.fn_rate_point)} | "
                f"{_fmt_range(e.fn_rate_ci_95_low, e.fn_rate_ci_95_high)} |"
            )
        lines.append("")

    _section("Per error_type", per_type)
    _section("Per source", per_source)

    if progression:
        lines.append("## Estimate progression")
        lines.append(
            "| runs | S_obs | detections_this_run | GT missing | "
            "Chao1 total | 95% CI | FN rate |"
        )
        lines.append("|---:|---:|---:|---:|---:|---|---:|")
        for row in progression:
            lines.append(
                f"| {row['runs_so_far']} | {row['n_observed']} | "
                f"{row['detections_this_run']} | "
                f"{_fmt_pct(row['good_turing_missing_mass'])} | "
                f"{row['chao1_estimated_total']:.1f} | "
                f"[{row['chao1_ci_95_low']:.1f}, "
                f"{row['chao1_ci_95_high']:.1f}] | "
                f"{_fmt_pct(row['fn_rate_point'])} |"
            )
        lines.append("")

    lines.append("## Runs")
    lines.append("| run_id | effective_seed | sources | detections | failed_calls |")
    lines.append("|---:|---:|---|---:|---:|")
    for r in run_info:
        lines.append(
            f"| {r['run_id']} | {r.get('effective_seed', '')} | "
            f"{', '.join(r.get('sources', []))} | {r.get('detections', 0)} | "
            f"{r.get('failed_calls', 0)} |"
        )
    lines.append("")

    lines.append("## How to read this")
    lines.append(
        "* **Good-Turing missing mass** (`n_1 / sum_r r*n_r`) is the Good "
        "(1953) estimator for the probability that the next independent "
        "sample surfaces a never-before-seen species. For us, one run "
        "\u2248 one sample, and \u201cspecies\u201d = a distinct defect "
        "(source + error_type + column + row). Higher = more defects "
        "still uncovered."
    )
    lines.append(
        "* **Chao1** uses singletons (n_1) and doubletons (n_2) to "
        "estimate the number of species the detector has not yet seen. "
        "`S_hat = S_obs + n_1^2 / (2 n_2)`. When n_2 is zero we fall "
        "back to the bias-corrected form `n_1 (n_1 - 1) / 2`."
    )
    lines.append(
        "* **95% CI** uses the Chao & Yang (1993) log-normal interval "
        "around S_hat, which preserves the mathematical constraint "
        "S >= S_obs."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "Estimate",
    "SpeciesKey",
    "SpeciesRecord",
    "aggregate_runs",
    "estimate",
    "estimate_by_group",
    "frequency_of_frequencies",
    "iter_species_keys",
    "render_markdown_report",
]
