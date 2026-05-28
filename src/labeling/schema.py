"""Pydantic schema for label records and API payloads.

Labels are append-only: one JSON line per (annotator, target, ts) write.
The latest entry per (annotator, target_id) wins at read time - see
``store.LabelStore.latest_by_target``. This keeps history + undo trivial
while staying crash-safe (every POST fsyncs a single line).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LabelStatus(str, Enum):
    """Annotator verdict on a cluster (or single detection or cell)."""

    TP = "tp"            # true positive - real error
    FP = "fp"            # false positive - detector was wrong
    PARTIAL = "partial"  # real error but description/fix is off
    UNSURE = "unsure"    # needs a second opinion
    SKIP = "skip"        # defer; keep at bottom of queue
    FN = "fn"            # false negative - cell error the LLM missed
    # Tombstone marker meaning "no effective label" for the target.
    # Written exclusively by the server when an auto-skip-from-
    # exclusion is reverted (DELETE /exclusions/{id}). Treated as
    # absent by ``LabelStore.latest_by_target`` / ``latest_for`` so
    # downstream consumers see the cluster as unlabeled again, while
    # the JSONL retains a full audit trail of the transition.
    UNLABELED = "unlabeled"


class LabelTarget(str, Enum):
    CLUSTER = "cluster"
    ISSUE = "issue"
    CELL = "cell"        # ad-hoc per-(source,row,column) cell label, used
                         # by the FN flow. target_id is built by
                         # :func:`cell_target_id` so it stays stable
                         # across reloads.
    # Per-cluster verdict (TP / FP / Partial / Unsure / Skip) for an
    # inconsistency column-cluster. The cluster *definition* (which
    # values, which canonical, what kind) is persisted separately in
    # ``column_clusters.jsonl`` via :class:`ColumnClusters`; this
    # target type just attaches a verdict so the cell-level label
    # page can treat inconsistency clusters the same as cell-level
    # detections. target_id is built by :func:`column_cluster_target_id`.
    COLUMN_CLUSTER = "column_cluster"


def cell_target_id(source: str, row: int, column: str) -> str:
    """Stable target_id for a cell-level label. Uses NUL separators so
    we never collide with cluster_id / issue_id formats."""
    return f"cell\x00{source}\x00{int(row)}\x00{column}"


def column_cluster_target_id(source: str, column: str, cluster_local_id: str) -> str:
    """Stable target_id for a column-cluster verdict.

    ``cluster_local_id`` is the per-record id (``c1``..``cN``) assigned
    by the column-board UI when the operator saves a ``ColumnClusters``
    record. Within a single (source, column) pair the local id is
    re-used across re-saves as long as the cluster's value-set survives
    intact, so verdicts persist through routine edits. Operators that
    fully rebuild a column's clustering should expect verdicts to
    re-anchor to the new local ids.
    """
    return f"colclust\x00{source}\x00{column}\x00{cluster_local_id}"


class LabelIn(BaseModel):
    """POST /api/labels payload."""

    target: LabelTarget = LabelTarget.CLUSTER
    target_id: str                   # cluster_id or issue_id
    source: str                      # CSV basename (for quick filtering)
    row: Optional[int] = None
    column: Optional[str] = None     # nullable: some labels are row-level
    # Full column set the verdict applies to. Stored on the label so the
    # exporter doesn't have to re-derive it from clusters.json (which
    # can drift if clusters are regenerated). Empty => fall back to
    # ``[column]`` then to the cluster's ``columns``.
    columns: List[str] = Field(default_factory=list)
    status: LabelStatus
    corrected_value: Optional[str] = None
    # Per-column corrections; empty => use the scalar ``corrected_value``
    # for every column in ``columns`` (back-compat with single-column).
    corrected_values: Dict[str, str] = Field(default_factory=dict)
    # Operator's confirmed canonical for column-cluster targets.
    # Only meaningful when ``target=column_cluster`` AND
    # ``status in {tp, partial}``: the operator endorses one of the
    # cluster's surface forms as the single ground-truth value, which
    # flips the exported benchmark cells from bi-directional
    # (``corrected_value`` is the full acceptable set) to
    # one-directional (``corrected_value`` is the canonical scalar).
    # ``None`` (default) means the operator did not endorse any
    # canonical, so the cluster stays bi-directional in the export.
    # The LLM's *proposed* canonical lives on the draft / cluster
    # itself and is preserved separately as ``llm_suggested_value``
    # in the exported benchmark.
    confirmed_canonical: Optional[str] = None
    # Operator-curated per-value corrections for column-cluster targets.
    # Only meaningful when ``target=column_cluster`` AND
    # ``status in {tp, unsure}``: maps each original surface form in
    # the cluster to its desired correction. A scalar value is a
    # one-directional fix ("typo -> canonical"); a list captures the
    # bi-directional / multi-canonical case where multiple surface
    # forms are simultaneously valid. A key that is ABSENT from the
    # mapping means the cell should be DROPPED from the export
    # (the LLM proposed a fix but the operator rejected it). When
    # this dict is non-empty the export pipeline uses it verbatim
    # in lieu of calling the LLM, so a saved value here is treated
    # as the operator's final word.
    value_corrections: Dict[str, Union[str, List[str]]] = Field(default_factory=dict)
    note: str = ""
    # Annotator defaults to the server's --annotator flag if omitted.
    annotator: Optional[str] = None


class Label(LabelIn):
    """Persisted label record (LabelIn + server-assigned fields)."""

    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LabelStats(BaseModel):
    """Per-status counts for a slice of clusters/issues."""

    total: int
    tp: int = 0
    fp: int = 0
    partial: int = 0
    unsure: int = 0
    skip: int = 0
    unlabeled: int = 0
    fn: int = 0


class BenchmarkCell(BaseModel):
    """One row of the exported benchmark (per-cell ground truth).

    Unified across all error types (typo, semantic_incompatible,
    inconsistency, row_duplication, ...). The ``corrected_value``
    field carries the operator's ground-truth correction, with a
    polymorphic shape that encodes the correction's *direction*
    implicitly so a separate ``direction`` field is not needed:

      * ``None``      — no correction (typically ``label="clean"``,
                       or an inconsistency cell whose row was
                       skipped, etc.).
      * ``str``       — ONE-DIRECTIONAL: a single canonical value
                       the cell should be rewritten to. Used by typo
                       (the operator-supplied fix) and by
                       inconsistency cells whose cluster has an
                       operator-confirmed canonical.
      * ``List[str]`` — BI-DIRECTIONAL: any value in the list is
                       acceptable. Used by inconsistency cells whose
                       cluster has no confirmed canonical — the
                       downstream cleaner may map any cell to any
                       value in the cluster, including
                       ``original_value`` itself.

    Downstream evaluators detect the direction with
    ``isinstance(corrected_value, list)``.
    """

    source: str
    row: int
    column: str
    original_value: str
    label: str                       # error | clean | ambiguous
    # Polymorphic correction; see class docstring for direction
    # semantics. Bi-directional clusters serialise as a JSON array;
    # one-directional fixes as a JSON string; ``null`` for no fix.
    corrected_value: Union[str, List[str], None] = None
    # Concrete replacement value the detector / synthesiser proposed
    # for THIS cell, captured at label time. Independent of the
    # operator's verdict: an FP cell still records the LLM's
    # (rejected) suggestion so downstream callers can A/B compare
    # human vs LLM corrections without re-loading the run's clusters.
    # For inconsistency cells this carries the LLM's *proposed*
    # canonical (the cluster's ``llm_proposed_canonical``), which is
    # preserved here as a hint independent of whether the operator
    # confirmed it. ``None`` when no detector / proposer produced a
    # usable suggestion.
    llm_suggested_value: Optional[str] = None
    error_types: List[str] = Field(default_factory=list)
    detection_ids: List[str] = Field(default_factory=list)
    cluster_ids: List[str] = Field(default_factory=list)
    # Inconsistency sub-pattern tag(s) for the contributing
    # cluster(s). Empty for typo / semantic_incompatible / row-level
    # findings. Mirrors ``cluster_ids`` order so downstream consumers
    # can zip them.
    cluster_kinds: List[str] = Field(default_factory=list)
    annotators: List[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------
# Multi-table inconsistency labels.
#
# Independent of the cluster/cell label vocabulary above. The flow is:
# the relation-discovery pipeline emits one ``violation_id`` per cross-
# sheet finding (1+ rows per finding, all sharing that id); the human
# reviewer compares the linked rows side-by-side and writes ONE verdict
# per ``violation_id``. Statuses match the LLM verifier verdicts
# verbatim so users can directly agree/disagree with the LLM's call.
# ---------------------------------------------------------------------


class MultiTableLabelStatus(str, Enum):
    """Annotator verdict on a multi-table inconsistency finding."""

    CONFIRMED_INCONSISTENCY = "confirmed_inconsistency"  # real cross-sheet error
    BENIGN_VARIATION = "benign_variation"                # acceptable variant
    NOT_A_VIOLATION = "not_a_violation"                  # false alarm
    UNSURE = "unsure"                                    # needs second opinion


class MultiTableLabelIn(BaseModel):
    """POST /api/runs/{run_id}/multi-table-labels payload."""

    violation_id: str                # shared across all linked rows
    workbook: str = ""               # for filtering / context only
    group_id: str = ""
    kind: str = ""                   # e.g. duplicate_row, attribute_conflict
    status: MultiTableLabelStatus
    note: str = ""
    annotator: Optional[str] = None


class MultiTableLabel(MultiTableLabelIn):
    """Persisted multi-table label record."""

    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------
# Exclusions: per-source decisions to skip whole CSVs / rows / columns
# during labelling.
#
# Independent of the cluster/cell label vocabulary above. The flow:
# the operator marks a CSV file (or specific rows / columns within it)
# as too messy to label; the run keeps an append-only record of those
# decisions in ``<run_dir>/exclusions.jsonl`` so a later step can
# reconstruct a sanitised dataset by removing the excluded slices.
# Soft-delete (tombstoning) is supported via an explicit ``deleted``
# flag on a follow-up record with the same ``id`` so the undo path
# stays append-only - no rewriting prior history.
# ---------------------------------------------------------------------


class ExclusionScope(str, Enum):
    """What part of a source CSV is being excluded."""

    CSV = "csv"          # the entire file - rows/columns must be empty
    ROWS = "rows"        # specific row indices (uses .rows)
    COLUMNS = "columns"  # specific column names (uses .columns)


class ExclusionIn(BaseModel):
    """POST /api/runs/{run_id}/exclusions payload."""

    scope: ExclusionScope
    source: str                            # CSV basename
    rows: List[int] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    note: str = ""
    annotator: Optional[str] = None


class Exclusion(ExclusionIn):
    """Persisted exclusion record."""

    id: str                                # uuid - stable across tombstoning
    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Tombstone marker - a follow-up record with the same ``id`` and
    # ``deleted=True`` retracts the prior exclusion. The store reduces
    # the JSONL to "active ids" by latest-wins so the API can present
    # an undo-able list without rewriting history.
    deleted: bool = False


# ---------------------------------------------------------------------
# Per-cell note overrides.
#
# A cluster label's ``note`` applies to every cell the cluster covers,
# which is the right default but blocks the operator from annotating
# a single cell differently from its siblings. ``CellNote`` records
# attach a free-text note to a single (source, row, column) triple;
# the exporter overlays them on top of cluster-derived notes so the
# emitted benchmark cell carries the cell-specific note instead.
#
# Append-only, latest-wins per (source, row, column). An empty
# ``note`` (after stripping) is a tombstone meaning "remove the
# override; fall back to the cluster note".
#
# File: <run_dir>/cell_notes.jsonl
# ---------------------------------------------------------------------


class CellNoteIn(BaseModel):
    """POST /api/runs/{run_id}/cell-notes payload."""

    source: str
    row: int
    column: str
    note: str = ""
    annotator: Optional[str] = None


class CellNote(CellNoteIn):
    """Persisted cell-note override record."""

    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------
# Column-clusters labels: standardization labelling for inconsistency.
#
# A separate labelling surface from the per-cluster cluster/cell vocabulary
# above. The unit of labelling is one (source, column) pair, and each
# record carries a disjoint set of value-clusters where every cluster has
# >= 2 distinct values that should be standardized together. Direction
# (bi- vs one-directional) is intentionally NOT recorded here -- the
# scorer derives it at export time from cluster frequencies and an
# optional canonical_value, so re-sweeping the dominance threshold T
# never requires re-labelling.
#
# File: <run_dir>/column_clusters.jsonl
# Reduce rule: latest record per (source, column) wins.
# ---------------------------------------------------------------------


class ClusterKind(str, Enum):
    """Inconsistency sub-pattern tag for a value cluster.

    Aligned with the single-column Inconsistency taxonomy (see
    ``data/error_type.csv``). Each tag names which kind of surface-
    form variance the cluster's values exhibit; downstream scorers
    default to ``TEXT`` semantics when the tag is ``UNTAGGED``. The
    LLM cluster-proposer is allowed to answer ``UNTAGGED`` when
    uncertain rather than picking a wrong tag.

    Mapping (operator + downstream fix-application semantics):
      * text              -- different surface forms of the same
                              real-world entity (aliases, alternate
                              names, abbreviations). E.g. "USA" vs.
                              "United States". Rewrite cell to one
                              canonical string.
      * casing            -- values differ only in letter case. E.g.
                              "james smith" vs. "James Smith". Apply
                              a deterministic column-wide casing rule.
      * pattern           -- values follow different structural
                              templates for the same content. E.g.
                              "800-123-4567" vs. "8001234567". Rewrite
                              into a canonical shape, keeping the
                              row-specific tokens.
      * punctuation       -- values differ only in punctuation /
                              whitespace / separators. E.g. "123-456"
                              vs. "123–456" (hyphen vs en-dash) or
                              extra spaces. Apply a deterministic
                              normalisation.
      * numerical_format  -- values encode the same number in
                              different ways (units, percent vs.
                              decimal, thousand separators). E.g.
                              "15%" vs. "0.15".
      * untagged          -- the operator can identify the group but
                              none of the above tags fits cleanly. The
                              scorer defaults to TEXT semantics here.

    Legacy tags (``entity_variant``, ``template_shape``,
    ``style_convention``, ``encoding_convention``) are accepted on
    input and mapped to the new tags via
    :meth:`ClusterKind.from_string` so persisted history keeps
    parsing.
    """

    UNTAGGED = "untagged"
    TEXT = "text"
    CASING = "casing"
    PATTERN = "pattern"
    PUNCTUATION = "punctuation"
    NUMERICAL_FORMAT = "numerical_format"

    @classmethod
    def from_string(cls, raw: object) -> "ClusterKind":
        """Best-effort parse of a kind string with legacy mapping.

        Returns :attr:`UNTAGGED` for ``None`` / non-string / unknown
        inputs so the caller never has to defend against malformed
        LLM output or legacy JSON. Legacy tags are mapped to their
        closest new-taxonomy equivalent so old drafts and labels
        remain parseable across the migration.
        """
        if not isinstance(raw, str):
            return cls.UNTAGGED
        key = raw.strip().lower()
        if not key:
            return cls.UNTAGGED
        # Fast path: already a current tag.
        try:
            return cls(key)
        except ValueError:
            pass
        # Legacy taxonomy -> new taxonomy. Best-effort mapping; the
        # old tag space wasn't a strict refinement of the new one
        # (e.g. style_convention covered case + separators + token
        # order at once), so we pick the dominant interpretation.
        legacy = _LEGACY_KIND_MAPPING.get(key)
        if legacy is not None:
            return legacy
        return cls.UNTAGGED


# Legacy -> new ClusterKind mapping. Persisted drafts written before
# the taxonomy alignment use the old tag space; this table keeps them
# parseable. Defined at module scope so it's importable for tests +
# the frontend's mirror copy.
_LEGACY_KIND_MAPPING: Dict[str, ClusterKind] = {
    "entity_variant": ClusterKind.TEXT,
    "template_shape": ClusterKind.PATTERN,
    "style_convention": ClusterKind.CASING,
    "encoding_convention": ClusterKind.NUMERICAL_FORMAT,
}


class ColumnLabelStatus(str, Enum):
    """Per-(source, column) labelling status."""

    UNLABELED = "unlabeled"                  # initial state; nothing saved
    LABELED = "labeled"                       # at least one cluster recorded
    NO_INCONSISTENCY = "no_inconsistency"     # column verified clean
    DROPPED_OVERLAPPING = "dropped_overlapping"
    DROPPED_MESSY = "dropped_messy"


class ClusterValue(BaseModel):
    """One distinct value inside a standardization cluster."""

    value: str
    count: int = Field(ge=1)                  # occurrences in the column


class StdCluster(BaseModel):
    """One value-cluster within a labelled column.

    Invariants (enforced):
      * ``values`` has length >= 2.
      * Within ``values``, every ``value`` string is unique.
      * When ``canonical_value`` is set, it must equal one of the
        cluster's ``ClusterValue.value`` entries.
      * When ``llm_proposed_canonical`` is set, it should equal one
        of the cluster's ``ClusterValue.value`` entries (validated
        leniently \u2014 stale proposals are nulled rather than
        rejecting the whole cluster).
    """

    cluster_id: str                            # "c1" .. "cN" within the record
    kind: ClusterKind = ClusterKind.UNTAGGED
    # Operator-confirmed canonical. Only set after a human reviewer
    # endorsed one of the cluster's surface forms; ``None`` means the
    # cluster is bi-directional (downstream may map any cell to any
    # other value in ``values``).
    canonical_value: Optional[str] = None     # null => no value is canonical
    # LLM-proposed canonical (the cluster proposer's hint). Carried
    # through the synthesis pipeline so the exported benchmark can
    # surface it as ``llm_suggested_value`` without re-reading the
    # drafts file. Independent of ``canonical_value``: a TP cluster
    # can have ``llm_proposed_canonical="California"`` even when the
    # operator did not confirm any canonical, in which case the cell
    # is bi-directional but downstream still gets the LLM's hint.
    llm_proposed_canonical: Optional[str] = None
    values: List[ClusterValue]

    # Accept legacy kind tags (entity_variant, template_shape, ...)
    # and remap them to the new taxonomy. Without this, persisted
    # column_clusters.jsonl / column_cluster_drafts.jsonl written
    # before the migration would fail validation on load.
    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v: object) -> ClusterKind:
        if isinstance(v, ClusterKind):
            return v
        return ClusterKind.from_string(v)

    @model_validator(mode="after")
    def _check_cluster(self) -> "StdCluster":
        if len(self.values) < 2:
            raise ValueError(
                f"cluster {self.cluster_id!r}: must have >= 2 distinct "
                f"values (got {len(self.values)})"
            )
        seen: set = set()
        for cv in self.values:
            if cv.value in seen:
                raise ValueError(
                    f"cluster {self.cluster_id!r}: duplicate value "
                    f"{cv.value!r} inside cluster"
                )
            seen.add(cv.value)
        if self.canonical_value is not None and self.canonical_value not in seen:
            raise ValueError(
                f"cluster {self.cluster_id!r}: canonical_value "
                f"{self.canonical_value!r} not present in cluster.values"
            )
        # Lenient on ``llm_proposed_canonical``: stale LLM hints
        # (e.g. drafts written before the cluster's value-set was
        # trimmed) get nulled rather than rejecting the whole
        # cluster. The hint is advisory only.
        if (
            self.llm_proposed_canonical is not None
            and self.llm_proposed_canonical not in seen
        ):
            self.llm_proposed_canonical = None
        return self


class ColumnClustersIn(BaseModel):
    """POST /api/runs/{run_id}/columns/{source}/{column}/clusters payload.

    Cross-cluster invariants (enforced):
      * No value string appears in two clusters (disjoint partition of
        the labelled subset of the column's distinct values).
      * Every ``cluster_id`` is unique within the record.
      * ``status=labeled`` requires ``clusters`` to be non-empty.
      * ``status`` in {no_inconsistency, dropped_*, unlabeled} requires
        ``clusters`` to be empty.
    """

    source: str                                # CSV basename
    column: str                                # column header
    status: ColumnLabelStatus
    drop_reason: str = ""                      # free text; only used when status=dropped_*
    clusters: List[StdCluster] = Field(default_factory=list)
    note: str = ""
    annotator: Optional[str] = None

    @model_validator(mode="after")
    def _check_record(self) -> "ColumnClustersIn":
        # Status / clusters consistency.
        if self.status == ColumnLabelStatus.LABELED:
            if not self.clusters:
                raise ValueError(
                    "status=labeled requires at least one cluster"
                )
        else:
            if self.clusters:
                raise ValueError(
                    f"status={self.status.value!r} requires no clusters; "
                    f"got {len(self.clusters)}"
                )
        # Unique cluster_ids within the record.
        seen_ids: set = set()
        for cl in self.clusters:
            if cl.cluster_id in seen_ids:
                raise ValueError(
                    f"duplicate cluster_id {cl.cluster_id!r} in record"
                )
            seen_ids.add(cl.cluster_id)
        # Disjoint values across clusters.
        seen_values: Dict[str, str] = {}
        for cl in self.clusters:
            for cv in cl.values:
                prior = seen_values.get(cv.value)
                if prior is not None and prior != cl.cluster_id:
                    raise ValueError(
                        f"value {cv.value!r} appears in two clusters: "
                        f"{prior!r} and {cl.cluster_id!r}"
                    )
                seen_values[cv.value] = cl.cluster_id
        return self


class ColumnClusters(ColumnClustersIn):
    """Persisted record: ``ColumnClustersIn`` + server-assigned fields."""

    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


