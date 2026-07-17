#!/usr/bin/env bash
# Full-coverage detection for the 15 mt100 inter-table CSVs across every
# NON-inter-table error type, mirroring the excel_500 per-type runs. Each run
# lands under outputs/ ready for labeling in the workbench:
#
#   single-column : outputs/mt100_<type>/
#   relation      : outputs/mt100_<type>_relations/  (discovery+validation+detection in place)
#
# After all runs finish + you label them, add an `mt100` pool to
# experiments/export_merged_benchmark.py::REGISTRY and rebuild.
#
# Read-only w.r.t. existing runs. Robust: continues past a failing run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=venv/bin/python
CSV=data/mt100_intertable_15flat
MODEL="gpt-5.2,gpt-5.4,gpt-5.5"
# LLM call concurrency for every run below. The column-clustering path
# (inconsistency/semantic/numeric) defaults to serial (max_workers=1), which is
# why those types were ~5x slower than typo; override to fan out. Calls
# round-robin across the built-in ~11-endpoint pool (DEFAULT_ENDPOINTS), so ~20
# in flight is ~2 concurrent per endpoint. Override the pool by exporting
# AZURE_OPENAI_ENDPOINTS=<comma,separated,urls> before running this script.
CONCURRENCY="${CONCURRENCY:-20}"
LOGDIR=logs/mt100_fullcov
mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/RUN_MANIFEST.txt"
: > "$MANIFEST"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_singlecol () {  # name  types  mode  colmode  [scopes]
  local name="$1" types="$2" mode="$3" colmode="$4" scopes="${5:-}"
  local scope_args=()
  [ -n "$scopes" ] && scope_args=(--scopes "$scopes")
  log "single-col START: $name (mode=$mode colmode=$colmode scopes=${scopes:-default} concurrency=$CONCURRENCY)"
  $PY -m experiments.run_cleaning_detection \
      --csv-dir "$CSV" --error-types "$types" \
      --mode "$mode" --column-mode "$colmode" "${scope_args[@]}" \
      --model "$MODEL" --chunk-size 100 --concurrency "$CONCURRENCY" \
      --run-name "mt100_${name}" \
      > "$LOGDIR/${name}.log" 2>&1
  local rc=$?
  log "single-col DONE:  $name  exit=$rc  -> outputs/mt100_${name}"
  echo "${name}	single-column	outputs/mt100_${name}	exit=$rc" >> "$MANIFEST"
}

run_relation () {  # name  types
  local name="$1" types="$2"
  local disc_root="outputs/_mt100_${name}_discroot"
  rm -rf "$disc_root"; mkdir -p "$disc_root"
  log "relation DISCOVERY START: $name ($types)"
  $PY -m experiments.run_relation_discovery \
      --csv-dir "$CSV" --error-types "$types" \
      --model "$MODEL" --adaptive --concurrency "$CONCURRENCY" \
      --output-dir "$disc_root" \
      > "$LOGDIR/${name}_1discovery.log" 2>&1
  local rc=$?
  local disc; disc=$(ls -d "$disc_root"/*/ 2>/dev/null | head -1); disc="${disc%/}"
  if [ -z "$disc" ] || [ ! -d "$disc" ]; then
    log "relation DISCOVERY FAILED: $name (no run dir, exit=$rc)"
    echo "${name}	relation	DISCOVERY_FAILED	exit=$rc" >> "$MANIFEST"
    return
  fi
  # Move the discovery dir up to outputs/ under a predictable name so the
  # whole pipeline (validation + detection write in-place) is self-contained.
  local final="outputs/mt100_${name}_relations"
  rm -rf "$final"; mv "$disc" "$final"; rm -rf "$disc_root"
  log "relation DISCOVERY DONE: $name  exit=$rc  -> $final"

  log "relation VALIDATION START: $name"
  $PY -m experiments.run_relation_validation "$final" --model "$MODEL" \
      --concurrency "$CONCURRENCY" \
      > "$LOGDIR/${name}_2validation.log" 2>&1
  log "relation VALIDATION DONE:  $name  exit=$?"

  log "relation DETECTION START: $name"
  $PY -m experiments.run_relation_detection "$final" \
      --error-types "$types" --model "$MODEL" \
      --concurrency "$CONCURRENCY" \
      --review-filter approved+unreviewed \
      --run-name "mt100_${name}_relations" \
      > "$LOGDIR/${name}_3detection.log" 2>&1
  local rcd=$?
  log "relation DETECTION DONE:  $name  exit=$rcd  -> $final"
  echo "${name}	relation	$final	exit=$rcd" >> "$MANIFEST"
}

log "=== mt100 full-coverage detection: launching 7 runs ==="

# --- single-column-ish types (each one run_cleaning_detection) --------
run_singlecol typo                  typo                  adaptive single
run_singlecol inconsistency         inconsistency         single   single
run_singlecol semantic_incompatible semantic_incompatible adaptive single
run_singlecol numeric_outlier       numeric_outlier       adaptive single
run_singlecol row_duplication       row_duplication       adaptive auto     multi-row

# --- relation types (discovery -> validation -> detection) ------------
run_relation fd_violation fd_violation
run_relation calc         row_calculation_inconsistency,string_calculation_inconsistency

log "=== ALL RUNS COMPLETE ==="
echo ""
echo "RUN MANIFEST ($MANIFEST):"
cat "$MANIFEST"
