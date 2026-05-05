"""Offline hyperparameter search over the per-arm stop predicate.

Workflow:

1. Run :mod:`experiments.run_cleaning_detection` once with
   ``--gt-no-stop --gt-num-runs K_max`` on a representative dataset.
   That produces ``<run_dir>/good_turing/per_pass_per_arm.jsonl``,
   one row per ``(pass_idx, arm)`` with the full estimator state
   computed from the cumulative aggregation up through that pass.
2. Run this script against the run directory:

   .. code-block:: bash

      python -m experiments.analyze_stop_thresholds \\
          outputs/<RUN_ID> \\
          [--out-dir <RUN_ID>/threshold_sweep] \\
          [--epsilon-grid 0.01,0.02,0.03,0.05,0.08,0.10] \\
          [--epsilon-chao1-grid 0.02,0.05,0.08,0.10,0.15,0.20] \\
          [--k-min-grid 3,4,5] \\
          [--two-consec true,false]

3. The script replays the per-arm stop predicate offline for every
   ``(stop_when, epsilon, epsilon_chao1, k_min, two_consec)`` tuple
   in the configured grid and writes:

   * ``threshold_sweep.csv``    -- one row per configuration with
                                   summary metrics (saturation rate,
                                   mean/median/p95 stop pass, recall
                                   vs.  K_max, regret).
   * ``threshold_sweep.json``   -- same metrics as a structured dict
                                   plus the raw per-arm stop-pass
                                   table (for re-plotting).
   * ``threshold_sweep.md``     -- human-readable report.
   * ``pareto_<stop_when>.png`` -- one Pareto plot per stop_when
                                   mode (passes saved vs. species
                                   recall).

Replay is purely arithmetic on the JSONL trace, so the sweep does
NOT spend any LLM tokens. A grid of 100+ configurations finishes in
a fraction of a second.

The ground truth for "what would the experiment have found" is the
final (K_max) row for each arm; "regret" of a candidate threshold is
the gap between what the arm saw at its (replayed) stop pass and
what it would have seen at K_max. We report two recall metrics:

* **species recall** = ``S_obs(stop) / S_obs(K_max)`` -- what
  fraction of the species we'd have observed if we'd run all the
  way through.
* **mass recall**    = ``N(stop) / N(K_max)`` -- raw emission share
  retained at stop. (Not a guaranteed monotone function of stop
  pass when the LLM produces variable per-pass volume, but a useful
  cross-check.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Local imports (project root on sys.path - matches build_report.py shape).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.scoped_gt import StopWhen  # noqa: E402

logger = logging.getLogger("analyze_stop_thresholds")


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


# An ``ArmTrace`` is the ordered sequence of per-pass snapshots for a
# single arm, indexed by ``runs_so_far`` (1-based). We use a list
# so callers can iterate in order without sorting per replay.
ArmKey = Tuple[str, Tuple[str, ...], str]  # (scope_kind, scope_id, error_type)


def _arm_key(row: Mapping[str, Any]) -> ArmKey:
    return (
        str(row["scope_kind"]),
        tuple(str(x) for x in (row.get("scope_id") or [])),
        str(row["error_type"]),
    )


def load_per_pass_traces(
    jsonl_path: Path,
) -> Tuple[Dict[ArmKey, List[Dict[str, Any]]], int]:
    """Load the per-pass per-arm dump into ``{arm: [rows ordered by pass]}``.

    Returns ``(traces, K_max)`` where ``K_max`` is the maximum
    ``runs_so_far`` observed across all arms (the experiment's
    effective K). Rows within each arm are sorted by ``runs_so_far``.
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"per-pass per-arm trace not found at {jsonl_path}. "
            "Did you run with --gt-no-stop and a recent build?"
        )
    traces: Dict[ArmKey, List[Dict[str, Any]]] = defaultdict(list)
    k_max = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            traces[_arm_key(row)].append(row)
            k_max = max(k_max, int(row.get("runs_so_far", 0)))
    for rows in traces.values():
        rows.sort(key=lambda r: int(r["runs_so_far"]))
    return dict(traces), k_max


# ---------------------------------------------------------------------------
# Predicate replay (matches src.adaptive.scoped_gt.evaluate_stop semantics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdConfig:
    stop_when: StopWhen
    epsilon: float          # GT mass tolerance
    epsilon_chao1: float    # Chao1 upper-CI count fraction tolerance
    k_min: int
    k_max: int
    require_two_consecutive: bool

    def label(self) -> str:
        return (
            f"stop={self.stop_when}|"
            f"eps={self.epsilon:.3f}|"
            f"eps_c={self.epsilon_chao1:.3f}|"
            f"kmin={self.k_min}|"
            f"two={'T' if self.require_two_consecutive else 'F'}"
        )


def _gt_ok(row: Mapping[str, Any], epsilon: float) -> bool:
    return float(row.get("good_turing_missing_mass", 0.0)) <= epsilon


def _chao1_ok(row: Mapping[str, Any], epsilon_chao1: float) -> bool:
    s_obs = float(row.get("n_observed", 0))
    if s_obs <= 0:
        return False  # degenerate; fail safe
    f0_upper = max(
        0.0,
        float(row.get("chao1_ci_95_high", 0.0)) - s_obs,
    )
    return (f0_upper / s_obs) <= epsilon_chao1


def _criterion_ok(row: Mapping[str, Any], cfg: ThresholdConfig) -> bool:
    if cfg.stop_when == "gt":
        return _gt_ok(row, cfg.epsilon)
    if cfg.stop_when == "chao1":
        return _chao1_ok(row, cfg.epsilon_chao1)
    if cfg.stop_when == "both":
        return _gt_ok(row, cfg.epsilon) and _chao1_ok(row, cfg.epsilon_chao1)
    raise ValueError(f"unknown stop_when: {cfg.stop_when!r}")


def replay_arm(
    rows: Sequence[Mapping[str, Any]],
    cfg: ThresholdConfig,
) -> Tuple[Optional[int], str]:
    """Replay the stop predicate for one arm under ``cfg``.

    Returns ``(stop_pass, status)``:

    * ``stop_pass`` -- ``runs_so_far`` value at which the arm would
      have saturated (1-based), or ``None`` if it would have run to
      ``k_max`` without satisfying the predicate.
    * ``status`` -- ``"saturated"`` if the predicate fired at or
      before ``k_max``; ``"exhausted"`` otherwise.

    Mirrors :func:`src.adaptive.scoped_gt.evaluate_stop` exactly:

    * ``num_runs >= k_min`` floor on saturation.
    * Streak gate: ``require_two_consecutive`` ? 2 : 1 consecutive
      passes meeting the criterion.
    * ``num_runs >= k_max`` -> ``exhausted``.
    """
    streak = 0
    threshold = 2 if cfg.require_two_consecutive else 1
    for row in rows:
        k = int(row["runs_so_far"])
        if k > cfg.k_max:
            break
        if _criterion_ok(row, cfg) and k >= cfg.k_min:
            streak += 1
            if streak >= threshold:
                return k, "saturated"
        else:
            streak = 0
        if k >= cfg.k_max:
            return None, "exhausted"
    return None, "exhausted"


# ---------------------------------------------------------------------------
# Sweep + metrics
# ---------------------------------------------------------------------------


@dataclass
class ArmReplayResult:
    arm: ArmKey
    stop_pass: Optional[int]      # None means ran to k_max
    status: str                   # "saturated" or "exhausted"
    s_obs_at_stop: int            # what we would have seen
    s_obs_at_kmax: int            # ground truth
    n_at_stop: int
    n_at_kmax: int
    p_miss_at_stop: float
    chao1_total_at_stop: float


def _evaluate_at_pass(
    rows: Sequence[Mapping[str, Any]],
    target_pass: Optional[int],
) -> Mapping[str, Any]:
    """Return the snapshot at ``target_pass`` (or the last row).

    ``target_pass`` is 1-based ``runs_so_far``; ``None`` means the
    final row (K_max) -- the ground-truth point.
    """
    if target_pass is None:
        return rows[-1]
    for row in rows:
        if int(row["runs_so_far"]) == target_pass:
            return row
    # Fallback: closest pass <= target_pass.
    candidates = [r for r in rows if int(r["runs_so_far"]) <= target_pass]
    return candidates[-1] if candidates else rows[0]


def replay_sweep(
    traces: Mapping[ArmKey, List[Dict[str, Any]]],
    cfg: ThresholdConfig,
    *,
    k_max_global: int,
) -> Dict[str, Any]:
    """Replay ``cfg`` against all arms and compute summary metrics.

    Returns a dict with:

    * ``cfg``                -- the threshold config (label form).
    * ``per_arm``            -- list of :class:`ArmReplayResult` dicts.
    * ``num_arms``           -- total arms in the trace.
    * ``num_saturated``      -- arms that would have stopped early.
    * ``saturation_rate``    -- ``num_saturated / num_arms``.
    * ``stop_pass_mean``     -- mean stop pass over all arms (``k_max``
                                 imputed for non-saturating arms).
    * ``stop_pass_median``   -- median stop pass.
    * ``stop_pass_p95``      -- 95th percentile.
    * ``passes_saved_total`` -- ``sum(k_max - stop_pass)`` over all
                                arms; the absolute "LLM passes saved"
                                budget across the run.
    * ``passes_saved_mean``  -- per-arm average savings.
    * ``species_recall_mean``-- mean of S_obs(stop) / S_obs(k_max).
    * ``species_recall_p05`` -- 5th percentile (worst-case arms).
    * ``mass_recall_mean``   -- mean of N(stop) / N(k_max).
    * ``mass_recall_p05``    -- 5th percentile.
    """
    per_arm: List[ArmReplayResult] = []
    for arm, rows in traces.items():
        stop_pass, status = replay_arm(rows, cfg)
        snap_stop = _evaluate_at_pass(rows, stop_pass)
        snap_kmax = rows[-1]
        per_arm.append(
            ArmReplayResult(
                arm=arm,
                stop_pass=stop_pass,
                status=status,
                s_obs_at_stop=int(snap_stop.get("n_observed", 0)),
                s_obs_at_kmax=int(snap_kmax.get("n_observed", 0)),
                n_at_stop=int(snap_stop.get("total_hits", 0)),
                n_at_kmax=int(snap_kmax.get("total_hits", 0)),
                p_miss_at_stop=float(
                    snap_stop.get("good_turing_missing_mass", 0.0)
                ),
                chao1_total_at_stop=float(
                    snap_stop.get("chao1_estimated_total", 0.0)
                ),
            )
        )

    num_arms = len(per_arm)
    if num_arms == 0:
        return {
            "cfg": cfg.label(),
            "num_arms": 0,
        }

    saturated = [r for r in per_arm if r.status == "saturated"]
    # Impute k_max for non-saturating arms so summary stats are
    # comparable across configurations of different strictness.
    stop_passes = [
        (r.stop_pass if r.stop_pass is not None else k_max_global)
        for r in per_arm
    ]
    saved = [k_max_global - p for p in stop_passes]

    def _safe_recall(num: int, denom: int) -> float:
        # Arms that observed nothing at K_max contribute neutrally
        # (we give them a recall of 1.0 since there was nothing to
        # miss) - otherwise an empty-arm-heavy trace would skew the
        # mean down for every configuration uniformly.
        if denom <= 0:
            return 1.0
        return min(1.0, num / denom)

    sp_recall = [
        _safe_recall(r.s_obs_at_stop, r.s_obs_at_kmax) for r in per_arm
    ]
    mass_recall = [
        _safe_recall(r.n_at_stop, r.n_at_kmax) for r in per_arm
    ]

    return {
        "cfg": cfg.label(),
        "stop_when": cfg.stop_when,
        "epsilon": cfg.epsilon,
        "epsilon_chao1": cfg.epsilon_chao1,
        "k_min": cfg.k_min,
        "k_max": cfg.k_max,
        "require_two_consecutive": cfg.require_two_consecutive,
        "num_arms": num_arms,
        "num_saturated": len(saturated),
        "saturation_rate": len(saturated) / num_arms,
        "stop_pass_mean": _mean(stop_passes),
        "stop_pass_median": _percentile(stop_passes, 50),
        "stop_pass_p95": _percentile(stop_passes, 95),
        "passes_saved_total": sum(saved),
        "passes_saved_mean": _mean(saved),
        "species_recall_mean": _mean(sp_recall),
        "species_recall_p05": _percentile(sp_recall, 5),
        "mass_recall_mean": _mean(mass_recall),
        "mass_recall_p05": _percentile(mass_recall, 5),
        "per_arm": [
            {
                "arm": r.arm,
                "stop_pass": r.stop_pass,
                "status": r.status,
                "s_obs_at_stop": r.s_obs_at_stop,
                "s_obs_at_kmax": r.s_obs_at_kmax,
                "n_at_stop": r.n_at_stop,
                "n_at_kmax": r.n_at_kmax,
                "p_miss_at_stop": r.p_miss_at_stop,
                "chao1_total_at_stop": r.chao1_total_at_stop,
                "species_recall": _safe_recall(
                    r.s_obs_at_stop, r.s_obs_at_kmax
                ),
                "mass_recall": _safe_recall(r.n_at_stop, r.n_at_kmax),
            }
            for r in per_arm
        ],
    }


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / len(xs)


def _percentile(xs: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile; robust on small samples."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (pct / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_bool_list(s: str) -> List[bool]:
    out: List[bool] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif t in ("0", "false", "f", "no", "n"):
            out.append(False)
    return out


def _parse_stopwhen_list(s: str) -> List[StopWhen]:
    valid = {"gt", "chao1", "both"}
    out: List[StopWhen] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in valid:
            out.append(t)  # type: ignore[arg-type]
    return out


def expand_grid(
    *,
    stop_whens: Sequence[StopWhen],
    epsilons: Sequence[float],
    epsilon_chao1s: Sequence[float],
    k_mins: Sequence[int],
    two_consecs: Sequence[bool],
    k_max: int,
) -> List[ThresholdConfig]:
    """Cartesian product, with one optimisation: when ``stop_when`` is
    ``"gt"`` the ``epsilon_chao1`` grid is irrelevant (the predicate
    ignores it), so we collapse it to a single representative value;
    same for ``"chao1"`` w.r.t. ``epsilon``. This avoids logging
    visually identical rows in the CSV output.
    """
    cfgs: List[ThresholdConfig] = []
    for sw in stop_whens:
        eps_iter = epsilons if sw in ("gt", "both") else (epsilons[:1] or [0.0])
        epsc_iter = (
            epsilon_chao1s if sw in ("chao1", "both")
            else (epsilon_chao1s[:1] or [0.0])
        )
        for eps in eps_iter:
            for epsc in epsc_iter:
                for kmin in k_mins:
                    for two in two_consecs:
                        cfgs.append(ThresholdConfig(
                            stop_when=sw,
                            epsilon=float(eps),
                            epsilon_chao1=float(epsc),
                            k_min=int(kmin),
                            k_max=int(k_max),
                            require_two_consecutive=bool(two),
                        ))
    return cfgs


# ---------------------------------------------------------------------------
# Output: CSV / JSON / Markdown
# ---------------------------------------------------------------------------


_SUMMARY_COLS = [
    "stop_when",
    "epsilon",
    "epsilon_chao1",
    "k_min",
    "k_max",
    "require_two_consecutive",
    "num_arms",
    "num_saturated",
    "saturation_rate",
    "stop_pass_mean",
    "stop_pass_median",
    "stop_pass_p95",
    "passes_saved_total",
    "passes_saved_mean",
    "species_recall_mean",
    "species_recall_p05",
    "mass_recall_mean",
    "mass_recall_p05",
]


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_SUMMARY_COLS)
        for r in rows:
            w.writerow([r.get(k, "") for k in _SUMMARY_COLS])


def write_json(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    # Drop the per-arm detail from the summary so the JSON file
    # stays small. Per-arm tables are huge under wide grids.
    summary = [{k: r.get(k) for k in _SUMMARY_COLS} for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sweep": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_markdown(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Stop-threshold sweep\n")
    lines.append(
        "Each row is one ``(stop_when, epsilon, epsilon_chao1, "
        "k_min, two_consec)`` configuration replayed offline against "
        "a forced-K_max trace. ``saturation_rate`` is the fraction of "
        "arms that would have been pruned; ``species_recall`` is the "
        "share of K_max-truth species that would have been observed "
        "by the time of stop. Higher is better for both, but a "
        "trade-off is expected.\n"
    )
    lines.append("| stop_when | eps | eps_c | k_min | two | sat % |"
                 " mean K | p95 K | passes saved | sp recall (p05) |"
                 " mass recall (p05) |")
    lines.append("|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['stop_when']} | {r['epsilon']:.3f} | "
            f"{r['epsilon_chao1']:.3f} | {r['k_min']} | "
            f"{'T' if r['require_two_consecutive'] else 'F'} | "
            f"{100.0 * r['saturation_rate']:.1f}% | "
            f"{r['stop_pass_mean']:.2f} | {r['stop_pass_p95']:.1f} | "
            f"{r['passes_saved_total']} | "
            f"{r['species_recall_mean']:.3f} "
            f"({r['species_recall_p05']:.3f}) | "
            f"{r['mass_recall_mean']:.3f} "
            f"({r['mass_recall_p05']:.3f}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pareto_plots(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> None:
    """One Pareto scatter per ``stop_when`` mode.

    x-axis: passes saved per arm (mean).
    y-axis: species recall (mean).

    Higher-and-rightward is better; the upper-right envelope is the
    Pareto frontier. We write one PNG per mode under ``out_dir`` so
    they're easy to drop into a notebook / report.

    Skipped silently if matplotlib is unavailable - the CSV / MD /
    JSON outputs are the load-bearing artifacts; plots are gravy.
    """
    try:
        import matplotlib  # noqa: WPS433
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping Pareto plots. "
            "Install with `pip install matplotlib` to enable."
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    by_mode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[str(r["stop_when"])].append(r)
    for mode, group in by_mode.items():
        if not group:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        xs = [r["passes_saved_mean"] for r in group]
        ys = [r["species_recall_mean"] for r in group]
        # Colour by epsilon (or epsilon_chao1 in chao1 mode) for a
        # quick visual of which side of the grid is winning.
        if mode == "chao1":
            cs = [r["epsilon_chao1"] for r in group]
            cbar_label = "epsilon_chao1"
        else:
            cs = [r["epsilon"] for r in group]
            cbar_label = "epsilon"
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=60, alpha=0.85)
        ax.set_xlabel("Passes saved per arm (mean)")
        ax.set_ylabel("Species recall vs. K_max (mean)")
        ax.set_title(f"Stop-threshold Pareto -- stop_when = {mode!r}")
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label)
        # Annotate every fifth point with its k_min / two-consec
        # tuple so the user can spot the regime in a glance.
        for i, r in enumerate(group):
            if i % max(1, len(group) // 8) != 0:
                continue
            ax.annotate(
                f"k={r['k_min']}|{'2' if r['require_two_consecutive'] else '1'}",
                (xs[i], ys[i]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        fig.tight_layout()
        out = out_dir / f"pareto_{mode}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        logger.info("wrote %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Offline hyperparameter search over the per-arm stop "
            "predicate. Replays a forced-K_max run trace against a "
            "grid of (stop_when, epsilon, epsilon_chao1, k_min, "
            "require_two_consecutive) tuples and writes summary "
            "metrics + Pareto plots."
        ),
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Path to the detection run directory (the one containing "
            "``good_turing/per_pass_per_arm.jsonl``)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the sweep artifacts. Defaults to "
            "``<run_dir>/threshold_sweep``."
        ),
    )
    p.add_argument(
        "--stop-when",
        default="gt,chao1,both",
        help="Comma-separated list of stop_when modes to sweep.",
    )
    p.add_argument(
        "--epsilon-grid",
        default="0.01,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated GT mass thresholds.",
    )
    p.add_argument(
        "--epsilon-chao1-grid",
        default="0.02,0.05,0.08,0.10,0.15,0.20",
        help="Comma-separated Chao1 missing-count fraction thresholds.",
    )
    p.add_argument(
        "--k-min-grid",
        default="3,4,5",
        help="Comma-separated k_min floors.",
    )
    p.add_argument(
        "--two-consec",
        default="true,false",
        help=(
            "Comma-separated booleans for the two-consecutive gate. "
            "Use 'true,false' to sweep both."
        ),
    )
    p.add_argument(
        "--k-max",
        type=int,
        default=None,
        help=(
            "Override k_max for the replay. Defaults to the max "
            "``runs_so_far`` observed in the trace."
        ),
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Pareto PNG output (useful in headless CI).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    run_dir: Path = args.run_dir.resolve()
    trace_path = run_dir / "good_turing" / "per_pass_per_arm.jsonl"
    traces, k_max_observed = load_per_pass_traces(trace_path)
    if not traces:
        logger.error("no arms in %s", trace_path)
        return 2

    k_max = int(args.k_max) if args.k_max is not None else int(k_max_observed)
    logger.info(
        "loaded %d arm(s) over up to K_max=%d pass(es) from %s",
        len(traces),
        k_max_observed,
        trace_path,
    )

    cfgs = expand_grid(
        stop_whens=_parse_stopwhen_list(args.stop_when),
        epsilons=_parse_float_list(args.epsilon_grid),
        epsilon_chao1s=_parse_float_list(args.epsilon_chao1_grid),
        k_mins=_parse_int_list(args.k_min_grid),
        two_consecs=_parse_bool_list(args.two_consec),
        k_max=k_max,
    )
    if not cfgs:
        logger.error("expanded grid is empty; check your --*-grid args")
        return 2
    logger.info("sweeping %d configuration(s)...", len(cfgs))

    rows: List[Dict[str, Any]] = []
    for cfg in cfgs:
        rows.append(replay_sweep(traces, cfg, k_max_global=k_max))

    out_dir: Path = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_dir / "threshold_sweep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "threshold_sweep.csv")
    write_json(rows, out_dir / "threshold_sweep.json")
    write_markdown(rows, out_dir / "threshold_sweep.md")
    if not args.no_plots:
        write_pareto_plots(rows, out_dir)

    # Print top-K rows by species_recall_mean to stdout for quick eyeballing.
    top = sorted(rows, key=lambda r: -r["species_recall_mean"])[:10]
    print(f"\nTop 10 by species_recall_mean (out of {len(rows)}):")
    print(
        f"{'stop':<6} {'eps':>6} {'eps_c':>7} {'kmin':>4} {'two':>3} "
        f"{'sat%':>5} {'mean K':>7} {'recall':>7} {'saved':>7}"
    )
    for r in top:
        print(
            f"{r['stop_when']:<6} {r['epsilon']:>6.3f} "
            f"{r['epsilon_chao1']:>7.3f} {r['k_min']:>4} "
            f"{'T' if r['require_two_consecutive'] else 'F':>3} "
            f"{100.0 * r['saturation_rate']:>4.1f}% "
            f"{r['stop_pass_mean']:>7.2f} "
            f"{r['species_recall_mean']:>7.3f} "
            f"{r['passes_saved_mean']:>7.2f}"
        )
    print(f"\nFull report: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
