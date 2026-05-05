"""Visualisation + recommendation helper for the K_max stop-threshold sweep.

Reads the forced-K_max trace at
``<run>/good_turing/per_pass_per_arm.jsonl`` and the sweep summary at
``<run>/threshold_sweep/threshold_sweep.json`` (produced by
:mod:`experiments.analyze_stop_thresholds`) and writes a set of
plots + a recommendation markdown to ``<run>/threshold_sweep/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("sweep_visualize")
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ArmKey = Tuple[str, Tuple[str, ...], str]


def _arm_key(row: Dict[str, Any]) -> ArmKey:
    return (
        str(row["scope_kind"]),
        tuple(str(x) for x in (row.get("scope_id") or [])),
        str(row["error_type"]),
    )


def load_traces(path: Path) -> Tuple[Dict[ArmKey, List[Dict[str, Any]]], int]:
    traces: Dict[ArmKey, List[Dict[str, Any]]] = defaultdict(list)
    k_max = 0
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        traces[_arm_key(r)].append(r)
        k_max = max(k_max, int(r.get("runs_so_far", 0)))
    for v in traces.values():
        v.sort(key=lambda x: int(x["runs_so_far"]))
    return dict(traces), k_max


def chao1_ratio(row: Dict[str, Any]) -> float:
    n = float(row.get("n_observed", 0))
    if n <= 0:
        return float("inf")
    return max(0.0, float(row.get("chao1_ci_95_high", 0.0)) - n) / n


# ---------------------------------------------------------------------------
# Replay helpers (inline, mirror analyze_stop_thresholds)
# ---------------------------------------------------------------------------

def replay_arm(rows, *, stop_when, eps_gt, eps_c, k_min, k_max, two_consec):
    threshold = 2 if two_consec else 1
    streak = 0
    for r in rows:
        k = int(r["runs_so_far"])
        if k > k_max:
            break
        gt_ok = float(r.get("good_turing_missing_mass", 0.0)) <= eps_gt
        c_ok = chao1_ratio(r) <= eps_c
        if stop_when == "gt":
            ok = gt_ok
        elif stop_when == "chao1":
            ok = c_ok
        else:
            ok = gt_ok and c_ok
        if ok and k >= k_min:
            streak += 1
            if streak >= threshold:
                return k
        else:
            streak = 0
    return None  # exhausted


# ---------------------------------------------------------------------------
# Plot 1: Saturation rate vs pass index, multiple eps lines per stop_when
# ---------------------------------------------------------------------------

def plot_saturation_curves(traces, k_max, out_dir):
    configs = {
        "gt": [("eps", e) for e in [0.01, 0.02, 0.05, 0.10, 0.20]],
        "chao1": [("eps_c", e) for e in [0.05, 0.10, 0.20, 0.50]],
        "both (eps_c=0.20)": [("eps", e) for e in [0.01, 0.02, 0.05, 0.10, 0.20]],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    arms = list(traces.values())

    for ax, (mode, items) in zip(axes, configs.items()):
        for label_kind, val in items:
            sat_at_k = []
            for k in range(1, k_max + 1):
                cnt = 0
                for rows in arms:
                    if mode == "gt":
                        sp = replay_arm(rows, stop_when="gt", eps_gt=val,
                                        eps_c=1.0, k_min=3, k_max=k,
                                        two_consec=True)
                    elif mode == "chao1":
                        sp = replay_arm(rows, stop_when="chao1", eps_gt=1.0,
                                        eps_c=val, k_min=3, k_max=k,
                                        two_consec=True)
                    else:
                        sp = replay_arm(rows, stop_when="both", eps_gt=val,
                                        eps_c=0.20, k_min=3, k_max=k,
                                        two_consec=True)
                    if sp is not None:
                        cnt += 1
                sat_at_k.append(cnt / len(arms))
            ax.plot(range(1, k_max + 1), sat_at_k,
                    label=f"{label_kind}={val:.2f}")
        ax.set_title(f"stop_when = {mode}")
        ax.set_xlabel("Number of passes (K)")
        ax.set_ylabel("Saturation rate")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Fraction of arms saturated by pass K (k_min=3, two-consec)",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "saturation_vs_K.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 2: Final guarantee at K=K_max per arm
# ---------------------------------------------------------------------------

def plot_final_guarantee(traces, k_max, out_dir):
    arms = list(traces.values())
    p_miss = sorted([a[-1]["good_turing_missing_mass"] for a in arms])
    chao = sorted([min(2.0, chao1_ratio(a[-1])) for a in arms])  # cap inf
    n_obs = sorted([a[-1]["n_observed"] for a in arms])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(range(len(p_miss)), p_miss, color="#4c72b0")
    axes[0].axhline(0.05, ls="--", c="red", lw=1, label="ε=0.05")
    axes[0].axhline(0.10, ls="--", c="orange", lw=1, label="ε=0.10")
    axes[0].set_title(f"Good-Turing p_miss at K={k_max}")
    axes[0].set_xlabel("arm (sorted)")
    axes[0].set_ylabel("p_miss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(chao)), chao, color="#55a868")
    axes[1].axhline(0.20, ls="--", c="red", lw=1, label="ε_c=0.20")
    axes[1].axhline(0.50, ls="--", c="orange", lw=1, label="ε_c=0.50")
    axes[1].set_title(f"Chao1 (CI95-S)/S at K={k_max}")
    axes[1].set_xlabel("arm (sorted)")
    axes[1].set_ylabel("ratio")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(n_obs)), n_obs, color="#c44e52")
    axes[2].set_title(f"n_observed (species) at K={k_max}")
    axes[2].set_xlabel("arm (sorted)")
    axes[2].set_ylabel("# species")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Per-arm \"ground-truth\" state after the full K={k_max}-pass sweep "
        f"({len(arms)} arms)",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "final_guarantee_K_max.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 3: Pareto frontier on saturation% vs species_recall, sized by stop_pass
# ---------------------------------------------------------------------------

def plot_pareto_combined(sweep_rows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    by_mode = defaultdict(list)
    for r in sweep_rows:
        by_mode[r["stop_when"]].append(r)
    colors = {"gt": "#4c72b0", "chao1": "#55a868", "both": "#c44e52"}
    for mode, rows in by_mode.items():
        xs = [r["saturation_rate"] * 100 for r in rows]
        ys = [r["species_recall_mean"] for r in rows]
        ss = [max(8, 80 - 2 * r["stop_pass_mean"]) for r in rows]
        ax.scatter(xs, ys, label=mode, c=colors[mode], s=ss, alpha=0.55,
                   edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Saturation rate (% of arms pruned early)")
    ax.set_ylabel("Mean species recall vs. K_max")
    ax.set_title(
        "Stop-threshold trade-off: saturation% vs recall\n"
        "(point size ∝ passes saved per arm)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="stop_when")
    ax.set_ylim(0.85, 1.0)
    fig.tight_layout()
    out = out_dir / "pareto_saturation_vs_recall.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 4: Per-config residual guarantee distribution
# ---------------------------------------------------------------------------

def plot_residual_distributions(traces, out_dir, configs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    p_miss_data = []
    chao_data = []
    labels = []
    for label, kw in configs:
        per_arm_p = []
        per_arm_c = []
        for rows in traces.values():
            sp = replay_arm(rows, **kw)
            snap = rows[-1] if sp is None else next(
                r for r in rows if int(r["runs_so_far"]) == sp
            )
            per_arm_p.append(snap["good_turing_missing_mass"])
            per_arm_c.append(min(2.0, chao1_ratio(snap)))
        p_miss_data.append(per_arm_p)
        chao_data.append(per_arm_c)
        labels.append(label)
    axes[0].boxplot(p_miss_data, labels=labels, vert=True, showmeans=True)
    axes[0].set_title("Residual GT p_miss at stop")
    axes[0].set_ylabel("p_miss")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].boxplot(chao_data, labels=labels, vert=True, showmeans=True)
    axes[1].set_title("Residual Chao1 ratio at stop")
    axes[1].set_ylabel("(CI95_high - S) / S")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(
        "Final guarantee delivered by selected stop configs "
        "(unsaturated arms held to K_max)",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "residual_guarantees.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 5: Heatmap of saturation% over (eps_gt, K) for stop_when=gt
# ---------------------------------------------------------------------------

def plot_heatmap(traces, k_max, out_dir):
    eps_grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    K_grid = list(range(3, k_max + 1, 2))
    arms = list(traces.values())
    grid = np.zeros((len(eps_grid), len(K_grid)))
    for i, eps in enumerate(eps_grid):
        for j, K in enumerate(K_grid):
            cnt = 0
            for rows in arms:
                sp = replay_arm(rows, stop_when="gt", eps_gt=eps, eps_c=1.0,
                                k_min=3, k_max=K, two_consec=True)
                if sp is not None:
                    cnt += 1
            grid[i, j] = cnt / len(arms)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower",
                   vmin=0, vmax=1.0)
    ax.set_xticks(range(len(K_grid)))
    ax.set_xticklabels(K_grid)
    ax.set_yticks(range(len(eps_grid)))
    ax.set_yticklabels([f"{e:.2f}" for e in eps_grid])
    ax.set_xlabel("K_max (passes budget)")
    ax.set_ylabel("ε (Good-Turing threshold)")
    ax.set_title("Saturation rate (stop_when=gt, k_min=3, two-consec)")
    for i in range(len(eps_grid)):
        for j in range(len(K_grid)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < 0.6 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction saturated")
    fig.tight_layout()
    out = out_dir / "saturation_heatmap_gt.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Recommendation table
# ---------------------------------------------------------------------------

def build_recommendation(traces, sweep_rows, out_dir):
    candidates = [
        ("baseline current (gt=0.05, eps_c=0.20, k_min=2, two=T)",
         dict(stop_when="both", eps_gt=0.05, eps_c=0.20, k_min=2, two_consec=True)),
        ("conservative (gt=0.02, eps_c=0.20, k_min=3, two=T)",
         dict(stop_when="both", eps_gt=0.02, eps_c=0.20, k_min=3, two_consec=True)),
        ("balanced (gt=0.05, eps_c=0.20, k_min=3, two=T)",
         dict(stop_when="both", eps_gt=0.05, eps_c=0.20, k_min=3, two_consec=True)),
        ("aggressive (gt=0.10, eps_c=0.50, k_min=3, two=T)",
         dict(stop_when="both", eps_gt=0.10, eps_c=0.50, k_min=3, two_consec=True)),
        ("gt-only balanced (gt=0.05, k_min=3, two=T)",
         dict(stop_when="gt", eps_gt=0.05, eps_c=1.0, k_min=3, two_consec=True)),
        ("gt-only strict (gt=0.02, k_min=3, two=T)",
         dict(stop_when="gt", eps_gt=0.02, eps_c=1.0, k_min=3, two_consec=True)),
    ]
    results = []
    arms = list(traces.values())
    for label, kw in candidates:
        for K in [10, 15, 20, 30, 40]:
            saturated = 0
            stop_passes = []
            recalls = []
            residual_p = []
            residual_c = []
            for rows in arms:
                sp = replay_arm(rows, k_max=K, **kw)
                if sp is not None:
                    saturated += 1
                    stop_passes.append(sp)
                    snap = next(r for r in rows if int(r["runs_so_far"]) == sp)
                else:
                    stop_passes.append(K)
                    # nearest <= K
                    cands = [r for r in rows if int(r["runs_so_far"]) <= K]
                    snap = cands[-1] if cands else rows[0]
                # recall vs K_max ground truth
                kmax_n = rows[-1]["n_observed"]
                stop_n = snap["n_observed"]
                recalls.append(stop_n / kmax_n if kmax_n > 0 else 1.0)
                residual_p.append(snap["good_turing_missing_mass"])
                residual_c.append(min(2.0, chao1_ratio(snap)))
            results.append({
                "label": label,
                "K_max": K,
                "sat_rate": saturated / len(arms),
                "mean_stop": float(np.mean(stop_passes)),
                "p95_stop": float(np.percentile(stop_passes, 95)),
                "passes_total": int(np.sum(stop_passes)),
                "passes_saved": int(K * len(arms) - np.sum(stop_passes)),
                "recall_mean": float(np.mean(recalls)),
                "recall_p10": float(np.percentile(recalls, 10)),
                "p_miss_mean": float(np.mean(residual_p)),
                "p_miss_p90": float(np.percentile(residual_p, 90)),
                "chao_mean": float(np.mean(residual_c)),
            })
    return results


def write_recommendation_md(results, out_dir):
    lines = []
    lines.append("# Stop-threshold recommendation\n")
    lines.append(
        "Six candidate configs replayed against the K_max=40 trace "
        "(31 column-error arms). For each candidate we report what "
        "happens when the user sets `--max-passes` to one of "
        "{10, 15, 20, 30, 40}: how many arms saturate by then, "
        "average passes spent, the species recall vs the full "
        "K=40 ground truth, and the residual guarantee (p_miss / "
        "Chao1 ratio) actually delivered.\n"
    )
    lines.append(
        "| config | K_max | sat % | mean K | p95 K | passes saved "
        "| recall mean | recall p10 | p_miss mean (p90) | chao ratio mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['K_max']} | "
            f"{100*r['sat_rate']:.1f}% | {r['mean_stop']:.1f} | "
            f"{r['p95_stop']:.0f} | {r['passes_saved']} | "
            f"{r['recall_mean']:.3f} | {r['recall_p10']:.3f} | "
            f"{r['p_miss_mean']:.3f} ({r['p_miss_p90']:.3f}) | "
            f"{r['chao_mean']:.3f} |"
        )
    (out_dir / "recommendation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)

    run = args.run_dir
    trace_path = run / "good_turing" / "per_pass_per_arm.jsonl"
    sweep_json = run / "threshold_sweep" / "threshold_sweep.json"
    out_dir = run / "threshold_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    traces, k_max = load_traces(trace_path)
    sweep_rows = json.loads(sweep_json.read_text())["sweep"]

    logger.info(
        "loaded %d arms, K_max=%d, %d sweep configs",
        len(traces), k_max, len(sweep_rows),
    )

    p1 = plot_saturation_curves(traces, k_max, out_dir)
    logger.info("wrote %s", p1)
    p2 = plot_final_guarantee(traces, k_max, out_dir)
    logger.info("wrote %s", p2)
    p3 = plot_pareto_combined(sweep_rows, out_dir)
    logger.info("wrote %s", p3)
    p5 = plot_heatmap(traces, k_max, out_dir)
    logger.info("wrote %s", p5)
    cfgs = [
        ("gt=0.02\nk_min=3", dict(stop_when="gt", eps_gt=0.02, eps_c=1.0,
                                   k_min=3, k_max=k_max, two_consec=True)),
        ("gt=0.05\nk_min=3", dict(stop_when="gt", eps_gt=0.05, eps_c=1.0,
                                   k_min=3, k_max=k_max, two_consec=True)),
        ("both gt=0.05\neps_c=0.20", dict(stop_when="both", eps_gt=0.05,
                                           eps_c=0.20, k_min=3,
                                           k_max=k_max, two_consec=True)),
        ("both gt=0.10\neps_c=0.50", dict(stop_when="both", eps_gt=0.10,
                                           eps_c=0.50, k_min=3,
                                           k_max=k_max, two_consec=True)),
        ("chao1=0.20\nk_min=3", dict(stop_when="chao1", eps_gt=1.0,
                                      eps_c=0.20, k_min=3,
                                      k_max=k_max, two_consec=True)),
    ]
    p4 = plot_residual_distributions(traces, out_dir, cfgs)
    logger.info("wrote %s", p4)

    rec = build_recommendation(traces, sweep_rows, out_dir)
    (out_dir / "recommendation.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )
    write_recommendation_md(rec, out_dir)
    logger.info("wrote %s", out_dir / "recommendation.md")


if __name__ == "__main__":
    main()
