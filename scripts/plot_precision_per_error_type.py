"""Plot precision per labeled error type.

For typo / fd_violation / numeric_outlier / semantic_incompatible, the
per-cell ``label_result/*.jsonl`` already includes ``clean`` rows so
precision is computed directly. The inconsistency result file omits
``clean`` clusters; precision is recovered at the cluster level from
``outputs/20260513T234733Z_gpt5-family/labels.jsonl`` (the labeling
log that produced ``label_result/benchmark_inconsistency_20260515T010112Z_gpt5-family.jsonl``).

Two precisions per type:
  * strict  = error / (error + clean)              -- ambiguous excluded
  * lenient = (error + ambig) / (error + clean + ambig)

Recall (TP / (TP + FN)) is not computable from this data because the
labels only cover items the detector flagged; there is no oracle for
errors the detector missed. Counts are printed alongside the plot.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL_RESULT = ROOT / "label_result"
INCONSISTENCY_LABELS = (
    ROOT / "outputs" / "20260513T234733Z_gpt5-family" / "labels.jsonl"
)
OUT_DIR = ROOT / "notebooks" / "eda_outputs" / "precision_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_FILES = {
    "typo":                 "benchmark_typo_20260507T215150Z_gpt5-family.canonical.jsonl",
    "fd_violation":         "benchmark_fd_violation_20260514T063141Z_gpt5-family.jsonl",
    "numeric_outlier":      "benchmark_numeric_outlier_20260510T211448Z_gpt5-family.jsonl",
    "semantic_incompatible":"benchmark_semantic_incompatible_20260508T164531Z_gpt5-family.jsonl",
}


def count_cell_labels(path: Path) -> Counter:
    c: Counter = Counter()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            c[rec.get("label", "")] += 1
    return c


def count_inconsistency_cluster_labels(path: Path) -> Counter:
    """Cluster-level counts: latest status per target_id."""
    latest: dict[str, str] = {}
    ts: dict[str, str] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("target") != "column_cluster":
                continue
            tid = rec["target_id"]
            this_ts = rec.get("ts", "")
            if tid not in ts or this_ts >= ts[tid]:
                ts[tid] = this_ts
                latest[tid] = rec.get("status", "")
    status_to_label = {"tp": "error", "fp": "clean", "unsure": "ambiguous"}
    c: Counter = Counter()
    for s in latest.values():
        lbl = status_to_label.get(s)
        if lbl is not None:
            c[lbl] += 1
        else:
            c[f"_{s}"] += 1  # e.g. skip / partial -- keep visibility
    return c


def precisions(c: Counter) -> tuple[float, float]:
    e = c.get("error", 0)
    cl = c.get("clean", 0)
    a = c.get("ambiguous", 0)
    p_strict = e / (e + cl) if (e + cl) else 0.0
    p_lenient = (e + a) / (e + cl + a) if (e + cl + a) else 0.0
    return p_strict, p_lenient


def main() -> None:
    counts: dict[str, Counter] = {}
    for et, fname in CELL_FILES.items():
        counts[et] = count_cell_labels(LABEL_RESULT / fname)
    counts["inconsistency"] = count_inconsistency_cluster_labels(
        INCONSISTENCY_LABELS
    )

    all_types = ["typo", "inconsistency", "fd_violation",
                 "numeric_outlier", "semantic_incompatible"]

    print(f"{'error_type':<22} {'error':>6} {'clean':>6} {'ambig':>6} "
          f"{'P_strict':>9} {'P_lenient':>10}")
    print("-" * 65)
    rows: list[tuple[str, Counter, float, float]] = []
    for et in all_types:
        c = counts[et]
        ps, pl = precisions(c)
        rows.append((et, c, ps, pl))
        print(f"{et:<22} {c.get('error',0):>6} {c.get('clean',0):>6} "
              f"{c.get('ambiguous',0):>6} {ps:>9.3f} {pl:>10.3f}")
    print()
    print(f"Granularity: cell-level for typo/fd/numeric/semantic; "
          f"cluster-level for inconsistency.")
    print("Recall (FN) is not computable from labels-on-detections "
          "alone -- there is no oracle for missed errors.")

    # Sort by strict precision, descending (best detector first, top of chart).
    rows.sort(key=lambda r: r[2], reverse=True)
    types = [r[0] for r in rows]
    p_strict_vals = [r[2] for r in rows]
    p_lenient_vals = [r[3] for r in rows]

    # --- Plot: two panels, sharing y-axis. ---
    # Left: precision bars (horizontal, two per type).
    # Right: stacked composition of labels (error / ambig / clean) as %.
    display_labels = list(types)

    fig, (ax_p, ax_c) = plt.subplots(
        1, 2, figsize=(13, 5.2),
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.10},
    )
    y = np.arange(len(types))
    h = 0.36
    color_strict = "#1f77b4"
    color_lenient = "#ff7f0e"

    bs = ax_p.barh(y - h / 2, p_strict_vals, h,
                   color=color_strict,
                   label="strict = error / (error + clean)")
    bl = ax_p.barh(y + h / 2, p_lenient_vals, h,
                   color=color_lenient,
                   label="lenient = (error + ambig) / all")

    xmax = max(0.45, max(p_lenient_vals) * 1.25)
    ax_p.set_xlim(0, xmax)
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(display_labels, fontsize=10)
    ax_p.invert_yaxis()
    ax_p.set_xlabel("Precision")
    ax_p.grid(axis="x", linestyle=":", alpha=0.5)
    ax_p.set_axisbelow(True)
    for bars, vals in ((bs, p_strict_vals), (bl, p_lenient_vals)):
        for bar, v in zip(bars, vals):
            ax_p.text(v + xmax * 0.008, bar.get_y() + bar.get_height() / 2,
                      f"{v:.3f}", va="center", ha="left", fontsize=9)

    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                ncol=2, fontsize=9, framealpha=0.95,
                title="Precision (TP rate)", title_fontsize=9)

    # Right panel: stacked composition (% of total labeled per type).
    e_pct, a_pct, c_pct, totals = [], [], [], []
    for _, c, _, _ in rows:
        e = c.get("error", 0)
        a = c.get("ambiguous", 0)
        cl = c.get("clean", 0)
        n = e + a + cl
        totals.append(n)
        e_pct.append(100 * e / n if n else 0)
        a_pct.append(100 * a / n if n else 0)
        c_pct.append(100 * cl / n if n else 0)

    bar_h = 0.55
    p1 = ax_c.barh(y, e_pct, bar_h, color="#2ca02c", label="error (TP)")
    p2 = ax_c.barh(y, a_pct, bar_h, left=e_pct, color="#bcbd22", label="ambiguous")
    base = [a + b for a, b in zip(e_pct, a_pct)]
    p3 = ax_c.barh(y, c_pct, bar_h, left=base, color="#d62728", label="clean (FP)")

    ax_c.set_xlim(0, 118)  # extra room for n=... labels
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_xlabel("Composition of labeled detections (%)")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.invert_yaxis()
    ax_c.grid(axis="x", linestyle=":", alpha=0.5)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13),
                ncol=3, fontsize=9, framealpha=0.95)

    # Inline counts on the composition bars.
    for i, (e, a, cl, n) in enumerate(zip(
        [r[1].get("error", 0) for r in rows],
        [r[1].get("ambiguous", 0) for r in rows],
        [r[1].get("clean", 0) for r in rows],
        totals,
    )):
        # Show total n at the right edge of the bar.
        ax_c.text(102, i, f"n={n:,}",
                  va="center", ha="left", fontsize=9, color="#333")
        # Inline counts inside each segment if wide enough.
        def _put(x_left, width, label, color="white"):
            if width >= 6:
                ax_c.text(x_left + width / 2, i, label,
                          va="center", ha="center", fontsize=8.5, color=color)
        _put(0, e_pct[i], f"{e}")
        _put(e_pct[i], a_pct[i], f"{a}")
        _put(e_pct[i] + a_pct[i], c_pct[i], f"{cl}")

    fig.suptitle(
        "Detection precision per labeled error type",
        fontsize=12.5, y=0.99,
    )

    plt.tight_layout()
    out_png = OUT_DIR / "precision_per_error_type.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")

    # Also dump a CSV summary for downstream consumption.
    out_csv = OUT_DIR / "precision_per_error_type.csv"
    with out_csv.open("w") as fh:
        fh.write("error_type,error,clean,ambiguous,"
                "precision_strict,precision_lenient\n")
        for et, ps, pl in zip(types, p_strict_vals, p_lenient_vals):
            c = counts[et]
            fh.write(
                f"{et},{c.get('error',0)},{c.get('clean',0)},"
                f"{c.get('ambiguous',0)},{ps:.4f},{pl:.4f}\n"
            )
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
