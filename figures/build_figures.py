"""Generate Tier 1 + Tier 2 figures for Paper B.

Reads log/hypotheses.csv (produced by parse_notes.py) and writes:
  fig_trajectory.pdf     -- per-hypothesis accuracy trajectory + running champion
  fig_success_rate.pdf   -- rolling-window success rate with phase overlays
  fig_attribution.pdf    -- attribution waterfall (69.67% -> 96.59%)
  fig_hypothesis_mix.pdf -- config vs code hypothesis mix by phase
  fig_cross_scale.pdf    -- cross-scale reversals (Mixup, LS, drop-path removal)
  fig_bias_decomp.pdf    -- workflow-induced vs inherent bias diverging bars

Style: grayscale-friendly, seaborn-less, tight_layout, PDF.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[0].parent / "log" / "hypotheses.csv"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PHASE_COLORS = {
    "1":  "#1f77b4",  # blue
    "1b": "#2ca02c",  # green
    "2":  "#d62728",  # red
    "3":  "#9467bd",  # purple
}
PHASE_LABEL = {"1": "Phase 1", "1b": "Phase 1b", "2": "Phase 2", "3": "Phase 3"}


def load() -> list[dict]:
    with CSV.open() as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Figure 1: accuracy trajectory.                                               #
# --------------------------------------------------------------------------- #
def fig_trajectory(rows):
    # Index every non-baseline hypothesis sequentially across phases.
    xs, ys, colors, markers, is_success = [], [], [], [], []
    phase_spans = {}  # phase -> (xmin, xmax)
    i = 0
    for r in rows:
        if r["outcome"] == "baseline":
            continue
        i += 1
        xs.append(i)
        ys.append(float(r["acc"]))
        colors.append(PHASE_COLORS[r["phase"]])
        is_success.append(r["outcome"] == "success")
        phase_spans.setdefault(r["phase"], [i, i])
        phase_spans[r["phase"]][1] = i

    # Per-scale champion curve: step up only on PUSHED within the same scale.
    # Phase 1 + 1b share the CIFAR-10 5.8M chain; Phase 2 is CIFAR-100 22M; Phase 3 is IN-1K 22M.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.0, 6.2),
                                          sharex=False,
                                          gridspec_kw={"height_ratios": [3.2, 1.8, 1.8]})

    # --- Top panel: CIFAR-10 chain (Phase 1 + 1b) ---
    c10_rows = [r for r in rows if r["phase"] in ("1", "1b") and r["outcome"] != "baseline"]
    baseline_c10 = next(r for r in rows if r["id"] == "Baseline-C10")
    ch = float(baseline_c10["acc"])
    champ_x, champ_y = [0], [ch]
    local_x = 0
    for r in c10_rows:
        local_x += 1
        a = float(r["acc"])
        col = PHASE_COLORS[r["phase"]]
        if r["outcome"] == "success":
            ax1.scatter(local_x, a, s=25, color=col, marker="o", zorder=3,
                         edgecolors="black", linewidths=0.4)
            ch = max(ch, a)
            champ_x.append(local_x)
            champ_y.append(ch)
        else:
            ax1.scatter(local_x, a, s=12, color=col, marker="x", alpha=0.65, zorder=2)
    ax1.plot(champ_x, champ_y, color="black", lw=1.3, zorder=4, label="Running champion")
    ax1.axhline(float(baseline_c10["acc"]), color="gray", lw=0.6, ls=":", alpha=0.6)

    # Phase shading.
    p1_end = sum(1 for r in c10_rows if r["phase"] == "1")
    ax1.axvspan(0.5, p1_end + 0.5, color=PHASE_COLORS["1"], alpha=0.06, zorder=0)
    ax1.axvspan(p1_end + 0.5, len(c10_rows) + 0.5, color=PHASE_COLORS["1b"], alpha=0.06, zorder=0)
    ax1.text(p1_end / 2, 70.5, "Phase 1 (config + code, no literature)", ha="center", fontsize=8, color="gray")
    ax1.text(p1_end + (len(c10_rows) - p1_end) / 2, 70.5, "Phase 1b (+ literature)", ha="center", fontsize=8, color="gray")

    # Regime annotations.
    ax1.annotate("Discovery\n(H1--H6)", xy=(3.5, 92), xytext=(3.5, 79), fontsize=7.5,
                   ha="center", arrowprops=dict(arrowstyle="->", lw=0.5, color="gray"))
    ax1.annotate("Exploitation\n(H7--H13)", xy=(10, 94.5), xytext=(10, 81), fontsize=7.5,
                   ha="center", arrowprops=dict(arrowstyle="->", lw=0.5, color="gray"))
    ax1.annotate("Config-saturation plateau\n(H14--H42)", xy=(28, 94.8), xytext=(28, 83),
                   fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="->", lw=0.5, color="gray"))

    ax1.set_xlim(0, len(c10_rows) + 1)
    ax1.set_ylim(68, 98)
    ax1.set_ylabel("CIFAR-10 top-1 (%)")
    ax1.set_title("(a) Accuracy trajectory, CIFAR-10 chain (5.8M params)")
    # Legend.
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", lw=1.3, label="Running champion"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                 markeredgecolor="black", markersize=6, lw=0, label="Accepted (PUSHED)"),
        Line2D([0], [0], marker="x", color="gray", lw=0, markersize=6, label="Rejected"),
    ]
    # Place the legend inside the axes in the empty area on the right side of
    # Phase 1b, below the champion plateau and above the Phase 1b label, with a
    # white background so it sits cleanly over the shading.
    ax1.legend(handles=legend_elems, loc="center right",
                bbox_to_anchor=(0.995, 0.62),
                framealpha=0.92, facecolor="white", edgecolor="lightgray",
                fontsize=7.5, borderaxespad=0.4)

    # --- Middle panel: CIFAR-100 chain (Phase 2) ---
    c100_rows = [r for r in rows if r["phase"] == "2" and r["outcome"] != "baseline"]
    baseline_c100 = next(r for r in rows if r["id"] == "Baseline-C100")
    ch = float(baseline_c100["acc"])
    champ_x, champ_y = [0], [ch]
    local_x = 0
    for r in c100_rows:
        local_x += 1
        a = float(r["acc"])
        col = PHASE_COLORS["2"]
        if r["outcome"] == "success":
            ax2.scatter(local_x, a, s=25, color=col, marker="o", zorder=3,
                         edgecolors="black", linewidths=0.4)
            ch = max(ch, a)
            champ_x.append(local_x)
            champ_y.append(ch)
        else:
            ax2.scatter(local_x, a, s=12, color=col, marker="x", alpha=0.65, zorder=2)
    ax2.plot(champ_x, champ_y, color="black", lw=1.3, zorder=4)
    ax2.axhline(float(baseline_c100["acc"]), color="gray", lw=0.6, ls=":", alpha=0.6)
    ax2.axvspan(0.5, len(c100_rows) + 0.5, color=PHASE_COLORS["2"], alpha=0.06, zorder=0)
    ax2.set_xlim(0, len(c100_rows) + 1)
    ax2.set_ylim(80.5, 84)
    ax2.set_xlabel("Hypothesis index (Phase 2, CIFAR-100, 22M params)")
    ax2.set_ylabel("Top-1 (%)")
    ax2.set_title("(b) CIFAR-100 chain (22M params)", fontsize=9)

    # --- Bottom panel: IN-1K trajectory (Phase 3). ---
    # Clip very-low-acc divergent runs (e.g. H3-8 = 62.93%, NaN divergence) out
    # of the main y-range and call them out with an off-bottom annotation; the
    # tight y-range keeps the meaningful 77--78% variation legible.
    baseline_in1k = next((r for r in rows if r["id"] == "Baseline-IN1K"), None)
    in1k_rows = [r for r in rows if r["phase"] == "3" and r["outcome"] != "baseline"]
    Y_LO, Y_HI = 76.5, 78.5
    if baseline_in1k is not None:
        b_acc = float(baseline_in1k["acc"])
        xs_in, ys_in = [0], [b_acc]
        for k, r in enumerate(in1k_rows, start=1):
            a = float(r["acc"])
            if a >= Y_LO:                        # only in-range points join the line
                xs_in.append(k)
                ys_in.append(a)
        ax3.plot(xs_in, ys_in, color="black", lw=0.8, zorder=2)
        for k, r in enumerate(in1k_rows, start=1):
            a = float(r["acc"])
            col = PHASE_COLORS["3"]
            if a < Y_LO:                          # divergent run: clip marker + label
                ax3.scatter(k, Y_LO + 0.03, s=30, color=col, marker="v",
                             alpha=0.9, zorder=3, edgecolors="black", linewidths=0.4)
                ax3.annotate(f"{r['id']}: {a:.1f}%\n(divergent)",
                              xy=(k, Y_LO + 0.05), xytext=(k + 0.6, Y_LO + 0.45),
                              fontsize=6.5, ha="left", color=col,
                              arrowprops=dict(arrowstyle="-", lw=0.4, color=col))
                continue
            if r["outcome"] == "success":
                ax3.scatter(k, a, s=25, color=col, marker="o", zorder=3,
                             edgecolors="black", linewidths=0.4)
            elif r["outcome"] == "pending":
                ax3.scatter(k, a, s=25, color=col, marker="s", alpha=0.4, zorder=3)
            else:
                ax3.scatter(k, a, s=12, color=col, marker="x", alpha=0.65, zorder=2)
        ax3.axhline(b_acc, color="gray", lw=0.6, ls=":", alpha=0.6)
        ax3.axvspan(0.5, max(1, len(in1k_rows)) + 0.5, color=PHASE_COLORS["3"], alpha=0.06, zorder=0)
    ax3.set_xlim(0, max(5, len(in1k_rows) + 2))
    ax3.set_ylim(Y_LO, Y_HI)
    ax3.set_xlabel("Hypothesis index (Phase 3, ImageNet-1K, 22M params)")
    ax3.set_ylabel("Top-1 (%)")
    ax3.set_title("(c) ImageNet-1K chain", fontsize=9)

    plt.tight_layout()
    out = HERE / "fig_trajectory.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 2: rolling success rate.                                              #
# --------------------------------------------------------------------------- #
def fig_success_rate(rows):
    # All four phases, in order. Baselines excluded.
    phase_order = ("1", "1b", "2", "3")
    relevant = [r for r in rows if r["phase"] in phase_order and r["outcome"] != "baseline"]
    ys = [1.0 if r["outcome"] == "success" else 0.0 for r in relevant]

    window = 10
    rolling = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        rolling.append(sum(ys[lo:i + 1]) / (i - lo + 1))

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(1, len(ys) + 1)

    # Phase boundaries and band fills (drawn first, behind line).
    counts = {ph: sum(1 for r in relevant if r["phase"] == ph) for ph in phase_order}
    bounds = {}
    cur = 0.5
    for ph in phase_order:
        nxt = cur + counts[ph]
        bounds[ph] = (cur, nxt)
        if counts[ph] > 0:
            ax.axvspan(cur, nxt, color=PHASE_COLORS[ph], alpha=0.08, zorder=0)
        cur = nxt

    # Mean success rate per phase: dashed reference + labelled at top of band.
    def mean_rate(phase_key):
        vals = [1.0 if r["outcome"] == "success" else 0.0 for r in relevant if r["phase"] == phase_key]
        return sum(vals) / max(1, len(vals))

    label_y = 0.95
    label_bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="none", alpha=0.85)
    for ph in phase_order:
        lo, hi = bounds[ph]
        if counts[ph] == 0:
            continue
        rate = mean_rate(ph)
        ax.hlines(rate, lo, hi, colors=PHASE_COLORS[ph],
                   linewidth=1.2, linestyles="dashed", zorder=2)
        ax.text((lo + hi) / 2, label_y, f"{PHASE_LABEL[ph]}: {rate:.0%}",
                 ha="center", va="top", fontsize=8, color=PHASE_COLORS[ph],
                 bbox=label_bbox, zorder=5)

    # Rolling line drawn last so it sits on top of the bands.
    ax.plot(x, rolling, color="black", lw=1.2, zorder=3)

    # Phase-boundary verticals.
    prev = 0.5
    for ph in phase_order[:-1]:
        prev += counts[ph]
        if counts[ph] > 0:
            ax.axvline(prev, color="gray", lw=0.6, ls=":", zorder=1)

    ax.set_xlim(0.5, len(relevant) + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Hypothesis index")
    ax.set_ylabel(f"{window}-hypothesis rolling success rate")
    ax.set_title(f"Rolling success rate across phases ({window}-hypothesis window)")

    plt.tight_layout()
    out = HERE / "fig_success_rate.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 3: attribution waterfall.                                             #
# --------------------------------------------------------------------------- #
def fig_attribution(rows):
    # Use pre-curated attribution from the paper's table, because we want the
    # narrative grouping (FFN / training recipe / DW shortcut / ...) rather
    # than per-hypothesis deltas (which are too noisy to read).
    items = [
        ("FFN on CA blocks (H1)",           21.41, PHASE_COLORS["1"]),
        ("Training recipe (H7-H29)",         2.58, PHASE_COLORS["1"]),
        ("DW shortcut (H6)",                 0.81, PHASE_COLORS["1"]),
        ("Aux deep supervision (H46,H50,H64)", 0.74, PHASE_COLORS["1b"]),
        ("Other incremental",                0.47, "gray"),
        ("CPE 5x5 (H51-H52)",                0.26, PHASE_COLORS["1b"]),
        ("Layer Scale + DW (H30,H56)",       0.26, PHASE_COLORS["1"]),
        ("4-stage hierarchy (H13)",          0.15, PHASE_COLORS["1"]),
        ("SwiGLU (H54)",                     0.12, PHASE_COLORS["1b"]),
        ("Learned temperature (H47)",        0.12, PHASE_COLORS["1b"]),
    ]
    base = 69.67
    end = base + sum(v for _, v, _ in items)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y_pos = np.arange(len(items))[::-1]
    # Waterfall: cumulative left-edge, bar width = gain.
    cum = base
    lefts, widths, colors, labels, values = [], [], [], [], []
    for name, gain, color in items:
        lefts.append(cum)
        widths.append(gain)
        colors.append(color)
        labels.append(name)
        values.append(gain)
        cum += gain

    ax.barh(y_pos, widths, left=lefts, color=colors, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Annotate values at the right of each bar.
    for yp, left, w in zip(y_pos, lefts, widths):
        ax.text(left + w + 0.15, yp, f"+{w:.2f} pp", va="center", fontsize=7.5)

    # Baseline and final vertical markers, labelled below the bars (out of title area).
    ax.axvline(base, color="gray", lw=0.8, ls="--")
    ax.axvline(end, color="black", lw=0.8, ls="--")
    ax.text(base, -0.8, f"baseline {base:.2f}%", ha="center", va="top",
             fontsize=7.5, color="gray")
    ax.text(end, -0.8, f"{end:.2f}%", ha="center", va="top", fontsize=7.5)

    ax.set_xlim(base - 1, end + 3)
    ax.set_ylim(-1.5, len(items) - 0.2)        # room below for baseline/end labels
    ax.set_xlabel("CIFAR-10 top-1 (%)")
    ax.set_title("Innovation attribution: CIFAR-10 $+26.92$ pp total gain (Phase 1 + 1b)")

    # Phase colour legend — placed outside the axes on the right so it never
    # masks the smallest bars at the bottom of the waterfall.
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=PHASE_COLORS["1"], label="Phase 1", edgecolor="black", linewidth=0.3),
           Patch(facecolor=PHASE_COLORS["1b"], label="Phase 1b", edgecolor="black", linewidth=0.3),
           Patch(facecolor="gray", label="mixed", edgecolor="black", linewidth=0.3)]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0),
               framealpha=0.0, fontsize=7.5, borderaxespad=0.0)

    plt.tight_layout()
    out = HERE / "fig_attribution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 4: hypothesis-type mix (config vs code) per phase.                     #
# --------------------------------------------------------------------------- #
def fig_hypothesis_mix(rows):
    phase_order = ["1", "1b", "2", "3"]
    counts = {p: Counter() for p in phase_order}
    for r in rows:
        if r["outcome"] == "baseline" or r["phase"] not in phase_order:
            continue
        # All Phase 3 hypotheses introduced new mechanisms (per the bold-direction
        # brief), so treat them as code regardless of how CODE_IDS was seeded.
        ctype = "code" if (r["change_type"] == "code" or r["phase"] == "3") else "config"
        counts[r["phase"]][ctype] += 1

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bottom_vals = np.zeros(len(phase_order))
    colors_map = {"config": "#aec7e8", "code": "#2ca02c"}
    for change_type, color in [("config", colors_map["config"]), ("code", colors_map["code"])]:
        vals = [counts[p][change_type] for p in phase_order]
        totals = [sum(counts[p].values()) for p in phase_order]
        pct = [100 * v / t if t else 0 for v, t in zip(vals, totals)]
        ax.bar(np.arange(len(phase_order)), pct, bottom=bottom_vals, color=color,
                 edgecolor="black", linewidth=0.3, label=change_type)
        for i, p in enumerate(pct):
            if p > 8:
                ax.text(i, bottom_vals[i] + p / 2, f"{p:.0f}%", ha="center", va="center",
                         fontsize=8, color="white" if color != "#aec7e8" else "black")
        bottom_vals += np.array(pct)

    ax.set_xticks(np.arange(len(phase_order)))
    ax.set_xticklabels([PHASE_LABEL[p] for p in phase_order])
    ax.set_ylabel("Fraction of hypotheses (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Hypothesis type mix by phase")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
               framealpha=0.0, fontsize=8, borderaxespad=0.0)

    plt.tight_layout()
    out = HERE / "fig_hypothesis_mix.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 5: cross-scale reversals.                                             #
# --------------------------------------------------------------------------- #
def fig_cross_scale(rows=None):
    # Hard-coded from the paper's table; the point is visual.
    items = [
        ("Mixup (on)",          -0.68, +0.77),   # Δ at 5.8M ; Δ at 22M
        ("Label smoothing (on)", -0.13, +0.30),
        ("Drop-path removal",   +0.46, -0.60),
    ]
    labels = [x[0] for x in items]
    d1 = np.array([x[1] for x in items])
    d2 = np.array([x[2] for x in items])

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(items))[::-1]
    w = 0.38
    ax.barh(y + w / 2, d1, height=w, color=PHASE_COLORS["1"], edgecolor="black", linewidth=0.3,
             label="CIFAR-10 (5.8M)")
    ax.barh(y - w / 2, d2, height=w, color=PHASE_COLORS["2"], edgecolor="black", linewidth=0.3,
             label="CIFAR-100 (22M)")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("$\\Delta$ val top-1 (pp) vs same-scale baseline")
    ax.set_title("Training-recipe decisions that reverse sign across scale")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
               framealpha=0.0, fontsize=8, borderaxespad=0.0)

    # Annotate bars with values.
    for yp, v in zip(y + w / 2, d1):
        xp = v + (0.03 if v >= 0 else -0.03)
        ax.text(xp, yp, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    for yp, v in zip(y - w / 2, d2):
        xp = v + (0.03 if v >= 0 else -0.03)
        ax.text(xp, yp, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)

    ax.set_xlim(-1.2, 1.3)
    plt.tight_layout()
    out = HERE / "fig_cross_scale.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 6: bias decomposition diverging bars.                                 #
# --------------------------------------------------------------------------- #
def fig_bias_decomp(rows=None):
    # From the paper's bias_decomposition table; bullet count -> strength.
    behaviours_bias = [
        ("Always build on current champion",                    3, 1),
        ("No parallel architecture tracks",                     3, 0),
        ("Binary commit/discard against one champion",          3, 0),
        ("Systematic boundary-finding (LR/kernel sweep)",       2, 1),
        ("Config tweaks over structural when both allowed",     2, 2),
        ("Risk aversion after bold failure",                    0, 3),
        ("Anchoring on familiar literature",                    0, 3),
    ]
    labels = [b[0] for b in behaviours_bias]
    induced = -np.array([b[1] for b in behaviours_bias])  # left
    inherent = np.array([b[2] for b in behaviours_bias])   # right

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, induced, color="#ff9896", edgecolor="black", linewidth=0.3,
             label="Workflow-induced")
    ax.barh(y, inherent, color="#98df8a", edgecolor="black", linewidth=0.3,
             label="LLM-inherent")
    ax.axvline(0, color="black", lw=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["primary", "moderate", "minor", "", "minor", "moderate", "primary"])
    ax.set_xlabel("Attribution strength")
    ax.set_title("Decomposition of observed agent biases")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    out = HERE / "fig_bias_decomp.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    rows = load()
    fig_trajectory(rows)
    fig_success_rate(rows)
    fig_attribution(rows)
    fig_hypothesis_mix(rows)
    fig_cross_scale(rows)
    fig_bias_decomp(rows)


if __name__ == "__main__":
    main()
