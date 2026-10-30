"""make_fig_smoothing.py
Conceptual workflow of continuous spatial kernel smoothing and edge pooling.
All 15 members have distinct matching colors:
- 7 members in the damage zone have bold, saturated colors.
- 8 members across the healthy span have distinct, soft, pastel/muted colors (not too dark/bold).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Standardised Color Palette
C_DAM_RED = "#DC2626"       # 1. Damaged Member (Bold Crimson Red)
C_BLUE = "#1D4ED8"          # 2. Adjacent Diagonal Left (Bold Cobalt Blue)
C_AMBER = "#D97706"         # 3. Top Chord Left (Bold Rich Amber)
C_TEAL = "#0F766E"          # 4. Adjacent Diagonal Right (Bold Deep Teal)
C_ORANGE = "#EA580C"        # 5. Top Chord Right (Bold Rust Orange)
C_EXCLUDED = "#94A3B8"      # Excluded candidates E \ E_K (Muted Slate Grey)

C_CONT = "#2B6CB0"          # Continuous Profile R(x) Blue
C_GRID = "#D8DEE6"          # Soft Gridline
DARK = "#1F2937"            # Dark Charcoal Text/Ticks
BORDER = "#374151"          # Spine Border

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.8,
    "axes.titlesize": 9.2,
    "axes.titleweight": "normal",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.2,
    "axes.edgecolor": BORDER,
    "axes.linewidth": 0.8,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def style(ax: plt.Axes, has_yticks: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    if not has_yticks:
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False, labelleft=False)
    else:
        ax.tick_params(axis="y", colors=DARK, labelsize=8.0, length=3.0)
        ax.grid(True, axis="y", color=C_GRID, lw=0.5, alpha=0.7)
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_axisbelow(True)


# Define all 15 candidate members: K=5 top candidates in E_K and 10 excluded candidates in E \ E_K
members = [
    # Top chords (3)
    {"p1": (-6, 2.0), "p2": (-2, 2.0), "xm": -4.0, "s": 0.38, "color": C_AMBER,    "lw": 2.4, "alpha": 0.95, "dam": False, "topk": True},
    {"p1": (-2, 2.0), "p2": (2, 2.0),  "xm": 0.0,  "s": 0.28, "color": C_ORANGE,   "lw": 2.4, "alpha": 0.95, "dam": False, "topk": True},
    {"p1": (2, 2.0),  "p2": (6, 2.0),  "xm": 4.0,  "s": 0.08, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    # Bottom chords (4)
    {"p1": (-8, 0.0), "p2": (-4, 0.0), "xm": -6.0, "s": 0.09, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (-4, 0.0), "p2": (0, 0.0),  "xm": -2.0, "s": 0.95, "color": C_DAM_RED,  "lw": 3.2, "alpha": 1.00, "dam": True,  "topk": True},
    {"p1": (0, 0.0),  "p2": (4, 0.0),  "xm": 2.0,  "s": 0.11, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (4, 0.0),  "p2": (8, 0.0),  "xm": 6.0,  "s": 0.08, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    # Diagonals (8)
    {"p1": (-8, 0.0), "p2": (-6, 2.0), "xm": -7.0, "s": 0.08, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (-4, 0.0), "p2": (-6, 2.0), "xm": -5.0, "s": 0.22, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (-4, 0.0), "p2": (-2, 2.0), "xm": -3.0, "s": 0.45, "color": C_BLUE,     "lw": 2.4, "alpha": 0.95, "dam": False, "topk": True},
    {"p1": (0, 0.0),  "p2": (-2, 2.0), "xm": -1.0, "s": 0.35, "color": C_TEAL,     "lw": 2.4, "alpha": 0.95, "dam": False, "topk": True},
    {"p1": (0, 0.0),  "p2": (2, 2.0),  "xm": 1.0,  "s": 0.20, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (4, 0.0),  "p2": (2, 2.0),  "xm": 3.0,  "s": 0.09, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (4, 0.0),  "p2": (6, 2.0),  "xm": 5.0,  "s": 0.07, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
    {"p1": (8, 0.0), "p2": (6, 2.0),  "xm": 7.0,  "s": 0.06, "color": C_EXCLUDED, "lw": 1.2, "alpha": 0.45, "dam": False, "topk": False},
]

fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [0.95, 1.0, 1.05]})
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.09, right=0.97, hspace=0.36)

# ============================================================================
# Panel (a): Sensor Graph with Top-K Candidate Members Highlighted
# ============================================================================
ax0 = axes[0]

# 1. Draw excluded members first (in background, muted slate grey)
for m in members:
    if not m["topk"]:
        (x1, y1), (x2, y2) = m["p1"], m["p2"]
        ax0.plot([x1, x2], [y1, y2], color=m["color"], lw=m["lw"], alpha=m["alpha"], zorder=1)

# 2. Draw top-K candidate members (in foreground, colored and bold)
for m in members:
    if m["topk"] and not m["dam"]:
        (x1, y1), (x2, y2) = m["p1"], m["p2"]
        ax0.plot([x1, x2], [y1, y2], color=m["color"], lw=m["lw"], alpha=m["alpha"], zorder=3)

# 3. Draw damaged member (prominent bold red)
for m in members:
    if m["dam"]:
        (x1, y1), (x2, y2) = m["p1"], m["p2"]
        ax0.plot([x1, x2], [y1, y2], color=m["color"], lw=m["lw"], alpha=m["alpha"], zorder=4)

# Midpoint of damaged member with 'x'
pt_dam_marker = ax0.scatter([-2.0], [0.0], color=C_DAM_RED, s=60, marker="x", linewidths=1.6, zorder=6)

# Sensor nodes (identical small blue dots)
x_bot = np.array([-8, -4, 0, 4, 8])
x_top = np.array([-6, -2, 2, 6])
all_x = np.concatenate([x_bot, x_top])
all_y = np.concatenate([[0.0] * len(x_bot), [2.0] * len(x_top)])
node_pts = ax0.scatter(all_x, all_y, color=C_CONT, s=30, zorder=5, label="Sensors")

handle_dam = ax0.plot([], [], color=C_DAM_RED, marker="x", markersize=6.0, markeredgewidth=1.6, linestyle="none", label="Damaged")[0]
handle_topk = ax0.plot([], [], color=C_BLUE, lw=2.0, linestyle="-", label=r"Top-$K$ ($\mathcal{E}_K$)")[0]
handle_excl = ax0.plot([], [], color=C_EXCLUDED, lw=1.2, linestyle="-", label="Excluded")[0]
handle_sensors = ax0.plot([], [], color=C_CONT, marker="o", markersize=5.0, linestyle="none", label="Sensors")[0]

ax0.set_title(r"(a) Candidate member connections $\sigma_e$", loc="left", fontsize=9.2, fontweight="bold", color=DARK, pad=5)
ax0.set_ylim(-0.7, 2.7)
ax0.legend(handles=[handle_dam, handle_topk, handle_excl, handle_sensors], loc="lower right", bbox_to_anchor=(1.0, 1.02),
           ncol=4, frameon=False, fontsize=7.8, handletextpad=0.25, columnspacing=0.8)
style(ax0, has_yticks=False)

# ============================================================================
# Panel (b): Gaussian Kernel Weighting for Top-K Candidates (K = 5)
# ============================================================================
ax1 = axes[1]
x_grid = np.linspace(-11.0, 9.5, 700)
h = 1.8

# Plot ONLY the K=5 Gaussians
top5 = [m for m in members if m["topk"]]
for m in top5:
    xm, s, color = m["xm"], m["s"], m["color"]
    kernel = s * np.exp(-((x_grid - xm) ** 2) / (2 * h ** 2))
    if m["dam"]:
        ax1.plot(x_grid, kernel, color=color, lw=2.2, alpha=0.95, zorder=5)
        ax1.scatter([xm], [s], color=color, s=40, marker="x", linewidths=1.6, zorder=6)
        ax1.vlines(xm, 0, s, color=color, linestyle="--", lw=1.2, alpha=0.7, zorder=4)
    else:
        ax1.plot(x_grid, kernel, color=color, lw=1.6, alpha=0.90, zorder=3)
        ax1.scatter([xm], [s], color=color, s=22, zorder=4)

ax1.set_title(r"(b) Gaussian kernel weighting for top-$K$ candidates ($K=5$)", loc="left", fontsize=9.2, fontweight="bold", color=DARK, pad=5)
ax1.set_ylabel(r"Weight $\kappa_h \cdot \sigma_e$")
ax1.set_ylim(0, 1.05)
ax1.set_yticks([0.0, 0.5, 1.0])
style(ax1, has_yticks=True)

# ============================================================================
# Panel (c): Continuous Pooled Profile R(x) from Top-K Candidates
# ============================================================================
ax2 = axes[2]
k_sum = sum(m["s"] * np.exp(-((x_grid - m["xm"]) ** 2) / (2 * h ** 2)) for m in top5)
w_sum = sum(np.exp(-((x_grid - m["xm"]) ** 2) / (2 * h ** 2)) for m in top5)

# Smooth pooling decaying to zero outside top-K support
Rx = k_sum / (w_sum + 0.50)
ax2.plot(x_grid, Rx, color=C_CONT, lw=2.1)
ax2.fill_between(x_grid, 0, Rx, color=C_CONT, alpha=0.14)

idx_max = np.argmax(Rx)
x_inferred = x_grid[idx_max]
line_peak = ax2.axvline(x_inferred, color=C_DAM_RED, linestyle="--", lw=1.5, label=r"Inferred peak $\widehat{x}$")
ax2.scatter([x_inferred], [Rx[idx_max]], color=C_DAM_RED, s=45, zorder=5)

ax2.axvspan(-4.0, 0.0, color=C_DAM_RED, alpha=0.08)

ax2.set_title(r"(c) Continuous pooled regional damage profile $R(x)$", loc="left", fontsize=9.2, fontweight="bold", color=DARK, pad=5)
ax2.set_xlabel(r"Longitudinal coordinate $x$", fontsize=9.5)
ax2.set_ylabel(r"Regional $R(x)$")
ax2.set_xlim(-9, 9)
ax2.set_ylim(0, 0.55)
ax2.set_yticks([0.0, 0.2, 0.4])
ax2.legend(handles=[line_peak], loc="upper right", frameon=False, fontsize=8.0, handletextpad=0.4)
style(ax2, has_yticks=True)

# Vertical alignment drop-line through all panels
for ax in axes:
    ax.axvline(-2.0, color=C_DAM_RED, linestyle=":", lw=1.0, alpha=0.45, zorder=0)

# ---------------------------------------------------------------------------
# Coordinate system (R1, Reviewer 2 comment 1)
# ---------------------------------------------------------------------------
x0, y0 = -10.5, 0.7
ax2.set_xlim(-11.0, 9)
for (dx, dy, lab) in ((0.9, 0.0, "$x$"), (0.0, 0.8, "$z$")):
    ax0.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.0, mutation_scale=8), zorder=6)
    ax0.text(x0 + dx + (0.25 if dx else 0.0), y0 + dy + (0.3 if dy else 0.0), lab,
             fontsize=8.2, ha="center", va="center", color=DARK)
ax2.tick_params(axis="x", bottom=True, labelbottom=True, colors=DARK, labelsize=8.0, length=3.0)
ax2.set_xticks([-8, -4, 0, 4, 8])

import os

# R1: write only into the revision folder; the submitted figures stay untouched.
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(os.path.dirname(script_dir), "figures", "figure2_spatial_smoothing_concept.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
print(f"Saved {out_path}")
