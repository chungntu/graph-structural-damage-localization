"""make_fig_edge_gradient.py
Single-panel edge-gradient intuition figure. This is a hypothetical
illustrative truss (not a real, measured structure), so it is drawn
as a plain schematic -- no bordered/gridded plot frame, no coordinate
axis -- with a single red accent reserved for the damaged connection.
The two top-chord nodes (N4, N5) sit on members that are not adjacent
to the cut, so their residual arrows are drawn with comparable,
undisturbed magnitude rather than one large and one small.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette matching make_fig_smoothing.py
C_CONT = "#2B6CB0"      # sensor-node blue
C_DAM_RED = "#DC2626"   # single accent: reserved for the damaged connection only
DARK = "#1F2937"        # dark charcoal text/ticks
DARK_MUTED = "#64748B"  # secondary math annotations (healthy side)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "legend.fontsize": 8.2,
    "text.color": DARK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# 5-node, 7-member 2-bay Warren truss (unchanged from the original figure)
nodes = {
    1: np.array([-3.0, 0.0]),
    2: np.array([ 0.0, 0.0]),
    3: np.array([ 3.0, 0.0]),
    4: np.array([-1.5, 1.35]),
    5: np.array([ 1.5, 1.35]),
}
edges = [
    (1, 2), (2, 3), (4, 5),
    (1, 4), (2, 4), (2, 5), (3, 5),
]
DAMAGED_EDGE = (2, 3)
HEALTHY_REF_EDGE = (1, 2)

# Nodal modal residuals r_{m,i}. N4/N5 (top chord) are not adjacent to
# the cut, so both carry comparable, undisturbed diffuse motion. N3 is
# isolated by the cut, so its residual stays small -- but still drawn
# large enough to read as a visible (short) arrow, not just a label.
residuals = {1: 0.65, 2: 0.72, 3: 0.28, 4: 0.60, 5: 0.58}
ARROW_SCALE = 0.85

fig, ax = plt.subplots(figsize=(7.0, 2.3))
fig.subplots_adjust(top=0.96, bottom=0.06, left=0.04, right=0.96)

# ---------------------------------------------------------------------------
# Truss Skeleton: All healthy members drawn uniformly with identical color and stroke;
# only the damaged connection is singled out in red.
# ---------------------------------------------------------------------------
C_STEEL = "#475569"   # Uniform steel-gray for all healthy truss members
LW_HEALTHY = 2.0       # Consistent line width for all healthy members

for (i, j) in edges:
    if (i, j) == DAMAGED_EDGE:
        continue
    p1, p2 = nodes[i], nodes[j]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_STEEL, lw=LW_HEALTHY, zorder=2, solid_capstyle="round")

# ---------------------------------------------------------------------------
# Nodal residual arrows -- diffuse motion across the structure
# ---------------------------------------------------------------------------
for idx, pos in nodes.items():
    dy = residuals[idx] * ARROW_SCALE
    muted = idx not in (1, 2, 3)
    ax.annotate("", xy=(pos[0], pos[1] + dy), xytext=(pos[0], pos[1]),
                 arrowprops=dict(arrowstyle="-|>", color=DARK_MUTED, lw=1.5, mutation_scale=10,
                                  alpha=0.4 if muted else 1.0), zorder=3)
    if not muted:
        # Shift r1, r2 to the left of arrows to prevent collision with diagonal members,
        # and shift r3 leftward closer to arrow 3
        offsets = {
            1: (-0.14, 0.55, "right"),
            2: (-0.14, 0.58, "right"),
            3: (0.08, 0.50, "left"),
        }
        ox, oy_fac, ha = offsets[idx]
        ax.text(pos[0] + ox, pos[1] + dy * oy_fac, f"$r_{{{idx}}}$",
                 fontsize=8.0, color=DARK_MUTED, va="center", ha=ha)

# ---------------------------------------------------------------------------
# Damaged member highlighted in red with midpoint 'x'
# ---------------------------------------------------------------------------
p2d, p3d = nodes[DAMAGED_EDGE[0]], nodes[DAMAGED_EDGE[1]]
ax.plot([p2d[0], p3d[0]], [p2d[1], p3d[1]], color=C_DAM_RED, lw=3.2, zorder=3, solid_capstyle="round")
ax.scatter([(p2d[0] + p3d[0]) / 2], [(p2d[1] + p3d[1]) / 2], color=C_DAM_RED, s=55,
            marker="x", linewidths=1.8, zorder=6)

# Unified vertical position for contrast annotations:
# Placed at an equal, lower height (y = 0.30) to clearly annotate their respective
# members while staying level and avoiding floating too high in the truss bays.
contrast_y = 0.30
ax.text((nodes[1][0] + nodes[2][0]) / 2, contrast_y,
         "$c_e \\approx 0$", fontsize=8.4, ha="center", va="center", color=DARK_MUTED)
ax.text((nodes[2][0] + nodes[3][0]) / 2, contrast_y, "$c_{e^*} \\gg 0$",
         fontsize=8.4, ha="center", va="center", color=C_DAM_RED, fontweight="bold")

# ---------------------------------------------------------------------------
# Sensor nodes -- uniform blue markers, matching Figure 4's convention
# ---------------------------------------------------------------------------
node_x = np.array([p[0] for p in nodes.values()])
node_y = np.array([p[1] for p in nodes.values()])
ax.scatter(node_x, node_y, color=C_CONT, s=34, zorder=4)
for idx, pos in nodes.items():
    if idx in (1, 2, 3):
        offset_y = -0.22 if pos[1] == 0.0 else 0.18
        ax.text(pos[0], pos[1] + offset_y, f"$N_{idx}$", fontsize=8.0, ha="center", va="center", color=DARK)

handle_dam = ax.plot([], [], color=C_DAM_RED, marker="x", markersize=6.5, markeredgewidth=1.6,
                       linestyle="none", label="Damaged")[0]
handle_sensors = ax.plot([], [], color=C_CONT, marker="o", markersize=5.5, linestyle="none", label="Sensors")[0]
ax.legend(handles=[handle_dam, handle_sensors], loc="upper right", bbox_to_anchor=(0.98, 0.98),
           ncol=2, frameon=False, fontsize=8.2, handletextpad=0.4, columnspacing=1.1)

# ---------------------------------------------------------------------------
# Coordinate system (R1, Reviewer 2 comment 1): x longitudinal, z vertical.
# Residual arrows are drawn along +z, the measured (vertical) direction.
# ---------------------------------------------------------------------------
origin = np.array([-3.45, 1.35])
axis_len = 0.42
for vec, lab in (((axis_len, 0.0), "$x$"), ((0.0, axis_len), "$z$")):
    ax.annotate("", xy=(origin[0] + vec[0], origin[1] + vec[1]), xytext=tuple(origin),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.1, mutation_scale=9), zorder=5)
    ax.text(origin[0] + vec[0] + (0.07 if vec[0] else 0.0), origin[1] + vec[1] + (0.08 if vec[1] else 0.0),
            lab, fontsize=8.4, ha="center", va="center", color=DARK)
ax.scatter([origin[0]], [origin[1]], color=DARK, s=6, zorder=5)

ax.set_xlim(-3.6, 3.8)
ax.set_ylim(-0.35, 1.95)
ax.axis("off")

import os

# R1: write only into the revision folder; the submitted figures stay untouched.
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(os.path.dirname(script_dir), "figures", "figure_edge_gradient_concept.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {out_path}")
