"""R2 Figure 6: physical bridge outlines behind the candidate sensor graph.

Figure 6: sensor layout and candidate graph, split into the two truss-wall
elevations and the plan view, with the shaker positions P1/P2 and axes.
Figure 7: documented damage states on the plan-view candidate graph, colored
by the benchmark damage type, with P1/P2.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from r1_data import (
    DAMAGE_TYPE,
    FIG_DIR,
    L_BAY,
    RELATION_STYLE,
    SHAKER_X,
    TYPE_COLOR,
    edge_rows,
    locked_ground_truth,
    sensor_coordinates,
)

DARK = "#1F2937"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "axes.edgecolor": "#374151",
    "axes.linewidth": 0.8,
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})

coords = sensor_coordinates()
rows = edge_rows()
SHAKER_STYLE = dict(marker="*", s=150, color="black", edgecolor="white", linewidth=0.6, zorder=8)


def triad(ax, x0, y0, lx, ly, xlab, ylab):
    for dx, dy, lab in ((lx, 0.0, xlab), (0.0, ly, ylab)):
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.0, mutation_scale=8), zorder=9)
        ax.text(x0 + dx + (0.55 if dx else 0.0), y0 + dy + (0.28 * ly if dy else 0.0), lab,
                ha="center", va="center", fontsize=8.2)


def draw_edges(ax, relations, proj, wall=None):
    for row in rows:
        if row["relation"] not in relations:
            continue
        if wall is not None and not (abs(float(row["y_i_m"]) - wall) < 0.01 and abs(float(row["y_j_m"]) - wall) < 0.01):
            continue
        color, ls, lw, _ = RELATION_STYLE[row["relation"]]
        a = proj(coords[row["sensor_i"]])
        b = proj(coords[row["sensor_j"]])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, ls=ls, lw=lw, zorder=2, solid_capstyle="round")


def shaker_lines(ax, top, label_y):
    for name, x in SHAKER_X.items():
        ax.axvline(x, color="black", ls=":", lw=0.9, zorder=1)
        ax.text(x, label_y, name, ha="center", va="bottom", fontsize=8.0, fontweight="bold")


def draw_bridge_outline(ax, wall):
    """Schematic elevation checked against the official HBTA Sensor layout.pdf.

    Chord ordinates at instrumented joints come from the frozen graph geometry.
    End heights and uninstrumented intermediate joints are schematic context;
    they are not used in any localization computation.
    """
    station_x = [-17.5, -14.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 14.0, 17.5]
    upper_z = [2.8, 3.4, 4.1, 4.6, 4.9, 5.0, 4.9, 4.6, 4.1, 3.4, 2.8]
    lower_z = 0.6
    gray = "#CBD0D6"
    style = dict(color=gray, lw=3.2, zorder=0, solid_capstyle="round")
    ax.plot(station_x, upper_z, **style)
    ax.plot([-17.5, 17.5], [lower_z, lower_z], **style)
    for x, z in zip(station_x, upper_z):
        ax.plot([x, x], [lower_z, z], color=gray, lw=2.0, zorder=0)
    # Alternating diagonals follow the instrumented wall shown in the layout.
    start_upper = wall > 0
    for j in range(len(station_x) - 1):
        upper_left = (j % 2 == 0) == start_upper
        za = upper_z[j] if upper_left else lower_z
        zb = lower_z if upper_left else upper_z[j + 1]
        ax.plot(station_x[j:j + 2], [za, zb], color=gray, lw=2.0, zorder=0)
    for x in (-17.5, 17.5):
        ax.add_patch(Polygon([[x, lower_z], [x - 0.5, 0.0], [x + 0.5, 0.0]],
                             closed=True, facecolor="white", edgecolor="#9CA3AF",
                             lw=0.9, zorder=1))
        ax.plot([x - 0.75, x + 0.75], [-0.12, -0.12], color="#9CA3AF", lw=0.9, zorder=1)


# =============================================================================
# Figure 6
# =============================================================================
fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.45], hspace=0.42)
xlim = (-19.0, 19.0)

for panel, (wall, title, ids) in enumerate((
    (2.25, "(a) North truss wall, elevation ($y = +2.25$ m)", range(1, 10)),
    (-2.25, "(b) South truss wall, elevation ($y = -2.25$ m)", range(10, 19)),
)):
    ax = fig.add_subplot(gs[panel])
    draw_bridge_outline(ax, wall)
    proj = lambda c: (c[0], c[2])  # noqa: E731
    draw_edges(ax, ("truss_chord", "truss_diagonal"), proj, wall=wall)
    ax.plot([-15.75, 15.75], [0.19, 0.19], color=RELATION_STYLE["deck_longitudinal"][0], lw=1.2, alpha=0.6, zorder=1)
    for idx in ids:
        name = f"AG{idx:02d}"
        x, _, z = coords[name]
        ax.scatter([x], [z], s=34, facecolor="white", edgecolor="#1E3A5F", linewidth=1.3, zorder=5)
        ax.text(x, z + (0.45 if z > 1.0 else -0.62), name, ha="center", va="center", fontsize=7.2)
    triad(ax, -18.1, 4.1, 1.4, 1.5, "$x$", "$z$")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.9, 6.0)
    ax.set_ylabel("$z$ (m)")
    ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold")
    ax.grid(False)
    ax.set_xticks(range(-14, 15, 7))
    ax.tick_params(labelbottom=True)

ax = fig.add_subplot(gs[2])
proj = lambda c: (c[0], c[1])  # noqa: E731
draw_edges(ax, ("deck_transverse", "deck_longitudinal", "deck_truss_interface", "lateral_bracing", "truss_chord"), proj)
for name, (x, y, z) in coords.items():
    if name.startswith("AG"):
        if z < 1.0:  # lower-chord joints carry the plan-view connections
            ax.scatter([x], [y], s=30, facecolor="white", edgecolor="#1E3A5F", linewidth=1.2, zorder=5)
            ax.text(x, y + (0.42 if y > 0 else -0.42), name, ha="center", va="center", fontsize=6.8)
    else:
        ax.scatter([x], [y], s=16, marker="s", facecolor="white", edgecolor="#B7791F", linewidth=1.0, zorder=5)
for station in range(10):
    x = -15.75 + L_BAY * station
    ax.text(x, -3.25, f"AL{4 * station + 1:02d}--{4 * station + 4:02d}".replace("--", "–"),
            ha="center", va="center", fontsize=6.6, rotation=0)
for name, x in SHAKER_X.items():
    ax.scatter([x], [0.0], **SHAKER_STYLE)
    ax.text(x + 0.45, -0.12, name, ha="left", va="top", fontsize=8.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85), zorder=9)
triad(ax, -17.6, -1.6, 1.4, 1.3, "$x$", "$y$")
ax.set_xlim(*xlim)
ax.set_ylim(-3.7, 3.1)
ax.set_xlabel("Longitudinal coordinate $x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.set_title("(c) Plan view: deck grid, lateral bracing and deck--truss interfaces".replace("--", "–"),
             loc="left", fontsize=8.8, fontweight="bold")
ax.grid(True, color="#E5E7EB", lw=0.5)
ax.set_xticks(range(-14, 15, 7))

handles = [Line2D([], [], color=c, ls=ls, lw=lw, label=lab.replace("--", "–")) for c, ls, lw, lab in RELATION_STYLE.values()]
handles += [
    Line2D([], [], color="#CBD0D6", lw=3.2, label="Physical bridge outline (schematic)"),
    Line2D([], [], ls="none", marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#1E3A5F", label="Truss sensor (AG)"),
    Line2D([], [], ls="none", marker="s", markersize=5, markerfacecolor="white", markeredgecolor="#B7791F", label="Deck sensor (AL)"),
    Line2D([], [], ls="none", marker="*", markersize=10, color="black", label="Shaker positions P1, P2"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035),
           columnspacing=1.2, handlelength=2.2)
fig.savefig(FIG_DIR / "hbta_sensor_topology_R2.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved Figure 6")


# Exact plotted sensor coordinates are inherited unchanged from R1.
