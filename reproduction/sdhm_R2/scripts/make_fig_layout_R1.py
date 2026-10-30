"""R1 redraw of Figures 6 and 7 (Reviewer 2, comment 2).

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


# =============================================================================
# Figure 6
# =============================================================================
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.45], hspace=0.42)
xlim = (-18.2, 17.4)

for panel, (wall, title, ids) in enumerate((
    (2.25, "(a) North truss wall, elevation ($y = +2.25$ m)", range(1, 10)),
    (-2.25, "(b) South truss wall, elevation ($y = -2.25$ m)", range(10, 19)),
)):
    ax = fig.add_subplot(gs[panel])
    proj = lambda c: (c[0], c[2])  # noqa: E731
    draw_edges(ax, ("truss_chord", "truss_diagonal"), proj, wall=wall)
    ax.plot([-15.75, 15.75], [0.19, 0.19], color=RELATION_STYLE["deck_longitudinal"][0], lw=1.2, alpha=0.6, zorder=1)
    for idx in ids:
        name = f"AG{idx:02d}"
        x, _, z = coords[name]
        ax.scatter([x], [z], s=34, facecolor="white", edgecolor="#1E3A5F", linewidth=1.3, zorder=5)
        ax.text(x, z + (0.45 if z > 1.0 else -0.62), name, ha="center", va="center", fontsize=7.2)
    triad(ax, -17.6, 2.3, 1.4, 1.5, "$x$", "$z$")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.9, 6.0)
    ax.set_ylabel("$z$ (m)")
    ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold")
    ax.grid(True, color="#E5E7EB", lw=0.5)
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
    Line2D([], [], ls="none", marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#1E3A5F", label="Truss sensor (AG)"),
    Line2D([], [], ls="none", marker="s", markersize=5, markerfacecolor="white", markeredgecolor="#B7791F", label="Deck sensor (AL)"),
    Line2D([], [], ls="none", marker="*", markersize=10, color="black", label="Shaker positions P1, P2"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035),
           columnspacing=1.2, handlelength=2.2)
fig.savefig(FIG_DIR / "hbta_sensor_topology_R1.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved Figure 6")

# =============================================================================
# Figure 7
# =============================================================================
truth = locked_ground_truth()
fig, ax = plt.subplots(figsize=(7.2, 3.9))
for row in rows:
    a, b = coords[row["sensor_i"]], coords[row["sensor_j"]]
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#CBD5E1", lw=0.9, zorder=1)
for name, (x, y, z) in coords.items():
    if name.startswith("AL") or z < 1.0:
        ax.scatter([x], [y], s=10, color="#475569", zorder=2)

# DS1/DS2: the instrumented stringer path bracketing the connection.
a, b = coords["AL14"], coords["AL18"]
ax.plot([a[0], b[0]], [a[1], b[1]], color=TYPE_COLOR[DAMAGE_TYPE["DS1"]], lw=3.2, zorder=4, solid_capstyle="round")

label_rows = {"DS3": 0, "DS4": 0, "DS6": 0, "DS8": 0, "DS5": 1, "DS1": 2}
label_text = {"DS1": "DS1, DS2", "DS3": "DS3", "DS4": "DS4", "DS5": "DS5, DS7", "DS6": "DS6, DS7", "DS8": "DS8"}
for state, level in label_rows.items():
    x = truth[state][0]
    color = TYPE_COLOR[DAMAGE_TYPE[state]]
    ax.plot([x, x], [-2.6, 2.6 + 0.62 * level], color=color, ls="--", lw=1.4, zorder=3)
    ax.text(x, 2.85 + 0.62 * level, label_text[state], ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, lw=0.8), zorder=6)
for name, x in SHAKER_X.items():
    ax.scatter([x], [0.0], **SHAKER_STYLE)
    ax.text(x + 0.45, -0.32, name, ha="left", va="top", fontsize=8.0, fontweight="bold", zorder=9)
triad(ax, -17.6, -2.4, 1.4, 1.2, "$x$", "$y$")
ax.set_xlim(*xlim)
ax.set_ylim(-3.0, 4.6)
ax.set_xlabel("Longitudinal coordinate $x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.set_xticks(range(-14, 15, 7))
ax.set_yticks([-2, 0, 2])
ax.grid(True, color="#F1F5F9", lw=0.5)
handles = [Line2D([], [], color=c, ls="--", lw=1.4, label=t) for t, c in TYPE_COLOR.items()]
handles += [
    Line2D([], [], color=TYPE_COLOR[DAMAGE_TYPE["DS1"]], lw=3.0, label="Candidate edge AL14–AL18 (DS1, DS2)"),
    Line2D([], [], color="#CBD5E1", lw=1.0, marker="o", markersize=3, markerfacecolor="#475569",
           markeredgecolor="#475569", label="Candidate graph (plan projection)"),
    Line2D([], [], ls="none", marker="*", markersize=10, color="black", label="Shaker positions P1, P2"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2), columnspacing=1.5)
fig.savefig(FIG_DIR / "hbta_ds1_sensor_graph_R1.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved Figure 7")
