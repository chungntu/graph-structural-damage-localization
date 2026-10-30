"""make_fig_modal_fusion.py
Conceptual workflow of Multi-Reference Modal Vector Fusion illustrated on the 5-panel Warren truss.
- Adjusted y-limits and legend placement in Panel (c) to completely eliminate any overlap.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Standardised Color Palette
C_CONT = "#2B6CB0"          # Primary Continuous Blue (Fused consensus)
C_BLUE = "#1D4ED8"          # Setup 1 (Cobalt Blue)
C_AMBER = "#D97706"         # Setup 2 (Rich Amber)
C_TEAL = "#0F766E"          # Setup 3 (Deep Teal)
C_PURPLE = "#7C3AED"        # Setup 4 (Vivid Purple)
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


def style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(False)


# Truss Geometry (9 nodes total)
nodes_0 = np.array([
    [-8.0, 0.0],  # 0
    [-4.0, 0.0],  # 1
    [ 0.0, 0.0],  # 2
    [ 4.0, 0.0],  # 3
    [ 8.0, 0.0],  # 4
    [-6.0, 2.0],  # 5
    [-2.0, 2.0],  # 6
    [ 2.0, 2.0],  # 7
    [ 6.0, 2.0],  # 8
])

# 15 member connectivity
edges = [
    # Top chords
    (5, 6), (6, 7), (7, 8),
    # Bottom chords
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Diagonals
    (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8)
]

# True fundamental vertical bending mode shape
L = 16.0
x_coords = nodes_0[:, 0]
x_norm = (x_coords + 8.0) / L  # 0 to 1
phi_true = np.sin(np.pi * x_norm)
phi_true = phi_true / np.linalg.norm(phi_true)  # unit norm

# 4 Setups with wide dispersion in participation scales and noise
np.random.seed(42)
setups = [
    {"name": "Setup 1", "color": C_BLUE,   "scale": 1.35, "theta": 0.05,           "noise_mag": 0.040},
    {"name": "Setup 2", "color": C_AMBER,  "scale": 0.65, "theta": np.pi - 0.10,   "noise_mag": 0.035},
    {"name": "Setup 3", "color": C_TEAL,   "scale": 1.12, "theta": 0.30,          "noise_mag": 0.045},
    {"name": "Setup 4", "color": C_PURPLE, "scale": 0.88, "theta": -np.pi + 0.15, "noise_mag": 0.040},
]

for s in setups:
    noise_vec = np.random.normal(0, s["noise_mag"], len(phi_true))
    z_complex = s["scale"] * phi_true * np.exp(1j * s["theta"]) + noise_vec * (1 + 1j * 0.5)
    s["z_raw"] = np.real(z_complex)
    
    # Phase alignment
    th_opt = np.angle(np.vdot(phi_true, z_complex))
    z_al = np.real(z_complex * np.exp(-1j * th_opt))
    s["z_aligned"] = z_al

# Fused consensus mode shape
all_aligned_stack = np.array([s["z_aligned"] for s in setups])
mean_aligned = np.mean(all_aligned_stack, axis=0)
u_fused = mean_aligned / np.linalg.norm(mean_aligned) * 1.0

# Displacement visual amplification factor
AMP = 1.6

# ----------------------------------------------------------------------------
# Figure Layout
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.05]})
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.03, right=0.97, hspace=0.36)


def draw_truss(ax: plt.Axes, node_pos: np.ndarray, color: str, lw: float = 1.5, alpha: float = 0.9,
               node_size: float = 24, zorder: int = 2) -> None:
    for (i, j) in edges:
        p1, p2 = node_pos[i], node_pos[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls="-", alpha=alpha, zorder=zorder)
    if node_size > 0:
        ax.scatter(node_pos[:, 0], node_pos[:, 1], color=color, s=node_size, alpha=alpha, zorder=zorder + 1)


# ============================================================================
# Panel (a): Raw Unaligned Modal Deformations on Truss Structure
# ============================================================================
ax0 = axes[0]

handles_a = []
for s in setups:
    nodes_disp = np.copy(nodes_0)
    nodes_disp[:, 1] += AMP * s["z_raw"]
    draw_truss(ax0, nodes_disp, color=s["color"], lw=1.5, alpha=0.88, node_size=22, zorder=3)
    l, = ax0.plot([], [], color=s["color"], lw=1.5, label=s["name"])
    handles_a.append(l)

ax0.set_title(r"(a) Raw unaligned modal vectors $\mathbf{z}_q$", loc="left", fontsize=9.2, fontweight="bold", color=DARK, pad=5)
ax0.set_ylim(-1.6, 3.6)
ax0.legend(handles=handles_a, loc="lower right", bbox_to_anchor=(1.0, 1.02),
           ncol=4, frameon=False, fontsize=8.0, handletextpad=0.3, columnspacing=1.0)
style(ax0)

# ============================================================================
# Panel (b): Phase-Aligned Modal Deformations (Widely Separated)
# ============================================================================
ax1 = axes[1]

for s in setups:
    nodes_disp = np.copy(nodes_0)
    nodes_disp[:, 1] += AMP * s["z_aligned"]
    draw_truss(ax1, nodes_disp, color=s["color"], lw=1.5, alpha=0.88, node_size=20, zorder=3)

ax1.set_title(r"(b) Phase-aligned modal vectors $\mathbf{z}_q^{\mathrm{al}}$", loc="left", fontsize=9.2, fontweight="bold", color=DARK, pad=5)
ax1.set_ylim(-0.6, 4.2)
style(ax1)

# ============================================================================
# Panel (c): Converged Fused Consensus Truss u* (Clean, NO overlap with legend)
# ============================================================================
ax2 = axes[2]

# Compute min and max deformed node positions across all 4 setups
all_deformed_nodes = []
for s in setups:
    nodes_disp = np.copy(nodes_0)
    nodes_disp[:, 1] += AMP * s["z_aligned"]
    all_deformed_nodes.append(nodes_disp)
all_deformed_nodes = np.array(all_deformed_nodes)

min_y = np.min(all_deformed_nodes[:, :, 1], axis=0)
max_y = np.max(all_deformed_nodes[:, :, 1], axis=0)

# Large prominent shaded range for bottom chord (nodes 0..4)
bot_idx = [0, 1, 2, 3, 4]
patch_range = ax2.fill_between(nodes_0[bot_idx, 0], min_y[bot_idx] - 0.08, max_y[bot_idx] + 0.08,
                               color=C_CONT, alpha=0.22, label="Multi-setup range")

# Large prominent shaded range for top chord (nodes 5..8)
top_idx = [5, 6, 7, 8]
ax2.fill_between(nodes_0[top_idx, 0], min_y[top_idx] - 0.08, max_y[top_idx] + 0.08,
                 color=C_CONT, alpha=0.22)

# Fused consensus deformed truss
nodes_fused = np.copy(nodes_0)
nodes_fused[:, 1] += AMP * u_fused
draw_truss(ax2, nodes_fused, color=C_CONT, lw=2.5, alpha=0.98, node_size=38, zorder=4)

# Legend handles placed outside top right (or top margin with zero overlap)
line_fused, = ax2.plot([], [], color=C_CONT, lw=2.5, marker="o", markersize=5.5, label=r"Fused consensus $\mathbf{u}^*$")

# Ample ylim so legend sits freely with zero obstruction
ax2.set_ylim(-0.6, 4.6)
ax2.legend(handles=[line_fused, patch_range], loc="upper right", frameon=False, fontsize=8.0, handletextpad=0.4, columnspacing=1.0)

ax2.set_title(r"(c) Converged fused consensus modal vector $\mathbf{u}^*$", loc="left", fontsize=9.2, fontweight="bold", color=DARK, pad=5)
ax2.set_xlabel(r"Sensor coordinate $x$", fontsize=9.5)
ax2.set_xlim(-9, 9)
style(ax2)

# ---------------------------------------------------------------------------
# Coordinate system (R1, Reviewer 2 comment 1): x longitudinal, z vertical.
# Nodal modal amplitudes are drawn as vertical (z) offsets of the truss nodes.
# ---------------------------------------------------------------------------
def draw_triad(ax: plt.Axes, x0: float, y0: float, lx: float = 0.9, ly: float = 0.9) -> None:
    for (dx, dy, lab) in ((lx, 0.0, "$x$"), (0.0, ly, "$z$")):
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.0, mutation_scale=8), zorder=6)
        ax.text(x0 + dx + (0.25 if dx else 0.0), y0 + dy + (0.35 if dy else 0.0), lab,
                fontsize=8.2, ha="center", va="center", color=DARK)


draw_triad(ax0, -10.4, -1.2)
draw_triad(ax1, -10.4, -0.2)
draw_triad(ax2, -10.4, -0.2)
ax2.set_xlim(-10.9, 9)
ax2.tick_params(axis="x", bottom=True, labelbottom=True)
ax2.set_xticks([-8, -4, 0, 4, 8])
ax2.set_xticklabels(["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
ax2.set_xlabel(r"Longitudinal sensor coordinate $x$", fontsize=9.5)

import os

# R1: write only into the revision folder; the submitted figures stay untouched.
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(os.path.dirname(script_dir), "figures", "figure1_modal_fusion_concept.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
print(f"Saved {out_path}")
