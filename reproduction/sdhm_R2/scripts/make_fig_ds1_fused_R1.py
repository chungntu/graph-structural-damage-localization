"""R1 correction of Figure 9: fused (P1+P2) DS1 spectrum of s_e.

The submitted figure fell back to the P2 spectrum because the fused entry was
missing from its input file; here the fused spectrum is computed directly.
Style follows the submitted figure.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r1_data import FIG_DIR, fused_ds1_spectrum

plt.rcParams.update({"font.sans-serif": "Helvetica, Arial, DejaVu Sans", "font.family": "sans-serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 1.0})
spec = fused_ds1_spectrum()
scores = spec["scores"]
t = spec["target_index"]
idx = np.arange(1, len(scores) + 1)
floor = float(np.median(np.delete(scores, t)))
colors = ["#2979FF" if s > 3 * floor else "#B0BEC5" for s in scores]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.vlines(idx, 0, scores, colors=colors, lw=1.3, alpha=0.85)
ax.vlines(t + 1, 0, scores[t], colors="#D32F2F", lw=1.6)
ax.scatter([t + 1], [scores[t]], color="#D32F2F", s=60, zorder=5, edgecolor="black", lw=0.8)
ax.annotate(f"DS1 target (AL14–AL18)\nRank {spec['target_rank']} of {len(scores)}, $s_e = {scores[t]:.3f}$",
            xy=(t + 1, scores[t]), xytext=(t - 62, scores[t] * 0.78),
            arrowprops=dict(facecolor="#D32F2F", shrink=0.06, width=1.2, headwidth=6),
            fontsize=8.5, fontweight="bold", color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#EF9A9A"))
ax.axhline(spec["median_score"], color="#546E7A", ls="--", lw=1.0)
ax.text(len(scores) + 1, spec["median_score"] + 0.004, f"median $= {spec['median_score']:.3f}$",
        ha="right", va="bottom", fontsize=8, color="#546E7A")
ax.set_xlim(0, len(scores) + 2)
ax.set_ylim(0, scores[t] * 1.18)
ax.set_xlabel(r"Candidate edge index $e \in \{1, \ldots, 122\}$", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Relative-motion score $s_e$\n(fused P1 + P2)", fontsize=9.5, fontweight="bold")
ax.grid(axis="y", ls="--", alpha=0.35)
ax.tick_params(labelsize=8.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_ds1_confirmatory_R1.png", dpi=600, bbox_inches="tight")
print("saved Figure 9:", f"target {scores[t]:.4f}, rank {spec['target_rank']}, median {spec['median_score']:.4f}")
