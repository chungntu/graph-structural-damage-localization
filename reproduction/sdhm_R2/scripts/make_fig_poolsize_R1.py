"""Figure 14: candidate pool-size sensitivity (same figure as submitted).

Extracted from the original combined robustness script so that the revision
code contains only the plotting code for figures used in the manuscript.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r1_data import FIG_DIR, OUTPUTS

plt.rcParams["font.sans-serif"] = "Helvetica, Arial, DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.linewidth"] = 1.0

k_sens = json.loads((OUTPUTS / "topk_spatial_localization.json").read_text(encoding="utf-8"))["k_sensitivity_at_primary_bandwidth"]
k_vals = np.array(sorted(int(k) for k in k_sens))
k_errs = np.array([k_sens[str(k)]["mean_absolute_error_m"] for k in k_vals])
k_accs = np.array([k_sens[str(k)]["state_fraction_within_one_panel"] * 100 for k in k_vals])
color_err, color_acc = "#D32F2F", "#1565C0"

fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=300)
l1 = ax.plot(k_vals, k_errs, color=color_err, lw=2.0, marker="s", markersize=5.0, label="Mean Absolute Error (m)", zorder=4)
ax.set_xlabel(r"Candidate pool size $K$", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Mean Absolute Error (m)", color=color_err, fontsize=9.5, fontweight="bold")
ax.tick_params(axis="y", labelcolor=color_err, labelsize=8.5)
ax.tick_params(axis="x", labelsize=8.5)
ax.set_ylim(1.0, 8.5)
ax.set_xlim(4.3, 15.7)
ax.set_xticks(range(5, 16))
ax2 = ax.twinx()
l2 = ax2.plot(k_vals, k_accs, color=color_acc, lw=2.0, marker="o", markersize=5.0, label="Localization Accuracy (%)", zorder=4)
ax2.set_ylabel("Localization Accuracy (%)", color=color_acc, fontsize=9.5, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color_acc, labelsize=8.5)
ax2.set_ylim(-5, 108)
ax.axvspan(7.5, 9.5, alpha=0.12, color="#4CAF50", zorder=0)
ax.text(8.5, 7.8, "Optimal Pool", ha="center", va="center", fontsize=8.5, color="#1B5E20", fontweight="bold")
ax2.plot([8, 9], [87.5, 87.5], marker="*", markersize=12, color="#FFD700", markeredgecolor="#1565C0",
         markeredgewidth=1.4, zorder=6, linestyle="none")
lines = l1 + l2
ax.legend(lines, [line.get_label() for line in lines], loc="lower left", bbox_to_anchor=(0.02, 0.05), fontsize=8.0, framealpha=0.95)
ax.grid(True, which="both", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_robustness_audits_poolsize.png", dpi=300)
print("saved Figure 14")
