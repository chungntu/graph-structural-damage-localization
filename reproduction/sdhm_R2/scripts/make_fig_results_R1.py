"""R1 versions of three result figures.

Figure 10: recomputed with the top-ranked connection (Section 4.2); P1/P2 named
           by their measured positions; damage-type sub-labels
           removed from the axis (types are now given in Table 1).
Figure 11: same content and style as submitted (reproducible plotting code).
Figure 12: P1 named by its measured position; the highlighted null trials and
           p-values follow Eq. (20) (Reviewer 1, comment 2).
Figure 13: the a priori benchmark bandwidth and the post hoc MAE minimum are
           marked separately; "optimal band" labels removed (Reviewer 1,
           comment 3).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r1_data import FIG_DIR, L_BAY, OUTPUTS, discrete_benchmark, load_permutation_runs

P1_LABEL = "P1 (off-midspan, $x = +7.0$ m)"
P2_LABEL = "P2 (midspan, $x = 0$)"

# =============================================================================
# Figure 10 -- discrete benchmark errors (spatial_region_localization.json)
# =============================================================================
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "mathtext.fontset": "cm"})
# R1 correction: top-ranked connection per position, as defined in Section 4.2
# (the submitted figure had used a Gaussian-smoothed profile instead).
bench = discrete_benchmark()
(FIG_DIR.parent / "scripts" / "r1_discrete_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
labels, p1, p2 = [], [], []
for s in [f"DS{i}" for i in range(1, 9)]:
    e1 = bench[s]["P1"]["errors_m"]
    e2 = bench[s]["P2"]["errors_m"]
    for site in range(len(e1)):
        labels.append(s if len(e1) == 1 else f"{s}\n(site {site + 1})")
        p1.append(e1[site])
        p2.append(e2[site])
fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(labels))
w = 0.36
r1 = ax.bar(x - w / 2, p1, w, label=P1_LABEL, color="#d95f02", alpha=0.9, edgecolor="#8c3800", lw=0.8)
r2 = ax.bar(x + w / 2, p2, w, label=P2_LABEL, color="#1f78b4", alpha=0.9, edgecolor="#0e4b75", lw=0.8)
tol = ax.axhline(L_BAY, color="#2ca02c", ls="--", lw=1.6, zorder=4,
                 label=r"Single-bay tolerance $\delta_{\mathrm{tol}} = L_{\mathrm{bay}}$")
ax.axhspan(0, L_BAY, color="#2ca02c", alpha=0.08, zorder=0)
ax.set_ylabel("Absolute localization error (m)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 18.5)
ax.grid(axis="y", ls=":", alpha=0.55)
ax.legend(handles=[r1, r2, tol], loc="upper right", framealpha=0.95)
for bars in (r1, r2):
    for rect in bars:
        if rect.get_height() < 0.05:  # exact identification: label the otherwise invisible bar
            ax.text(rect.get_x() + rect.get_width() / 2, 0.12, "0.00", ha="center", va="bottom",
                    fontsize=8, color="#0e4b75", rotation=90)
for i in range(len(labels)):
    ok = p1[i] <= L_BAY + 1e-9 and p2[i] <= L_BAY + 1e-9
    ax.text(x[i], max(p1[i], p2[i]) + 0.45, "PASS" if ok else "FAIL", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1b7837" if ok else "#b2182b")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_single_edge_ds_benchmark_R1.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved Figure 10")

# =============================================================================
# Figure 11 -- continuous benchmark errors (topk_spatial_localization.json),
# same content and style as the submitted figure
# =============================================================================
topk = json.loads((OUTPUTS / "topk_spatial_localization.json").read_text(encoding="utf-8"))
tol = topk["success_tolerance_m"]
labels, errs = [], []
for s in [f"DS{i}" for i in range(1, 9)]:
    e = topk["primary"]["states"][s]["matched_absolute_errors_m"]
    for site, err in enumerate(e):
        labels.append(s if len(e) == 1 else f"{s}\n(Site {site + 1})")
        errs.append(err)
state_ok = {s: all(err <= tol for err in topk["primary"]["states"][s]["matched_absolute_errors_m"])
            for s in [f"DS{i}" for i in range(1, 9)]}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(labels))
colors = ["#43A843" if state_ok[lab.split("\n")[0]] else "#D93F3F" for lab in labels]
bars = ax.bar(x, errs, 0.52, color=colors, edgecolor="#222222", lw=0.85)
tol_line = ax.axhline(tol, color="#2ca02c", ls="--", lw=1.6, zorder=4,
                      label=r"Single-Bay Tolerance $\delta_{\mathrm{tol}} = L_{\mathrm{bay}}$")
ax.axhspan(0, tol, color="#2ca02c", alpha=0.08, zorder=0)
bars.set_label(r"Top-$K$ Restricted Spatial Kernel Smoothing $R(x)$")
ax.set_ylabel("Absolute Localization Error (m)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 10.1)
ax.grid(axis="y", ls=":", alpha=0.55)
ax.legend(handles=[tol_line, bars], loc="upper left", framealpha=0.95)
for i, e in enumerate(errs):
    ok = e <= tol
    ax.text(x[i], e + 0.30, "PASS" if ok else "FAIL", ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#1b7837" if ok else "#b2182b")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_spatial_smoothing_ds_benchmark.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved Figure 11")

# =============================================================================
# Figure 12 -- topology permutation (ds1_sweep_validation.json runs)
# =============================================================================
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({"font.sans-serif": "Helvetica, Arial, DejaVu Sans", "font.family": "sans-serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 1.0})
runs = load_permutation_runs()
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4), sharey=True)
bins = np.arange(0.5, 123.5, 1)
ranks = np.arange(1, 123)
for ax, pos, title, color, edge in (
    (axes[0], "P1", "(a) Off-midspan excitation P1 ($x = +7.0$ m)", "#F57C00", "#E65100"),
    (axes[1], "P2", "(b) Midspan excitation P2 ($x = 0$)", "#D32F2F", "#B71C1C"),
):
    null = runs[pos]["null_ranks"]
    obs = runs[pos]["observed_rank"]
    counts, _ = np.histogram(null, bins=bins)
    n_le = int(np.sum(null <= obs))
    p = (1 + n_le) / (1 + len(null))
    ax.bar(ranks, counts, width=0.82, color="#CFD8DC", edgecolor="#78909C", lw=0.35, alpha=0.85,
           label="Randomized topologies ($R = 1000$)")
    ax.bar(ranks[:obs], counts[:obs], width=0.82, color=color, edgecolor=edge, lw=0.9, zorder=5,
           label=(rf"Trials with Rank $\leq {obs}$: {n_le}/1000" if obs > 1 else f"Trials with Rank $= 1$: {n_le}/1000")
           + f"\n(observed Rank {obs}; $p = {p:.3f}$)")
    ax.set_xlabel(r"Diagnostic rank ($k = 1, \dots, 122$)", fontsize=9.0, fontweight="bold")
    ax.set_title(title, fontsize=9.8, fontweight="bold", pad=7)
    ax.set_xlim(0, 123)
    ax.set_ylim(0, 26)
    ax.grid(True, ls="--", alpha=0.35)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
axes[0].set_ylabel("Trial count", fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_robustness_topology_permutation_R1.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved Figure 12")

# =============================================================================
# Figure 13 -- bandwidth sensitivity (bandwidth_logscale_sweep.json)
# =============================================================================
sweep = json.loads((OUTPUTS / "revision_r1" / "bandwidth_logscale_sweep.json").read_text(encoding="utf-8"))
h_norm = np.array([v["bandwidth_m"] for v in sweep.values()]) / L_BAY
mae = np.array([v["mean_absolute_error_m"] for v in sweep.values()])
acc = np.array([v["state_fraction_within_one_panel"] * 100 for v in sweep.values()])
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.set_xscale("log")
l1 = ax.plot(h_norm, mae, color="#D32F2F", lw=2.0, marker="s", markersize=4.5, label="Mean absolute error (m)", zorder=4)
ax.set_xlabel(r"Normalized spatial kernel bandwidth $h / L_{\mathrm{bay}}$", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Mean absolute error (m)", color="#D32F2F", fontsize=9.5, fontweight="bold")
ax.tick_params(axis="y", labelcolor="#D32F2F", labelsize=8.5)
ax.set_ylim(1.0, 13.0)
ax.set_xlim(0.005, 11.5)
ax2 = ax.twinx()
l2 = ax2.plot(h_norm, acc, color="#1565C0", lw=2.0, marker="o", markersize=4.5,
              label="States within tolerance (%)", zorder=4)
ax2.set_ylabel("States within tolerance (%)", color="#1565C0", fontsize=9.5, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="#1565C0", labelsize=8.5)
ax2.set_ylim(-5, 108)
bench = ax.axvline(0.5, color="black", ls="--", lw=1.4, zorder=3,
                   label=r"Benchmark setting $h = L_{\mathrm{bay}}/2$ (fixed a priori)")
i_min = int(np.argmin(mae))
star = ax.plot(h_norm[i_min], mae[i_min], ls="none", marker="*", markersize=13, color="#FFD700",
               markeredgecolor="#B71C1C", markeredgewidth=1.3, zorder=6,
               label=rf"Lowest MAE in this post hoc sweep ($h \approx L_{{\mathrm{{bay}}}}/4$)")
handles = l1 + l2 + [bench] + star
ax.legend(handles, [h.get_label() for h in handles], loc="center left", bbox_to_anchor=(0.01, 0.5), fontsize=7.8, framealpha=0.95)
ax.grid(True, which="both", ls="--", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_robustness_audits_bandwidth_R1.png", dpi=600)
plt.close(fig)
print("saved Figure 13")
