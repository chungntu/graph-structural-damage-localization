"""R1 correction of Figure 8.

Panel (a) now shows a measured swept-sine FRF together with the three-mode
rational-fraction model actually fitted to it (same estimator and fitting code
as the modal-feature extraction). The submitted panel had shown a curve built
from the modal parameters with an arbitrary scale and offset, against an FRF
from a random-excitation record. Panels (b)-(d) now plot the consensus vector
obtained with the fusion of Eq. (12) instead of a plain complex average.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from r1_data import CODE_ROOT, FIG_DIR
from hbta_frf.config import INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ, RESPONSE_SENSORS
from hbta_frf.frf import estimate_h1_frf, load_vertical_recording
from hbta_frf.graph_damage_gp import fit_mode_shape_template
from hbta_frf.modal_model import fit_complex_residues, fit_shared_pole_model, modal_basis

import os
from pathlib import Path

# Raw benchmark file (Zenodo, doi:10.5281/zenodo.10507956); override with HBTA_RAW_HDF5.
RAW_HDF5 = Path(os.environ.get("HBTA_RAW_HDF5", "data_100Hz.h5"))
RECORDING = "MVS_P1_UDS_SM_Z_01"   # healthy, off-midspan excitation: all three modes well excited
WINDOW_S = (45.0, 285.0)           # increasing-frequency pass
SENSOR = "AL20"
COHERENCE_THRESHOLD = 0.5

# -----------------------------------------------------------------------------
# (a) measured FRF and fitted model, exactly as in the feature extraction
# -----------------------------------------------------------------------------
cache = np.load(CODE_ROOT / "cache" / "all_damage_sweep_modal_features.npz")
selected = cache["selected_sensor_indices"]
response, shaker, fs = load_vertical_recording(RAW_HDF5, RECORDING, RESPONSE_SENSORS)
a, b = int(round(WINDOW_S[0] * fs)), int(round(WINDOW_S[1] * fs))
freqs, frf, coherence, _ = estimate_h1_frf(response[None, :, a:b], shaker[None, a:b], fs, 2048, 0.75, 1.0, 30.0)
band = (freqs >= 5.0) & (freqs <= 26.0)
f_band, H, C = freqs[band], frf[0][:, band], coherence[0][:, band]
fit = fit_shared_pole_model(f_band, H[selected], np.where(C[selected] >= COHERENCE_THRESHOLD, C[selected], 0.0),
                            INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ)
weights = np.where(C >= COHERENCE_THRESHOLD, C, 0.0)
residues, prediction = fit_complex_residues(modal_basis(f_band, fit.modal_frequencies_hz, fit.damping_ratios), H, weights)
s = RESPONSE_SENSORS.index(SENSOR)
nmse = float(np.sum(weights[s] * np.abs(H[s] - prediction[s]) ** 2) / np.sum(weights[s] * np.abs(H[s]) ** 2))
f_fine = np.linspace(5.0, 26.0, 1600)
H_fit = modal_basis(f_fine, fit.modal_frequencies_hz, fit.damping_ratios) @ residues[s]
mag_meas = 20 * np.log10(np.abs(H[s]))
mag_fit = 20 * np.log10(np.abs(H_fit))
print(f"{RECORDING} {SENSOR}: modes {np.round(fit.modal_frequencies_hz, 2)}, NMSE {nmse:.4f}")

# -----------------------------------------------------------------------------
# (b)-(d) modal deflection profiles (as in the submitted figure)
# -----------------------------------------------------------------------------
modal_data = np.load(CODE_ROOT / "cache" / "ds1_sweep_modal_features.npz")
sensors = list(modal_data["response_sensor_names"])
modal_freqs = np.mean(modal_data["modal_frequencies_hz"], axis=0)
shapes = modal_data["mode_shapes"]
positions = modal_data["positions"]
states = modal_data["states"]
shapes_p1 = shapes[(states == "UDS") & (positions == "P1")]
shapes_p2 = shapes[(states == "UDS") & (positions == "P2")]
station_x = np.array([-15.75, -12.25, -8.75, -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75])


def extract_longitudinal_profile(shape_complex):
    dominant_phase = np.angle(shape_complex[np.argmax(np.abs(shape_complex))])
    shape_real = np.real(shape_complex * np.exp(-1j * dominant_phase))
    shape_real /= np.max(np.abs(shape_real))
    return np.array([np.mean([shape_real[sensors.index(f"AL{st * 4 + y + 1:02d}")] for y in range(4)]) for st in range(10)])


p1_mean = [np.mean([extract_longitudinal_profile(x[m]) for x in shapes_p1], axis=0) for m in range(3)]
p2_mean = [np.mean([extract_longitudinal_profile(x[m]) for x in shapes_p2], axis=0) for m in range(3)]
# The consensus vector is the iterative multi-reference fusion of Eq. (12) over
# all healthy passes (the submitted figure had plotted a plain complex average).
eq12 = fit_mode_shape_template(shapes[states == "UDS"])
fused_profiles = [extract_longitudinal_profile(eq12[m]) for m in range(3)]

# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9.2, "axes.titlesize": 9.8,
    "axes.titleweight": "bold", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
    "axes.edgecolor": "#1E293B", "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
})
fig = plt.figure(figsize=(13.8, 7.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
ax_top = fig.add_subplot(gs[0, :])
colors_m = ["#DC2626", "#059669", "#7C3AED"]
ax_top.plot(f_band, mag_meas, color="#94A3B8", lw=1.3)
ax_top.plot(f_fine, mag_fit, color="#1D4ED8", lw=2.2)
for m, fm in enumerate(fit.modal_frequencies_hz):
    ax_top.axvline(fm, color=colors_m[m], ls="--", lw=1.0, alpha=0.75)
    y_p = mag_fit[np.argmin(np.abs(f_fine - fm))]
    ax_top.scatter([fm], [y_p], color=colors_m[m], s=48, edgecolors="white", linewidths=1.0, zorder=6)
    ax_top.annotate(f"Mode {m + 1}", xy=(fm, y_p), xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9.2, fontweight="bold", color=colors_m[m], zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))
ax_top.set_xlabel("Excitation frequency (Hz)")
ax_top.set_ylabel("FRF magnitude (dB)")
ax_top.set_title(f"(a) Measured swept-sine FRF and fitted 3-mode model (sensor {SENSOR}, healthy, P1, increasing-frequency pass)",
                 loc="left")
ax_top.set_xlim(4.5, 26.5)
ax_top.set_ylim(min(mag_meas.min(), mag_fit.min()) - 4, max(mag_meas.max(), mag_fit.max()) + 7)
handles = [
    Line2D([0], [0], color="#94A3B8", lw=1.5, label=r"Measured FRF $\widehat{H}(\omega)$"),
    Line2D([0], [0], color="#1D4ED8", lw=2.2, label=f"Fitted 3-mode rational model (NMSE = {nmse:.3f})"),
] + [Line2D([0], [0], color=colors_m[m], ls="--", lw=1.2, marker="o", ms=4.5,
            label=f"Mode {m + 1}: {fit.modal_frequencies_hz[m]:.2f} Hz") for m in range(3)]
ax_top.legend(handles=handles, loc="lower left", ncol=2, fontsize=8.2, framealpha=0.95, facecolor="white", edgecolor="#CBD5E1")
ax_top.grid(True, color="#E2E8F0", lw=0.6)

mode_info = [(f"Mode 1 ({modal_freqs[0]:.2f} Hz)", (-0.1, 1.18), "o"),
             (f"Mode 2 ({modal_freqs[1]:.2f} Hz)", (-1.18, 1.18), "s"),
             (f"Mode 3 ({modal_freqs[2]:.2f} Hz)", (-1.18, 1.18), "^")]
for m in range(3):
    ax = fig.add_subplot(gs[1, m])
    title_mode, y_lims, marker = mode_info[m]
    ax.plot(station_x, p1_mean[m], color="#D97706", lw=1.8, ls="--")
    ax.plot(station_x, p2_mean[m], color="#2563EB", lw=1.8, ls=":")
    ax.plot(station_x, fused_profiles[m], color="#0F172A", lw=2.2, marker=marker, ms=4.8, zorder=6)
    if y_lims[0] < -0.2:
        ax.axhline(0, color="#94A3B8", lw=0.7, alpha=0.6)
    ax.set_title(f"({chr(98 + m)}) {title_mode}", loc="left")
    ax.set_xlabel("Longitudinal coordinate $x$ (m)")
    if m == 0:
        ax.set_ylabel(r"Normalized modal deflection $\phi(x)$")
    ax.set_ylim(y_lims)
    ax.set_xlim(-17.5, 17.5)
    ax.grid(True, color="#E2E8F0", lw=0.6)
fig.legend(handles=[Line2D([0], [0], color="#D97706", lw=1.6, ls="--", label="P1 excitation"),
                    Line2D([0], [0], color="#2563EB", lw=1.6, ls=":", label="P2 excitation"),
                    Line2D([0], [0], color="#0F172A", lw=2.2, marker="o", ms=5.0, label=r"Fused consensus vector $\mathbf{u}_m^*$")],
           loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9.0, framealpha=0.95,
           facecolor="white", edgecolor="#CBD5E1")
fig.savefig(FIG_DIR / "experimental_records_modal_quality_R1.png", dpi=600, bbox_inches="tight")
print("saved Figure 8")
