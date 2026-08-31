"""Audit whether a three-mode physical mean explains held-out healthy FRFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ
from .modal_model import fit_shared_pole_model


def fixed_weighted_nmse(
    observations: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return one fixed-reference weighted NMSE value per FRF block."""
    numerator = np.sum(
        weights[None, :, :] * np.abs(observations - prediction[None, :, :]) ** 2,
        axis=(1, 2),
    )
    denominator = (
        np.sum(weights * np.abs(prediction) ** 2) + np.finfo(float).eps
    )
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    frequencies = data["frequencies_hz"].astype(float)
    frf = data["frf"].astype(np.complex128)
    coherence = data["coherence"].astype(float)
    recording_ids = data["recording_ids"]
    band = (frequencies >= 5.0) & (frequencies <= 26.0)
    frequencies = frequencies[band]
    frf = frf[:, :, band]
    coherence = coherence[:, :, band]

    training = recording_ids == 0
    heldout = recording_ids == 1
    training_target = np.median(frf[training], axis=0)
    fixed_weights = np.median(coherence[training], axis=0)
    fixed_weights = np.where(fixed_weights >= 0.5, fixed_weights, 0.0)

    fit = fit_shared_pole_model(
        frequencies,
        training_target,
        fixed_weights,
        INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
    )
    modal_training = fixed_weighted_nmse(
        frf[training], fit.prediction, fixed_weights
    )
    modal_heldout = fixed_weighted_nmse(frf[heldout], fit.prediction, fixed_weights)
    empirical_training = fixed_weighted_nmse(
        frf[training], training_target, fixed_weights
    )
    empirical_heldout = fixed_weighted_nmse(
        frf[heldout], training_target, fixed_weights
    )

    result = {
        "model_band_hz": [5.0, 26.0],
        "initial_modal_frequencies_hz": list(
            map(float, INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ)
        ),
        "fitted_modal_frequencies_hz": fit.modal_frequencies_hz.tolist(),
        "fitted_damping_ratios": fit.damping_ratios.tolist(),
        "median_training_modal_nmse": float(np.median(modal_training)),
        "median_heldout_modal_nmse": float(np.median(modal_heldout)),
        "median_training_empirical_mean_nmse": float(
            np.median(empirical_training)
        ),
        "median_heldout_empirical_mean_nmse": float(np.median(empirical_heldout)),
        "modal_to_empirical_heldout_nmse_ratio": float(
            np.median(modal_heldout) / np.median(empirical_heldout)
        ),
        "usable_weight_fraction": float(np.mean(fixed_weights > 0.0)),
        "optimizer_success": fit.optimizer_success,
        "optimizer_message": fit.optimizer_message,
        "interpretation": (
            "The physical mean is useful only if its held-out error is finite and "
            "it captures a non-trivial part of the healthy FRF. It need not beat "
            "the empirical mean; the grey-box GP is intended to model its discrepancy."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].boxplot(
        [modal_training, modal_heldout, empirical_training, empirical_heldout],
        tick_labels=["modal\ntrain", "modal\nheld-out", "mean\ntrain", "mean\nheld-out"],
        showfliers=False,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Fixed-reference weighted NMSE")
    axes[0].grid(alpha=0.25)

    representative_sensor = int(np.argmax(np.sum(fixed_weights, axis=1)))
    axes[1].plot(
        frequencies,
        20.0 * np.log10(np.maximum(np.abs(training_target[representative_sensor]), 1e-12)),
        label="healthy training median",
        linewidth=1.5,
    )
    axes[1].plot(
        frequencies,
        20.0 * np.log10(np.maximum(np.abs(fit.prediction[representative_sensor]), 1e-12)),
        label="three-mode mean",
        linewidth=1.2,
    )
    for mode in fit.modal_frequencies_hz:
        axes[1].axvline(mode, color="black", alpha=0.18, linewidth=0.8)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("FRF magnitude (dB)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
