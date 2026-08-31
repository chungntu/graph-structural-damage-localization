"""Test whether modal-pole features reject FRF residue/acquisition nuisance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def diagonal_distance(
    values: np.ndarray, mean: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    return np.sum((values - mean[None, :]) ** 2 / variance[None, :], axis=1)


def auc_pairwise(negative: np.ndarray, positive: np.ndarray) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    modal = data["modal_frequencies_hz"]
    recording_ids = data["recording_ids"]
    states = data["states"].astype(str)
    reference = modal[recording_ids == 0]
    external_healthy = modal[recording_ids == 1]
    calibration = reference[8:11]
    heldout = reference[11:14]
    damage_by_state = {
        state: modal[states == state]
        for state in sorted(set(states) - {"UDS"})
    }
    resolution = float(data["frequency_resolution_hz"])

    results: dict[str, object] = {
        "feature_definition": (
            "three shared modal frequencies; complex residues are fitted freely "
            "for every sensor and every block"
        ),
        "selected_sensor_names": data["selected_sensor_names"].astype(str).tolist(),
        "frequency_resolution_hz": resolution,
        "optimizer_success_fraction": float(np.mean(data["optimizer_success"])),
        "training_sizes": {},
    }
    for size in (1, 2, 4, 8):
        training = reference[:size]
        mean = np.mean(training, axis=0)
        if size > 1:
            variance = np.var(training, axis=0, ddof=1)
        else:
            variance = np.zeros(modal.shape[1])
        variance = np.maximum(variance, resolution**2)
        calibration_scores = diagonal_distance(calibration, mean, variance)
        heldout_scores = diagonal_distance(heldout, mean, variance)
        external_scores = diagonal_distance(external_healthy, mean, variance)
        threshold = float(np.quantile(calibration_scores, 0.95, method="higher"))
        state_scores = {
            state: diagonal_distance(values, mean, variance)
            for state, values in damage_by_state.items()
        }
        state_auc = {
            state: auc_pairwise(heldout_scores, scores)
            for state, scores in state_scores.items()
        }
        state_tpr = {
            state: float(np.mean(scores > threshold))
            for state, scores in state_scores.items()
        }
        results["training_sizes"][str(size)] = {
            "healthy_mean_frequencies_hz": mean.tolist(),
            "healthy_standard_deviation_floor_hz": np.sqrt(variance).tolist(),
            "threshold": threshold,
            "same_session_false_alarm_rate": float(
                np.mean(heldout_scores > threshold)
            ),
            "external_healthy_false_alarm_rate": float(
                np.mean(external_scores > threshold)
            ),
            "macro_state_auc": float(np.mean(list(state_auc.values()))),
            "macro_state_tpr": float(np.mean(list(state_tpr.values()))),
            "per_state_auc": state_auc,
            "per_state_tpr": state_tpr,
        }

    recording_names = data["recording_names"].astype(str)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for mode_index, axis in enumerate(axes):
        groups = [modal[recording_ids == index, mode_index] for index in range(10)]
        axis.boxplot(groups, showfliers=False)
        axis.set_xticks(
            np.arange(1, 11),
            labels=[name.split("_")[2] for name in recording_names],
            rotation=45,
        )
        axis.set_ylabel(f"Mode {mode_index + 1} frequency (Hz)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
