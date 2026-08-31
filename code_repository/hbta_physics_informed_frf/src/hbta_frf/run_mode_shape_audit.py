"""Evaluate normalized modal residue shapes using the complex MAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def complex_mac(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Complex modal assurance criterion over the final sensor axis."""
    numerator = np.abs(np.sum(np.conj(left) * right, axis=-1)) ** 2
    denominator = (
        np.sum(np.abs(left) ** 2, axis=-1)
        * np.sum(np.abs(right) ** 2, axis=-1)
        + np.finfo(float).eps
    )
    return numerator / denominator


def scores_against_training(
    observations: np.ndarray, training: np.ndarray
) -> np.ndarray:
    """Median training-reference mean-mode dissimilarity for each observation."""
    pair_scores = []
    for reference in training:
        mac = complex_mac(observations, reference[None, :, :])
        pair_scores.append(np.mean(1.0 - mac, axis=1))
    return np.median(np.column_stack(pair_scores), axis=1)


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
    shapes = data["mode_shapes"].astype(np.complex128)
    recording_ids = data["recording_ids"]
    states = data["states"].astype(str)
    reference = shapes[recording_ids == 0]
    external_healthy = shapes[recording_ids == 1]
    calibration = reference[8:11]
    heldout = reference[11:14]
    damage_by_state = {
        state: shapes[states == state]
        for state in sorted(set(states) - {"UDS"})
    }

    results: dict[str, object] = {
        "feature_definition": (
            "mean across three modes of one minus complex MAC; modal residues "
            "remove the unknown input participation scalar"
        ),
        "training_sizes": {},
    }
    plot_values: dict[str, list[float]] = {}
    for size in (1, 2, 4, 8):
        training = reference[:size]
        calibration_scores = scores_against_training(calibration, training)
        heldout_scores = scores_against_training(heldout, training)
        external_scores = scores_against_training(external_healthy, training)
        threshold = float(np.quantile(calibration_scores, 0.95, method="higher"))
        state_scores = {
            state: scores_against_training(values, training)
            for state, values in damage_by_state.items()
        }
        state_auc = {
            state: auc_pairwise(heldout_scores, score)
            for state, score in state_scores.items()
        }
        state_tpr = {
            state: float(np.mean(score > threshold))
            for state, score in state_scores.items()
        }
        external_state_auc = {
            state: auc_pairwise(external_scores, score)
            for state, score in state_scores.items()
        }
        results["training_sizes"][str(size)] = {
            "threshold": threshold,
            "same_session_false_alarm_rate": float(
                np.mean(heldout_scores > threshold)
            ),
            "external_healthy_false_alarm_rate": float(
                np.mean(external_scores > threshold)
            ),
            "same_session_macro_state_auc": float(
                np.mean(list(state_auc.values()))
            ),
            "external_healthy_macro_state_auc": float(
                np.mean(list(external_state_auc.values()))
            ),
            "macro_state_tpr": float(np.mean(list(state_tpr.values()))),
            "per_state_auc": state_auc,
            "per_state_external_healthy_auc": external_state_auc,
            "per_state_tpr": state_tpr,
        }
        if size == 8:
            plot_values = {
                "calibration": calibration_scores.tolist(),
                "same-session healthy": heldout_scores.tolist(),
                "external healthy": external_scores.tolist(),
                **{state: score.tolist() for state, score in state_scores.items()},
            }

    labels = list(plot_values)
    values = [plot_values[label] for label in labels]
    figure, axis = plt.subplots(figsize=(10.8, 4.2))
    axis.boxplot(values, showfliers=False)
    axis.axhline(
        results["training_sizes"]["8"]["threshold"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="calibrated threshold",
    )
    axis.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=38)
    axis.set_ylabel("Mean modal-shape dissimilarity (1 - complex MAC)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
