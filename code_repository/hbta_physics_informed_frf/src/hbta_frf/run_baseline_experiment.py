"""Run a minimal white/black/grey-box HBTA scarcity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ
from .kernel_model import smooth_complex_residual
from .modal_model import fit_shared_pole_model


def block_scores(
    observations: np.ndarray,
    prediction: np.ndarray,
    fixed_weights: np.ndarray,
) -> np.ndarray:
    """Energy-normalized complex residual, aggregated robustly across sensors."""
    prediction_energy = np.sum(
        fixed_weights * np.abs(prediction) ** 2, axis=1
    )
    positive = prediction_energy[prediction_energy > 0]
    energy_floor = np.quantile(positive, 0.10) if len(positive) else 1.0
    denominator = np.maximum(prediction_energy, energy_floor)
    error = np.sum(
        fixed_weights[None, :, :]
        * np.abs(observations - prediction[None, :, :]) ** 2,
        axis=2,
    )
    return np.median(error / denominator[None, :], axis=1)


def auc_pairwise(negative: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a random positive score exceeds a negative score."""
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def tune_smoother(
    frequencies: np.ndarray,
    development_train: np.ndarray,
    development_validation: np.ndarray,
    fixed_weights: np.ndarray,
    prior_mean: np.ndarray,
) -> tuple[float, float, float]:
    """Choose two GP smoothing parameters using healthy data only."""
    target = np.mean(development_train, axis=0)
    candidates_length = (0.15, 0.3, 0.6, 1.2, 2.4)
    candidates_nugget = (0.001, 0.01, 0.1, 1.0)
    best = (np.inf, candidates_length[0], candidates_nugget[0])
    for length_scale in candidates_length:
        for nugget in candidates_nugget:
            prediction = smooth_complex_residual(
                frequencies,
                target,
                prior_mean,
                length_scale,
                nugget,
            )
            loss = float(
                np.median(
                    block_scores(
                        development_validation, prediction, fixed_weights
                    )
                )
            )
            if loss < best[0]:
                best = (loss, length_scale, nugget)
    return best[1], best[2], best[0]


def summarize_model(
    prediction: np.ndarray,
    calibration: np.ndarray,
    heldout_healthy: np.ndarray,
    external_healthy: np.ndarray,
    damage_by_state: dict[str, np.ndarray],
    fixed_weights: np.ndarray,
) -> dict:
    calibration_scores = block_scores(calibration, prediction, fixed_weights)
    healthy_scores = block_scores(heldout_healthy, prediction, fixed_weights)
    external_healthy_scores = block_scores(
        external_healthy, prediction, fixed_weights
    )
    threshold = float(np.quantile(calibration_scores, 0.95, method="higher"))
    damage_scores = {
        state: block_scores(values, prediction, fixed_weights)
        for state, values in damage_by_state.items()
    }
    pooled_damage = np.concatenate(list(damage_scores.values()))
    per_state_auc = {
        state: auc_pairwise(healthy_scores, scores)
        for state, scores in damage_scores.items()
    }
    per_state_tpr = {
        state: float(np.mean(scores > threshold))
        for state, scores in damage_scores.items()
    }
    return {
        "calibration_threshold": threshold,
        "calibration_score_median": float(np.median(calibration_scores)),
        "heldout_healthy_score_median": float(np.median(healthy_scores)),
        "heldout_healthy_false_alarm_rate": float(np.mean(healthy_scores > threshold)),
        "external_healthy_score_median": float(
            np.median(external_healthy_scores)
        ),
        "external_healthy_false_alarm_rate": float(
            np.mean(external_healthy_scores > threshold)
        ),
        "pooled_damage_auc": auc_pairwise(healthy_scores, pooled_damage),
        "macro_state_auc": float(np.mean(list(per_state_auc.values()))),
        "macro_state_tpr": float(np.mean(list(per_state_tpr.values()))),
        "per_state_auc": per_state_auc,
        "per_state_tpr": per_state_tpr,
    }


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
    states = data["states"].astype(str)
    band = (frequencies >= 5.0) & (frequencies <= 26.0)
    frequencies = frequencies[band]
    frf = frf[:, :, band]
    coherence = coherence[:, :, band]

    reference = frf[recording_ids == 0]
    reference_coherence = coherence[recording_ids == 0]
    external_healthy = frf[recording_ids == 1]
    calibration = reference[8:11]
    heldout_healthy = reference[11:14]
    damage_by_state = {
        state: frf[states == state]
        for state in sorted(set(states) - {"UDS"})
    }

    development_train = reference[:4]
    development_validation = reference[4:8]
    development_weights = np.median(reference_coherence[:4], axis=0)
    development_weights = np.where(
        development_weights >= 0.5, development_weights, 0.0
    )
    development_target = np.mean(development_train, axis=0)
    development_modal = fit_shared_pole_model(
        frequencies,
        development_target,
        development_weights,
        INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
    )
    black_parameters = tune_smoother(
        frequencies,
        development_train,
        development_validation,
        development_weights,
        np.zeros_like(development_target),
    )
    grey_parameters = tune_smoother(
        frequencies,
        development_train,
        development_validation,
        development_weights,
        development_modal.prediction,
    )

    results: dict[str, object] = {
        "status": "preliminary_screening_not_final_inference",
        "question": (
            "Does a modal prior improve healthy complex-FRF modelling and "
            "one-class damage screening under limited healthy blocks?"
        ),
        "data_split": {
            "reference_recording": "MVS_P2_UDS_NM_Z_01",
            "training_blocks": "first k of blocks 0-7",
            "healthy_development_blocks": "blocks 0-3 train, 4-7 validation",
            "calibration_blocks": "reference blocks 8-10",
            "heldout_healthy_blocks": "reference blocks 11-13",
            "external_nuisance_stress_test": (
                "MVS_P2_UDS_NM_Z_02: different day, 10 C, train passing"
            ),
            "locked_damage_recordings": "MVS_P2_DS1-DS8_NM_Z_01",
        },
        "gp_hyperparameters_selected_on_healthy_only": {
            "black_box": {
                "length_scale_hz": black_parameters[0],
                "nugget_ratio": black_parameters[1],
                "development_loss": black_parameters[2],
            },
            "grey_box": {
                "length_scale_hz": grey_parameters[0],
                "nugget_ratio": grey_parameters[1],
                "development_loss": grey_parameters[2],
            },
        },
        "training_sizes": {},
    }

    for training_size in (1, 2, 4, 8):
        training = reference[:training_size]
        target = np.mean(training, axis=0)
        fixed_weights = np.median(
            reference_coherence[:training_size], axis=0
        )
        fixed_weights = np.where(fixed_weights >= 0.5, fixed_weights, 0.0)
        modal = fit_shared_pole_model(
            frequencies,
            target,
            fixed_weights,
            INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
        )
        predictions = {
            "white_modal": modal.prediction,
            "black_gp": smooth_complex_residual(
                frequencies,
                target,
                np.zeros_like(target),
                black_parameters[0],
                black_parameters[1],
            ),
            "grey_modal_gp": smooth_complex_residual(
                frequencies,
                target,
                modal.prediction,
                grey_parameters[0],
                grey_parameters[1],
            ),
        }
        size_result = {
            name: summarize_model(
                prediction,
                calibration,
                heldout_healthy,
                external_healthy,
                damage_by_state,
                fixed_weights,
            )
            for name, prediction in predictions.items()
        }
        size_result["modal_parameters"] = {
            "frequencies_hz": modal.modal_frequencies_hz.tolist(),
            "damping_ratios": modal.damping_ratios.tolist(),
            "optimizer_success": modal.optimizer_success,
        }
        results["training_sizes"][str(training_size)] = size_result
        print(f"completed training size {training_size}", flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharex=True)
    model_labels = {
        "white_modal": "modal only",
        "black_gp": "GP only",
        "grey_modal_gp": "modal + GP",
    }
    colors = {
        "white_modal": "#777777",
        "black_gp": "#3178c6",
        "grey_modal_gp": "#d35400",
    }
    sizes = np.asarray([1, 2, 4, 8])
    metrics = (
        ("macro_state_auc", "Macro DS1-DS8 AUROC"),
        ("heldout_healthy_false_alarm_rate", "Same-session healthy false-alarm rate"),
        ("macro_state_tpr", "Macro TPR at calibrated threshold"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        for model, label in model_labels.items():
            values = [
                results["training_sizes"][str(size)][model][metric]
                for size in sizes
            ]
            axis.plot(
                sizes,
                values,
                marker="o",
                label=label,
                color=colors[model],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, labels=[str(value) for value in sizes])
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Healthy training blocks")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    print(f"wrote {args.output_json} and {args.output_figure}")


if __name__ == "__main__":
    main()
