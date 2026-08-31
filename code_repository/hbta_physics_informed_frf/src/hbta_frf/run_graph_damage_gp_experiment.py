"""Run the HBTA rank-one damage-conditioned graph-GP go/no-go experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_damage_gp import (
    aligned_residuals,
    damage_alignment_score,
    damage_log_likelihood_ratio,
    fit_healthy_graph_model,
    fit_mode_shape_template,
    healthy_nll,
    isotropic_nll,
    load_physical_graph,
    shuffled_graph,
)


def auc_pairwise(negative: np.ndarray, positive: np.ndarray) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def tune_graph_model(
    graph,
    development_training_shapes: np.ndarray,
    development_validation_shapes: np.ndarray,
) -> tuple[dict, object, np.ndarray]:
    """Select graph length scale and nugget using healthy UDS01 only."""
    template = fit_mode_shape_template(development_training_shapes)
    training_residuals = aligned_residuals(development_training_shapes, template)
    validation_residuals = aligned_residuals(
        development_validation_shapes, template
    )
    grid = []
    best = None
    for alpha_factor in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        for nugget_ratio in (0.01, 0.03, 0.1, 0.3, 1.0):
            model = fit_healthy_graph_model(
                graph,
                training_residuals,
                alpha_factor,
                nugget_ratio,
            )
            loss = float(np.mean(healthy_nll(validation_residuals, model)))
            item = {
                "alpha_factor": alpha_factor,
                "nugget_ratio": nugget_ratio,
                "healthy_validation_nll": loss,
            }
            grid.append(item)
            if best is None or loss < best[0]:
                best = (loss, model, item)
    assert best is not None
    return {"selected": best[2], "grid": grid}, best[1], template


def summarize_scores(
    calibration_scores: np.ndarray,
    same_session_healthy_scores: np.ndarray,
    external_healthy_scores: np.ndarray,
    damage_scores: dict[str, np.ndarray],
) -> dict:
    threshold = float(
        np.quantile(calibration_scores, 0.95, method="higher")
    )
    same_auc = {
        state: auc_pairwise(same_session_healthy_scores, scores)
        for state, scores in damage_scores.items()
    }
    external_auc = {
        state: auc_pairwise(external_healthy_scores, scores)
        for state, scores in damage_scores.items()
    }
    tpr = {
        state: float(np.mean(scores > threshold))
        for state, scores in damage_scores.items()
    }
    return {
        "calibration_threshold": threshold,
        "calibration_median": float(np.median(calibration_scores)),
        "same_session_healthy_median": float(
            np.median(same_session_healthy_scores)
        ),
        "external_healthy_median": float(np.median(external_healthy_scores)),
        "same_session_false_alarm_rate": float(
            np.mean(same_session_healthy_scores > threshold)
        ),
        "external_healthy_false_alarm_rate": float(
            np.mean(external_healthy_scores > threshold)
        ),
        "same_session_macro_auc": float(np.mean(list(same_auc.values()))),
        "external_healthy_macro_auc": float(
            np.mean(list(external_auc.values()))
        ),
        "macro_tpr_at_calibrated_threshold": float(np.mean(list(tpr.values()))),
        "per_state_same_session_auc": same_auc,
        "per_state_external_healthy_auc": external_auc,
        "per_state_tpr": tpr,
    }


def score_all(
    shapes: np.ndarray,
    states: np.ndarray,
    recording_ids: np.ndarray,
    template: np.ndarray,
    training_residuals: np.ndarray,
    graph,
    model,
) -> tuple[dict[str, dict], dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    residuals = aligned_residuals(shapes, template)
    isotropic = isotropic_nll(training_residuals, residuals)
    graph_nll = healthy_nll(residuals, model)
    damage_llr, likelihood_edges, best_reductions = damage_log_likelihood_ratio(
        residuals, graph, model
    )
    alignment, alignment_edges = damage_alignment_score(
        residuals, graph, model
    )
    scores = {
        "isotropic_gp": isotropic,
        "healthy_graph_gp": graph_nll,
        "damage_conditioned_graph_gp": damage_llr,
        "scale_invariant_damage_alignment": alignment,
    }
    masks = {
        "calibration": (recording_ids == 0) & (np.arange(len(shapes)) >= 8) & (np.arange(len(shapes)) <= 10),
        "same_session": (recording_ids == 0) & (np.arange(len(shapes)) >= 11) & (np.arange(len(shapes)) <= 13),
        "external": recording_ids == 1,
    }
    damage_masks = {
        state: states == state for state in sorted(set(states) - {"UDS"})
    }
    summaries = {
        name: summarize_scores(
            values[masks["calibration"]],
            values[masks["same_session"]],
            values[masks["external"]],
            {state: values[mask] for state, mask in damage_masks.items()},
        )
        for name, values in scores.items()
    }
    return summaries, scores, (
        likelihood_edges,
        best_reductions,
        alignment_edges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--shuffle-count", type=int, default=20)
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    frf_data = np.load(args.frf_dataset, allow_pickle=False)
    shapes = features["mode_shapes"].astype(np.complex128)
    states = features["states"].astype(str)
    recording_ids = features["recording_ids"]
    graph = load_physical_graph(
        args.edge_list, frf_data["response_sensor_names"]
    )

    reference_shapes = shapes[recording_ids == 0]
    tuning, _, _ = tune_graph_model(
        graph, reference_shapes[:4], reference_shapes[4:8]
    )
    template = fit_mode_shape_template(reference_shapes[:8])
    training_residuals = aligned_residuals(reference_shapes[:8], template)
    selected = tuning["selected"]
    model = fit_healthy_graph_model(
        graph,
        training_residuals,
        selected["alpha_factor"],
        selected["nugget_ratio"],
    )
    summaries, all_scores, inferred = score_all(
        shapes,
        states,
        recording_ids,
        template,
        training_residuals,
        graph,
        model,
    )

    likelihood_edges, best_reductions, alignment_edges = inferred
    inferred_by_state = {}
    for state in sorted(set(states) - {"UDS"}):
        indices = np.flatnonzero(states == state)
        edge_counter = Counter(alignment_edges[indices].tolist())
        edge_index, count = edge_counter.most_common(1)[0]
        inferred_by_state[state] = {
            "dominant_edge_index": int(edge_index),
            "dominant_edge": list(graph.edge_names[edge_index]),
            "relation": graph.relations[edge_index],
            "block_fraction": float(count / len(indices)),
            "median_selected_reduction": float(
                np.median(best_reductions[indices])
            ),
        }

    shuffled_controls = []
    for seed in range(args.shuffle_count):
        control_graph = shuffled_graph(graph, seed)
        control_tuning, _, _ = tune_graph_model(
            control_graph, reference_shapes[:4], reference_shapes[4:8]
        )
        control_selected = control_tuning["selected"]
        control_model = fit_healthy_graph_model(
            control_graph,
            training_residuals,
            control_selected["alpha_factor"],
            control_selected["nugget_ratio"],
        )
        control_summary, _, _ = score_all(
            shapes,
            states,
            recording_ids,
            template,
            training_residuals,
            control_graph,
            control_model,
        )
        metric = control_summary["scale_invariant_damage_alignment"]
        shuffled_controls.append(
            {
                "seed": seed,
                "selected": control_selected,
                "external_healthy_false_alarm_rate": metric[
                    "external_healthy_false_alarm_rate"
                ],
                "external_healthy_macro_auc": metric[
                    "external_healthy_macro_auc"
                ],
            }
        )
        print(f"completed shuffled control {seed + 1}/{args.shuffle_count}", flush=True)

    shuffled_auc = np.asarray(
        [item["external_healthy_macro_auc"] for item in shuffled_controls]
    )
    physical_auc = summaries["scale_invariant_damage_alignment"][
        "external_healthy_macro_auc"
    ]
    result = {
        "status": "preliminary_go_no_go",
        "hypothesis": (
            "A rank-one damage-conditioned graph GP distinguishes localized "
            "stiffness loss from recording-domain shift better than unstructured "
            "and healthy-only GP baselines."
        ),
        "data_split": {
            "development_training": "UDS01 blocks 0-3",
            "development_validation": "UDS01 blocks 4-7",
            "final_healthy_fit": "UDS01 blocks 0-7",
            "threshold_calibration": "UDS01 blocks 8-10",
            "same_session_healthy_test": "UDS01 blocks 11-13",
            "external_healthy_stress_test": "all 14 UDS02 blocks",
            "damage_test": "all 14 blocks for each DS1-DS8",
        },
        "graph": {
            "nodes": len(graph.sensor_names),
            "edges": len(graph.edge_names),
            "source_edge_list": str(args.edge_list.resolve()),
            "edge_weighting": "relation-balanced inverse structural-path length",
        },
        "healthy_only_hyperparameter_selection": tuning,
        "fitted_model": {
            "alpha": model.alpha,
            "alpha_factor": model.alpha_factor,
            "nugget_ratio": model.nugget_ratio,
            "mode_scales": model.mode_scales.tolist(),
        },
        "models": summaries,
        "damage_conditioned_inferred_edges": inferred_by_state,
        "shuffled_graph_controls": {
            "count": args.shuffle_count,
            "external_macro_auc_median": float(np.median(shuffled_auc)),
            "external_macro_auc_05_95": np.quantile(
                shuffled_auc, [0.05, 0.95]
            ).tolist(),
            "physical_auc_percentile_among_shuffles": float(
                np.mean(shuffled_auc <= physical_auc)
            ),
            "runs": shuffled_controls,
        },
    }
    primary = summaries["scale_invariant_damage_alignment"]
    result["go_no_go"] = {
        "external_false_alarm_gate_le_1_over_14": bool(
            primary["external_healthy_false_alarm_rate"] <= 1.0 / 14.0
        ),
        "external_macro_auc_gate_ge_0_85": bool(
            primary["external_healthy_macro_auc"] >= 0.85
        ),
        "beats_both_baselines_external_auc": bool(
            primary["external_healthy_macro_auc"]
            > max(
                summaries["isotropic_gp"]["external_healthy_macro_auc"],
                summaries["healthy_graph_gp"]["external_healthy_macro_auc"],
            )
        ),
        "physical_graph_above_95_percent_shuffles": bool(
            np.mean(shuffled_auc <= physical_auc) >= 0.95
        ),
    }
    result["go_no_go"]["overall_go"] = bool(
        all(result["go_no_go"].values())
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    calibration_mask = (recording_ids == 0) & (np.arange(len(shapes)) >= 8) & (np.arange(len(shapes)) <= 10)
    same_mask = (recording_ids == 0) & (np.arange(len(shapes)) >= 11) & (np.arange(len(shapes)) <= 13)
    external_mask = recording_ids == 1
    damage_masks = {
        state: states == state for state in sorted(set(states) - {"UDS"})
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 8.2))
    labels = ["cal", "same UDS", "external UDS", *damage_masks]
    for axis, (model_name, scores) in zip(axes.ravel(), all_scores.items()):
        values = [
            scores[calibration_mask],
            scores[same_mask],
            scores[external_mask],
            *[scores[mask] for mask in damage_masks.values()],
        ]
        axis.boxplot(values, showfliers=False)
        axis.axhline(
            summaries[model_name]["calibration_threshold"],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        axis.set_xticks(
            np.arange(1, len(labels) + 1), labels=labels, rotation=55
        )
        axis.set_title(model_name.replace("_", " "))
        axis.set_ylabel("Anomaly evidence (higher is more damaged)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    print(json.dumps(result["go_no_go"], indent=2))
    print(f"wrote {args.output_json} and {args.output_figure}")


if __name__ == "__main__":
    main()
