"""Conformal HBTA evaluation of scale-invariant rank-one graph-GP alignment."""

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
    fit_healthy_graph_model,
    fit_mode_shape_template,
    healthy_nll,
    isotropic_nll,
    load_physical_graph,
    shuffled_graph,
)
from .run_graph_damage_gp_experiment import auc_pairwise


ALPHA_FACTORS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
NUGGET_RATIOS = (0.01, 0.03, 0.1, 0.3, 1.0)


def twofold_tune(graph, reference_shapes: np.ndarray) -> dict:
    """Healthy-only twofold tuning over an even number of reference estimates."""
    reference_count = len(reference_shapes)
    if reference_count < 4 or reference_count % 2:
        raise ValueError(
            "twofold_tune requires an even number of at least four references"
        )
    midpoint = reference_count // 2
    first = np.arange(midpoint)
    second = np.arange(midpoint, reference_count)
    folds = ((first, second), (second, first))
    grid = []
    for alpha_factor in ALPHA_FACTORS:
        for nugget_ratio in NUGGET_RATIOS:
            losses = []
            for training_indices, validation_indices in folds:
                template = fit_mode_shape_template(reference_shapes[training_indices])
                training_residuals = aligned_residuals(
                    reference_shapes[training_indices], template
                )
                validation_residuals = aligned_residuals(
                    reference_shapes[validation_indices], template
                )
                model = fit_healthy_graph_model(
                    graph,
                    training_residuals,
                    alpha_factor,
                    nugget_ratio,
                )
                losses.append(float(np.mean(healthy_nll(validation_residuals, model))))
            grid.append(
                {
                    "alpha_factor": alpha_factor,
                    "nugget_ratio": nugget_ratio,
                    "mean_twofold_healthy_nll": float(np.mean(losses)),
                    "fold_losses": losses,
                }
            )
    selected = min(grid, key=lambda item: item["mean_twofold_healthy_nll"])
    return {"selected": selected, "grid": grid}


def fit_full(graph, reference_shapes: np.ndarray, selected: dict):
    template = fit_mode_shape_template(reference_shapes)
    training_residuals = aligned_residuals(reference_shapes, template)
    model = fit_healthy_graph_model(
        graph,
        training_residuals,
        selected["alpha_factor"],
        selected["nugget_ratio"],
    )
    return template, training_residuals, model


def score_models(shapes, template, training_residuals, graph, model):
    residuals = aligned_residuals(shapes, template)
    alignment, edges = damage_alignment_score(residuals, graph, model)
    return {
        "isotropic_gp": isotropic_nll(training_residuals, residuals),
        "healthy_graph_gp": healthy_nll(residuals, model),
        "scale_invariant_damage_alignment": alignment,
    }, edges


def leave_one_out_calibration(graph, reference_shapes, selected) -> dict[str, np.ndarray]:
    scores = {
        "isotropic_gp": [],
        "healthy_graph_gp": [],
        "scale_invariant_damage_alignment": [],
    }
    for heldout_index in range(len(reference_shapes)):
        training = np.delete(reference_shapes, heldout_index, axis=0)
        template, training_residuals, model = fit_full(graph, training, selected)
        current, _ = score_models(
            reference_shapes[heldout_index : heldout_index + 1],
            template,
            training_residuals,
            graph,
            model,
        )
        for name in scores:
            scores[name].append(float(current[name][0]))
    return {name: np.asarray(values) for name, values in scores.items()}


def summarize(calibration, external, damage_by_state) -> dict:
    threshold = float(np.quantile(calibration, 0.95, method="higher"))
    auc = {
        state: auc_pairwise(external, scores)
        for state, scores in damage_by_state.items()
    }
    tpr = {
        state: float(np.mean(scores > threshold))
        for state, scores in damage_by_state.items()
    }
    return {
        "conformal_threshold": threshold,
        "calibration_scores": calibration.tolist(),
        "external_healthy_false_alarm_rate": float(np.mean(external > threshold)),
        "external_healthy_macro_auc": float(np.mean(list(auc.values()))),
        "macro_tpr_at_conformal_threshold": float(np.mean(list(tpr.values()))),
        "per_state_external_healthy_auc": auc,
        "per_state_tpr": tpr,
    }


def rotation_audit(graph, reference, external, damage_shapes) -> list[dict]:
    runs = []
    for rotation in range(14):
        order = np.roll(np.arange(14), -rotation)
        training_indices = order[:8]
        calibration_indices = order[8:11]
        # Eight-block tuning uses its first and second halves, matching the
        # original scarcity protocol without seeing external or damage data.
        subreference = reference[training_indices]
        grid = []
        for alpha_factor in ALPHA_FACTORS:
            for nugget_ratio in NUGGET_RATIOS:
                template = fit_mode_shape_template(subreference[:4])
                train_residuals = aligned_residuals(subreference[:4], template)
                validation_residuals = aligned_residuals(subreference[4:8], template)
                candidate = fit_healthy_graph_model(
                    graph, train_residuals, alpha_factor, nugget_ratio
                )
                grid.append(
                    (
                        float(np.mean(healthy_nll(validation_residuals, candidate))),
                        alpha_factor,
                        nugget_ratio,
                    )
                )
        _, alpha_factor, nugget_ratio = min(grid)
        selected = {
            "alpha_factor": alpha_factor,
            "nugget_ratio": nugget_ratio,
        }
        template, train_residuals, model = fit_full(
            graph, subreference, selected
        )
        calibration_scores = score_models(
            reference[calibration_indices],
            template,
            train_residuals,
            graph,
            model,
        )[0]["scale_invariant_damage_alignment"]
        external_scores = score_models(
            external, template, train_residuals, graph, model
        )[0]["scale_invariant_damage_alignment"]
        damage_scores = {
            state: score_models(values, template, train_residuals, graph, model)[0][
                "scale_invariant_damage_alignment"
            ]
            for state, values in damage_shapes.items()
        }
        threshold = float(np.max(calibration_scores))
        runs.append(
            {
                "rotation": rotation,
                "external_false_alarm_rate": float(
                    np.mean(external_scores > threshold)
                ),
                "external_macro_auc": float(
                    np.mean(
                        [
                            auc_pairwise(external_scores, scores)
                            for scores in damage_scores.values()
                        ]
                    )
                ),
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--shuffle-count", type=int, default=100)
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    shapes = features["mode_shapes"].astype(np.complex128)
    recording_ids = features["recording_ids"]
    states = features["states"].astype(str)
    reference = shapes[recording_ids == 0]
    external = shapes[recording_ids == 1]
    damage_shapes = {
        state: shapes[states == state]
        for state in sorted(set(states) - {"UDS"})
    }
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])

    tuning = twofold_tune(graph, reference)
    selected = tuning["selected"]
    calibration = leave_one_out_calibration(graph, reference, selected)
    template, training_residuals, model = fit_full(graph, reference, selected)
    external_scores, external_edges = score_models(
        external, template, training_residuals, graph, model
    )
    damage_scores = {}
    damage_edges = {}
    for state, values in damage_shapes.items():
        current_scores, current_edges = score_models(
            values, template, training_residuals, graph, model
        )
        damage_scores[state] = current_scores
        damage_edges[state] = current_edges

    summaries = {
        name: summarize(
            calibration[name],
            external_scores[name],
            {state: values[name] for state, values in damage_scores.items()},
        )
        for name in calibration
    }

    inferred = {}
    for state, edges in damage_edges.items():
        edge_index, count = Counter(edges.tolist()).most_common(1)[0]
        inferred[state] = {
            "dominant_edge_index": int(edge_index),
            "dominant_edge": list(graph.edge_names[edge_index]),
            "relation": graph.relations[edge_index],
            "block_fraction": float(count / len(edges)),
        }

    shuffled_auc = []
    shuffled_runs = []
    for seed in range(args.shuffle_count):
        control = shuffled_graph(graph, seed)
        control_selected = twofold_tune(control, reference)["selected"]
        control_template, control_training, control_model = fit_full(
            control, reference, control_selected
        )
        control_external = score_models(
            external,
            control_template,
            control_training,
            control,
            control_model,
        )[0]["scale_invariant_damage_alignment"]
        values = []
        for damage in damage_shapes.values():
            score = score_models(
                damage,
                control_template,
                control_training,
                control,
                control_model,
            )[0]["scale_invariant_damage_alignment"]
            values.append(auc_pairwise(control_external, score))
        current_auc = float(np.mean(values))
        shuffled_auc.append(current_auc)
        shuffled_runs.append(
            {
                "seed": seed,
                "external_macro_auc": current_auc,
                "selected_alpha_factor": control_selected["alpha_factor"],
                "selected_nugget_ratio": control_selected["nugget_ratio"],
            }
        )
        if (seed + 1) % 10 == 0:
            print(f"completed {seed + 1}/{args.shuffle_count} shuffled controls", flush=True)

    rotations = rotation_audit(graph, reference, external, damage_shapes)
    rotation_auc = np.asarray([item["external_macro_auc"] for item in rotations])
    rotation_fa = np.asarray(
        [item["external_false_alarm_rate"] for item in rotations]
    )
    shuffled_auc_array = np.asarray(shuffled_auc)
    primary = summaries["scale_invariant_damage_alignment"]
    physical_auc = primary["external_healthy_macro_auc"]
    physical_percentile = float(np.mean(shuffled_auc_array <= physical_auc))
    permutation_p = float(
        (1 + np.sum(shuffled_auc_array >= physical_auc))
        / (1 + len(shuffled_auc_array))
    )
    go_no_go = {
        "external_false_alarm_gate_le_1_over_14": bool(
            primary["external_healthy_false_alarm_rate"] <= 1.0 / 14.0
        ),
        "external_macro_auc_gate_ge_0_85": bool(physical_auc >= 0.85),
        "beats_both_baselines_external_auc": bool(
            physical_auc
            > max(
                summaries["isotropic_gp"]["external_healthy_macro_auc"],
                summaries["healthy_graph_gp"]["external_healthy_macro_auc"],
            )
        ),
        "physical_graph_permutation_p_le_0_05": bool(permutation_p <= 0.05),
        "rotation_minimum_external_auc_ge_0_80": bool(
            np.min(rotation_auc) >= 0.80
        ),
    }
    go_no_go["overall_preliminary_go"] = bool(all(go_no_go.values()))

    result = {
        "status": "preliminary_conformal_go_no_go",
        "protocol": {
            "healthy_reference": "all 14 UDS01 blocks",
            "calibration": "leave-one-block-out UDS01 nonconformity scores",
            "external_healthy": "all 14 UDS02 blocks, never used in fitting",
            "damage": "all DS1-DS8 blocks, never used in fitting or calibration",
            "threshold": "95th percentile with higher order statistic (maximum for n=14)",
        },
        "selected_hyperparameters": selected,
        "models": summaries,
        "inferred_edges": inferred,
        "shuffled_graph_control": {
            "count": args.shuffle_count,
            "median_external_macro_auc": float(np.median(shuffled_auc_array)),
            "quantiles_05_95": np.quantile(
                shuffled_auc_array, [0.05, 0.95]
            ).tolist(),
            "physical_percentile": physical_percentile,
            "empirical_one_sided_p_value": permutation_p,
            "runs": shuffled_runs,
        },
        "temporal_rotation_audit": {
            "count": len(rotations),
            "external_macro_auc_median_min_max": [
                float(np.median(rotation_auc)),
                float(np.min(rotation_auc)),
                float(np.max(rotation_auc)),
            ],
            "external_false_alarm_median_min_max": [
                float(np.median(rotation_fa)),
                float(np.min(rotation_fa)),
                float(np.max(rotation_fa)),
            ],
            "warning": (
                "Three-block threshold calibration is unstable; the primary "
                "operational result uses leave-one-out conformal calibration."
            ),
            "runs": rotations,
        },
        "go_no_go": go_no_go,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    labels = ["external UDS", *damage_shapes]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    for axis, name in zip(axes, calibration):
        values = [
            external_scores[name],
            *[damage_scores[state][name] for state in damage_shapes],
        ]
        axis.boxplot(values, showfliers=False)
        axis.axhline(
            summaries[name]["conformal_threshold"],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        axis.set_xticks(
            np.arange(1, len(labels) + 1), labels=labels, rotation=55
        )
        axis.set_title(name.replace("_", " "))
        axis.set_ylabel("Anomaly evidence")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)
    print(json.dumps(go_no_go, indent=2))
    print(f"wrote {args.output_json} and {args.output_figure}")


if __name__ == "__main__":
    main()
