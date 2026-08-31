"""Input-location-invariant vertical spatial localization for HBTA DS1--DS8."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_damage_gp import (
    aligned_residuals,
    damage_alignment_matrix,
    endpoint_contrast_alignment_matrix,
    endpoint_node_energy_matrix,
    normalized_endpoint_node_energy_matrix,
    normalized_node_energy_matrix,
    fit_healthy_graph_model,
    fit_mode_shape_template,
    load_physical_graph,
)
from .run_graph_damage_gp_conformal import twofold_tune
from .run_spatial_region_localization import (
    PRIMARY_TOLERANCE_M,
    SCAN_CENTERS_M,
    gaussian_region_matrix,
    load_edge_midpoints,
    locked_ground_truth,
    matched_errors,
    select_peaks,
)


def fused_shape(shapes: np.ndarray) -> np.ndarray:
    return fit_mode_shape_template(shapes)


def make_edge_scores(features: dict, graph, selected: dict) -> dict:
    states = features["states"].astype(str)
    recordings = features["recordings"].astype(str)
    directions = features["sweep_directions"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    healthy_joint = np.asarray(
        [
            fused_shape(
                shapes[
                    (states == "UDS")
                    & (directions == direction)
                    & np.char.endswith(recordings, suffix)
                ]
            )
            for suffix in ("01", "02")
            for direction in ("up", "down")
        ]
    )
    template = fit_mode_shape_template(healthy_joint)
    healthy_residual = aligned_residuals(healthy_joint, template)
    model = fit_healthy_graph_model(
        graph,
        healthy_residual,
        selected["alpha_factor"],
        selected["nugget_ratio"],
    )
    output = {
        "endpoint_node_energy": {},
        "normalized_endpoint_node_energy": {},
        "direct_node_coordinate": {},
        "endpoint_contrast": {},
        "graph_propagated": {},
    }
    for state in (f"DS{i}" for i in range(1, 9)):
        joint = np.asarray(
            [
                fused_shape(shapes[(states == state) & (directions == direction)])
                for direction in ("up", "down")
            ]
        )
        residual = aligned_residuals(joint, template)
        output["endpoint_node_energy"][state] = np.median(
            endpoint_node_energy_matrix(residual, graph), axis=0
        )
        output["normalized_endpoint_node_energy"][state] = np.median(
            normalized_endpoint_node_energy_matrix(residual, graph), axis=0
        )
        output["direct_node_coordinate"][state] = np.median(
            normalized_node_energy_matrix(residual), axis=0
        )
        output["endpoint_contrast"][state] = np.median(
            endpoint_contrast_alignment_matrix(residual, graph), axis=0
        )
        output["graph_propagated"][state] = np.median(
            damage_alignment_matrix(residual, graph, model), axis=0
        )
    return output


def load_sensor_longitudinal_coordinates(
    path: Path,
    sensor_names: tuple[str, ...],
) -> np.ndarray:
    """Read one frozen longitudinal coordinate for every measured sensor."""
    coordinates: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            coordinates.setdefault(row["sensor_i"], float(row["x_i_m"]))
            coordinates.setdefault(row["sensor_j"], float(row["x_j_m"]))
    missing = [name for name in sensor_names if name not in coordinates]
    if missing:
        raise ValueError(f"Missing longitudinal coordinates for {missing}")
    return np.asarray([coordinates[name] for name in sensor_names], dtype=float)


def summarize(
    edge_scores: dict[str, np.ndarray],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    bandwidth_m: float = 3.5,
) -> dict:
    region = gaussian_region_matrix(edge_x, SCAN_CENTERS_M, bandwidth_m)
    errors = []
    correct = []
    states = {}
    for state, target_values in truth.items():
        target = np.asarray(target_values)
        curve = region @ edge_scores[state]
        predicted = select_peaks(curve, SCAN_CENTERS_M, len(target))
        current = matched_errors(predicted, target)
        errors.extend(current.tolist())
        success = bool(np.all(current <= PRIMARY_TOLERANCE_M))
        correct.append(success)
        states[state] = {
            "ground_truth_x_m": target.tolist(),
            "predicted_x_m": predicted.tolist(),
            "matched_absolute_errors_m": current.tolist(),
            "all_targets_within_one_panel": success,
        }
    return {
        "bandwidth_m": bandwidth_m,
        "mean_absolute_error_m": float(np.mean(errors)),
        "median_absolute_error_m": float(np.median(errors)),
        "target_fraction_within_one_panel": float(
            np.mean(np.asarray(errors) <= PRIMARY_TOLERANCE_M)
        ),
        "state_fraction_within_one_panel": float(np.mean(correct)),
        "states": states,
    }


def permutation(
    edge_scores: dict[str, np.ndarray],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    observed: dict,
    count: int,
) -> dict:
    generator = np.random.default_rng(20260829)
    null_error = np.empty(count)
    null_accuracy = np.empty(count)
    for index in range(count):
        current = summarize(
            edge_scores, generator.permutation(edge_x), truth
        )
        null_error[index] = current["mean_absolute_error_m"]
        null_accuracy[index] = current["state_fraction_within_one_panel"]
    return {
        "count": count,
        "mean_error_one_sided_p": float(
            (1 + np.sum(null_error <= observed["mean_absolute_error_m"]))
            / (1 + count)
        ),
        "state_accuracy_one_sided_p": float(
            (
                1
                + np.sum(
                    null_accuracy >= observed["state_fraction_within_one_panel"]
                )
            )
            / (1 + count)
        ),
        "mean_error_null_05_50_95": np.quantile(
            null_error, [0.05, 0.5, 0.95]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-features", type=Path, required=True)
    parser.add_argument("--noise-features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    cached = np.load(args.sweep_features, allow_pickle=False)
    features = {key: cached[key] for key in cached.files}
    noise = np.load(args.noise_features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    selected = twofold_tune(
        graph, noise["mode_shapes"][noise["recording_ids"] == 0]
    )["selected"]
    positive_laplacian_eigenvalues = np.linalg.eigvalsh(graph.laplacian)
    positive_laplacian_eigenvalues = positive_laplacian_eigenvalues[
        positive_laplacian_eigenvalues > 1e-10
    ]
    selected_alpha = float(
        selected["alpha_factor"] * np.median(positive_laplacian_eigenvalues)
    )
    edge_scores = make_edge_scores(features, graph, selected)
    edge_x = load_edge_midpoints(args.edge_list, graph)
    node_x = load_sensor_longitudinal_coordinates(
        args.edge_list, graph.sensor_names
    )
    truth = locked_ground_truth()
    result = {
        "status": "exploratory_input_location_invariant_spatial_localization",
        "definition": (
            "P1 and P2 modal-response vectors are fused before localization "
            "because a structural mode shape is independent of input location"
        ),
        "modal_vector_fusion": {
            "iterations": 5,
            "input_location_weighting": "equal after phase alignment and unit-norm normalization",
            "healthy_references": "four fused references from two recordings and two sweep directions",
            "damage_estimates": "two fused estimates from the upward and downward sweep passages",
            "pass_aggregation": "median edge score",
        },
        "healthy_only_graph_hyperparameters": {
            "alpha_factor": float(selected["alpha_factor"]),
            "alpha": selected_alpha,
            "nugget_ratio": float(selected["nugget_ratio"]),
            "selection_rule": "twofold held-out healthy negative log likelihood",
        },
        "region_scan": {
            "minimum_m": float(SCAN_CENTERS_M.min()),
            "maximum_m": float(SCAN_CENTERS_M.max()),
            "spacing_m": float(SCAN_CENTERS_M[1] - SCAN_CENTERS_M[0]),
            "primary_bandwidth_m": 3.5,
            "success_tolerance_m": 3.5,
            "sensitivity_bandwidths_m": [1.75, 2.625, 3.5, 5.25],
        },
        "comparator_definition": (
            "endpoint_node_energy uses raw endpoint magnitude; "
            "normalized_endpoint_node_energy normalizes every modal residual "
            "field before endpoint averaging; direct_node_coordinate smooths "
            "the normalized node field directly at sensor coordinates without "
            "forming candidate-edge scores"
        ),
        "ground_truth_x_m": truth,
        "methods": {},
    }
    for method, scores in edge_scores.items():
        support_x = node_x if method == "direct_node_coordinate" else edge_x
        primary = summarize(scores, support_x, truth)
        result["methods"][method] = {
            "primary": primary,
            "bandwidth_sensitivity": {
                str(bandwidth): summarize(scores, support_x, truth, bandwidth)
                for bandwidth in (1.75, 2.625, 3.5, 5.25)
            },
            "permutation": permutation(
                scores, support_x, truth, primary, args.permutations
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(5, 1, figsize=(10.5, 16.0), sharex=True)
    y = np.arange(1, 9)
    for axis, method in zip(
        axes,
        (
            "endpoint_node_energy",
            "normalized_endpoint_node_energy",
            "direct_node_coordinate",
            "endpoint_contrast",
            "graph_propagated",
        ),
    ):
        summary = result["methods"][method]["primary"]["states"]
        for index, state in enumerate((f"DS{i}" for i in range(1, 9))):
            axis.scatter(
                summary[state]["ground_truth_x_m"],
                np.full(len(summary[state]["ground_truth_x_m"]), y[index]),
                marker="*",
                s=130,
                color="black",
            )
            axis.scatter(
                summary[state]["predicted_x_m"],
                np.full(len(summary[state]["predicted_x_m"]), y[index]),
                marker="o",
                s=50,
                color="#315b7d",
            )
        axis.set_yticks(y, [f"DS{i}" for i in range(1, 9)])
        axis.set_title(method.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Longitudinal coordinate x (m); star: damage, circle: inferred")
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
