"""Spatial-region localization of HBTA DS1--DS8 across P1 and P2.

Unlike structural-family classification, this audit pools every graph-edge
type near a candidate longitudinal coordinate.  Ground-truth longitudinal
coordinates are digitized from the published HBTA damage schematic before
the response scores are inspected.  The drawing coordinate is calibrated by
the known DS1 graph location at x=-3.5 m.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .graph_damage_gp import load_physical_graph
from .run_all_damage_sweep_region import edge_score_matrices
from .run_graph_damage_gp_conformal import twofold_tune


# Pixel coordinates refer to the 3x crop saved as
# outputs/hbta_damage_map_source_crop.png.  They are centroids of thresholded
# red intervention marks; axes are the drawn longitudinal bridge centerlines.
DECK_AXIS_LEFT = np.asarray([155.0, 540.0])
DECK_AXIS_RIGHT = np.asarray([1290.0, 410.0])
DECK_DAMAGE_PIXELS = {
    "DS3": np.asarray([480.7586, 520.9598]),
    "DS4": np.asarray([578.3333, 501.3111]),
    "DS1": np.asarray([646.9927, 503.9307]),
    "DS2": np.asarray([646.9927, 503.9307]),
    "DS8": np.asarray([828.6515, 451.9169]),
}
BRACING_AXIS_LEFT = np.asarray([155.0, 1158.0])
BRACING_AXIS_RIGHT = np.asarray([1288.0, 1028.0])
BRACING_DAMAGE_PIXELS = {
    "DS5": np.asarray([489.1023, 1192.9395]),
    "DS6": np.asarray([666.6335, 1061.6545]),
}

PRIMARY_BANDWIDTH_M = 3.5
PRIMARY_TOLERANCE_M = 3.5
SCAN_STEP_M = 0.25
SCAN_CENTERS_M = np.arange(-14.0, 14.0 + SCAN_STEP_M / 2.0, SCAN_STEP_M)
BANDWIDTHS_M = (1.75, 2.625, 3.5, 5.25)


def projected_coordinate(
    pixel: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    span_m: float = 35.0,
) -> float:
    """Project a schematic mark to the bridge longitudinal drawing axis."""
    direction = right - left
    unit = direction / np.linalg.norm(direction)
    scale = span_m / float((right - left) @ unit)
    return float(-span_m / 2.0 + scale * ((pixel - left) @ unit))


def locked_ground_truth() -> dict[str, tuple[float, ...]]:
    # The drawing endpoints give the scale.  A small common offset is then
    # applied so the independently established DS1 location is exactly -3.5 m.
    raw_deck = {
        state: projected_coordinate(pixel, DECK_AXIS_LEFT, DECK_AXIS_RIGHT)
        for state, pixel in DECK_DAMAGE_PIXELS.items()
    }
    deck_offset = -3.5 - raw_deck["DS1"]
    deck = {state: value + deck_offset for state, value in raw_deck.items()}
    bracing = {
        state: projected_coordinate(
            pixel, BRACING_AXIS_LEFT, BRACING_AXIS_RIGHT
        )
        for state, pixel in BRACING_DAMAGE_PIXELS.items()
    }
    return {
        "DS1": (deck["DS1"],),
        "DS2": (deck["DS2"],),
        "DS3": (deck["DS3"],),
        "DS4": (deck["DS4"],),
        "DS5": (bracing["DS5"],),
        "DS6": (bracing["DS6"],),
        "DS7": (bracing["DS5"], bracing["DS6"]),
        "DS8": (deck["DS8"],),
    }


def load_edge_midpoints(edge_list: Path, graph) -> np.ndarray:
    rows = []
    with edge_list.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    row["sensor_i"],
                    row["sensor_j"],
                    0.5 * (float(row["x_i_m"]) + float(row["x_j_m"])),
                )
            )
    lookup = {frozenset((left, right)): midpoint for left, right, midpoint in rows}
    return np.asarray(
        [lookup[frozenset(edge)] for edge in graph.edge_names], dtype=float
    )


def gaussian_region_matrix(
    edge_x: np.ndarray,
    centers: np.ndarray,
    bandwidth_m: float,
) -> np.ndarray:
    kernel = np.exp(
        -0.5 * ((centers[:, None] - edge_x[None, :]) / bandwidth_m) ** 2
    )
    return kernel / np.maximum(
        np.sum(kernel, axis=1, keepdims=True), np.finfo(float).eps
    )


def select_peaks(
    curve: np.ndarray,
    centers: np.ndarray,
    count: int,
    minimum_separation_m: float = 3.5,
) -> np.ndarray:
    distance = max(1, int(round(minimum_separation_m / SCAN_STEP_M)))
    candidates, _ = find_peaks(curve, distance=distance)
    candidates = list(candidates[np.argsort(-curve[candidates])])
    # Include boundaries and provide a deterministic fallback when smoothing
    # produces fewer local maxima than required.
    candidates.extend([0, len(curve) - 1])
    for index in np.argsort(-curve):
        if all(abs(index - chosen) >= distance for chosen in candidates[:count]):
            candidates.append(int(index))
    chosen: list[int] = []
    for index in candidates:
        if all(abs(index - previous) >= distance for previous in chosen):
            chosen.append(int(index))
        if len(chosen) == count:
            break
    return np.sort(centers[np.asarray(chosen, dtype=int)])


def matched_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    cost = np.abs(predicted[:, None] - truth[None, :])
    rows, columns = linear_sum_assignment(cost)
    return cost[rows, columns]


def summarize_spatial_localization(
    edge_matrices: dict[str, dict[str, np.ndarray]],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    bandwidth_m: float,
) -> dict:
    region_matrix = gaussian_region_matrix(edge_x, SCAN_CENTERS_M, bandwidth_m)
    results = {}
    all_errors = []
    state_both = []
    for state, truth_values in truth.items():
        truth_array = np.asarray(truth_values, dtype=float)
        state_results = {}
        state_success = []
        for position in ("P1", "P2"):
            record_edge_score = np.median(
                edge_matrices[position][state], axis=0
            )
            curve = region_matrix @ record_edge_score
            predicted = select_peaks(
                curve, SCAN_CENTERS_M, count=len(truth_array)
            )
            errors = matched_errors(predicted, truth_array)
            all_errors.extend(errors.tolist())
            success = bool(np.all(errors <= PRIMARY_TOLERANCE_M))
            state_success.append(success)
            state_results[position] = {
                "predicted_x_m": predicted.tolist(),
                "matched_absolute_errors_m": errors.tolist(),
                "all_targets_within_one_panel": success,
                "dominant_region_score": float(np.max(curve)),
            }
        both = bool(all(state_success))
        state_both.append(both)
        results[state] = {
            "ground_truth_x_m": list(truth_values),
            "positions": state_results,
            "correct_at_both_positions": both,
        }
    return {
        "bandwidth_m": bandwidth_m,
        "tolerance_m": PRIMARY_TOLERANCE_M,
        "mean_absolute_localization_error_m": float(np.mean(all_errors)),
        "median_absolute_localization_error_m": float(np.median(all_errors)),
        "target_fraction_within_one_panel": float(
            np.mean(np.asarray(all_errors) <= PRIMARY_TOLERANCE_M)
        ),
        "state_fraction_correct_at_both_positions": float(np.mean(state_both)),
        "states": results,
    }


def permutation_test(
    edge_matrices: dict[str, dict[str, np.ndarray]],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    observed: dict,
    bandwidth_m: float,
    count: int,
) -> dict:
    generator = np.random.default_rng(20260829)
    null_mean_error = np.empty(count)
    null_target_fraction = np.empty(count)
    null_state_both = np.empty(count)
    for index in range(count):
        shuffled_x = generator.permutation(edge_x)
        summary = summarize_spatial_localization(
            edge_matrices, shuffled_x, truth, bandwidth_m
        )
        null_mean_error[index] = summary["mean_absolute_localization_error_m"]
        null_target_fraction[index] = summary["target_fraction_within_one_panel"]
        null_state_both[index] = summary[
            "state_fraction_correct_at_both_positions"
        ]

    def lower_p(null: np.ndarray, value: float) -> float:
        return float((1 + np.sum(null <= value)) / (1 + len(null)))

    def upper_p(null: np.ndarray, value: float) -> float:
        return float((1 + np.sum(null >= value)) / (1 + len(null)))

    return {
        "count": count,
        "scheme": (
            "permute longitudinal edge coordinates jointly for all states and "
            "both excitation positions; retain edge scores and graph relations"
        ),
        "mean_error_one_sided_p": lower_p(
            null_mean_error, observed["mean_absolute_localization_error_m"]
        ),
        "target_fraction_one_sided_p": upper_p(
            null_target_fraction, observed["target_fraction_within_one_panel"]
        ),
        "state_both_positions_one_sided_p": upper_p(
            null_state_both,
            observed["state_fraction_correct_at_both_positions"],
        ),
        "mean_error_null_05_50_95": np.quantile(
            null_mean_error, [0.05, 0.5, 0.95]
        ).tolist(),
        "target_fraction_null_05_50_95": np.quantile(
            null_target_fraction, [0.05, 0.5, 0.95]
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

    sweep = np.load(args.sweep_features, allow_pickle=False)
    features = {key: sweep[key] for key in sweep.files}
    noise = np.load(args.noise_features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    noise_reference = noise["mode_shapes"][noise["recording_ids"] == 0]
    noise_selected = twofold_tune(graph, noise_reference)["selected"]
    matrices = edge_score_matrices(features, graph, noise_selected)
    edge_x = load_edge_midpoints(args.edge_list, graph)
    truth = locked_ground_truth()

    result = {
        "status": "exploratory_drawing_locked_spatial_region_localization",
        "ground_truth": {
            "coordinates_x_m": truth,
            "source": "published HBTA damage schematic, red-mark digitization",
            "calibration": "common deck offset fixes independently known DS1 x=-3.5 m",
            "scope": (
                "longitudinal region; not exact joint/member identification; "
                "DS7 has two target regions"
            ),
            "primary_tolerance_m": PRIMARY_TOLERANCE_M,
        },
        "spatial_model": {
            "candidate_centers_m": [
                float(SCAN_CENTERS_M[0]),
                float(SCAN_CENTERS_M[-1]),
                SCAN_STEP_M,
            ],
            "primary_kernel": "normalized Gaussian average over all edge types",
            "primary_bandwidth_m": PRIMARY_BANDWIDTH_M,
        },
        "methods": {},
    }
    for method, edge_matrices in matrices.items():
        primary = summarize_spatial_localization(
            edge_matrices, edge_x, truth, PRIMARY_BANDWIDTH_M
        )
        result["methods"][method] = {
            "primary": primary,
            "bandwidth_sensitivity": {
                str(bandwidth): summarize_spatial_localization(
                    edge_matrices, edge_x, truth, bandwidth
                )
                for bandwidth in BANDWIDTHS_M
            },
            "permutation": permutation_test(
                edge_matrices,
                edge_x,
                truth,
                primary,
                PRIMARY_BANDWIDTH_M,
                args.permutations,
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    colors = {"P1": "#315b7d", "P2": "#d26a3a"}
    offsets = {"P1": -0.12, "P2": 0.12}
    y = np.arange(1, 9)
    for axis, method in zip(axes, ("endpoint_contrast", "graph_propagated")):
        summary = result["methods"][method]["primary"]["states"]
        for state_index, state in enumerate([f"DS{i}" for i in range(1, 9)]):
            for truth_x in truth[state]:
                axis.scatter(
                    truth_x,
                    y[state_index],
                    marker="*",
                    s=130,
                    color="black",
                    zorder=5,
                )
            for position in ("P1", "P2"):
                predicted = summary[state]["positions"][position]["predicted_x_m"]
                axis.scatter(
                    predicted,
                    np.full(len(predicted), y[state_index] + offsets[position]),
                    marker="o",
                    s=45,
                    color=colors[position],
                    label=position if state_index == 0 else None,
                )
        axis.set_yticks(y, [f"DS{i}" for i in range(1, 9)])
        axis.set_ylabel("Damage state")
        axis.set_title(method.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[-1].set_xlabel(
        "Longitudinal coordinate x (m); star: digitized damage, circle: inferred region"
    )
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
