"""Simple-baseline and graph-construction sensitivity audit for HBTA DS1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_damage_gp import (
    PhysicalGraph,
    aligned_residuals,
    endpoint_contrast_alignment_matrix,
    endpoint_node_energy_matrix,
    normalized_endpoint_node_energy_matrix,
    fit_mode_shape_template,
    load_physical_graph,
    shuffled_graph,
)
from .run_ds1_localization_audit import GROUND_TRUTH_EDGE, locate_edge_index


def rank_of(values: np.ndarray, index: int) -> int:
    return int(1 + np.sum(values > values[index]))


def aggregate_summary(matrix: np.ndarray, target_index: int) -> dict:
    aggregate = np.median(matrix, axis=0)
    return {
        "target_pass_ranks": [rank_of(row, target_index) for row in matrix],
        "target_aggregate_rank": rank_of(aggregate, target_index),
        "target_aggregate_score": float(aggregate[target_index]),
    }


def edge_scores(residuals: np.ndarray, graph: PhysicalGraph, method: str) -> np.ndarray:
    if method == "normalized_edge_gradient":
        return endpoint_contrast_alignment_matrix(residuals, graph)
    projection = np.einsum("en,bmn->bme", graph.incidence, residuals, optimize=True)
    if method == "raw_edge_difference":
        return np.mean(np.abs(projection) ** 2, axis=1)
    if method == "endpoint_node_energy":
        return endpoint_node_energy_matrix(residuals, graph)
    if method == "normalized_endpoint_node_energy":
        return normalized_endpoint_node_energy_matrix(residuals, graph)
    raise ValueError(f"Unknown score method: {method}")


def healthy_loro_scores(
    shapes: np.ndarray,
    recordings: np.ndarray,
    graph: PhysicalGraph,
    method: str,
) -> np.ndarray:
    matrices = []
    for recording in np.unique(recordings):
        heldout = recordings == recording
        template = fit_mode_shape_template(shapes[~heldout])
        residuals = aligned_residuals(shapes[heldout], template)
        matrices.append(edge_scores(residuals, graph, method))
    return np.concatenate(matrices)


def evaluate_method(
    features: dict,
    position: str,
    graph: PhysicalGraph,
    method: str,
    permutations: int = 0,
) -> dict:
    positions = features["positions"].astype(str)
    states = features["states"].astype(str)
    recordings = features["recordings"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    current = positions == position
    healthy = current & (states == "UDS")
    damaged = current & (states == "DS1")
    target_index = locate_edge_index(graph.edge_names, GROUND_TRUTH_EDGE)
    template = fit_mode_shape_template(shapes[healthy])
    damaged_residuals = aligned_residuals(shapes[damaged], template)
    damaged_scores = edge_scores(damaged_residuals, graph, method)
    healthy_scores = healthy_loro_scores(
        shapes[healthy], recordings[healthy], graph, method
    )
    result = {
        "damage": aggregate_summary(damaged_scores, target_index),
        "healthy_loro": aggregate_summary(healthy_scores, target_index),
    }
    if method == "normalized_edge_gradient":
        healthy_center = np.median(healthy_scores, axis=0)
        corrected = damaged_scores - healthy_center[None, :]
        result["healthy_median_subtracted"] = aggregate_summary(corrected, target_index)
    if permutations:
        observed = np.mean(
            1.0 / np.asarray(result["damage"]["target_pass_ranks"], dtype=float)
        )
        observed_corrected = (
            np.mean(
                1.0
                / np.asarray(
                    result["healthy_median_subtracted"]["target_pass_ranks"],
                    dtype=float,
                )
            )
            if method == "normalized_edge_gradient"
            else None
        )
        null = []
        null_corrected = []
        for seed in range(permutations):
            control = shuffled_graph(graph, seed)
            matrix = edge_scores(damaged_residuals, control, method)
            summary = aggregate_summary(matrix, target_index)
            null.append(
                np.mean(1.0 / np.asarray(summary["target_pass_ranks"], dtype=float))
            )
            if method == "normalized_edge_gradient":
                control_healthy = healthy_loro_scores(
                    shapes[healthy], recordings[healthy], control, method
                )
                center = np.median(control_healthy, axis=0)
                corrected_summary = aggregate_summary(
                    matrix - center[None, :], target_index
                )
                null_corrected.append(
                    np.mean(
                        1.0
                        / np.asarray(
                            corrected_summary["target_pass_ranks"], dtype=float
                        )
                    )
                )
        null = np.asarray(null)
        result["damage"]["permutation_mrr_one_sided_p"] = float(
            (1 + np.sum(null >= observed)) / (1 + permutations)
        )
        if method == "normalized_edge_gradient":
            null_corrected = np.asarray(null_corrected)
            result["healthy_median_subtracted"][
                "permutation_mrr_one_sided_p"
            ] = float(
                (1 + np.sum(null_corrected >= observed_corrected))
                / (1 + permutations)
            )
    return result


def read_coordinates(path: Path, sensor_names: tuple[str, ...]) -> np.ndarray:
    coordinates: dict[str, tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            coordinates.setdefault(
                row["sensor_i"],
                (float(row["x_i_m"]), float(row["y_i_m"]), float(row["z_i_m"])),
            )
            coordinates.setdefault(
                row["sensor_j"],
                (float(row["x_j_m"]), float(row["y_j_m"]), float(row["z_j_m"])),
            )
    return np.asarray([coordinates[name] for name in sensor_names], dtype=float)


def graph_from_pairs(
    sensor_names: tuple[str, ...],
    pairs: list[tuple[str, str]],
    relation: str,
) -> PhysicalGraph:
    name_to_index = {name: index for index, name in enumerate(sensor_names)}
    canonical = sorted({tuple(sorted(pair)) for pair in pairs})
    incidence = np.zeros((len(canonical), len(sensor_names)), dtype=float)
    for index, (left, right) in enumerate(canonical):
        incidence[index, name_to_index[left]] = 1.0
        incidence[index, name_to_index[right]] = -1.0
    weights = np.ones(len(canonical), dtype=float)
    return PhysicalGraph(
        sensor_names=sensor_names,
        edge_names=tuple(canonical),
        relations=tuple(relation for _ in canonical),
        incidence=incidence,
        weights=weights,
        laplacian=incidence.T @ incidence,
    )


def subset_graph(graph: PhysicalGraph, keep: np.ndarray) -> PhysicalGraph:
    incidence = graph.incidence[keep]
    weights = graph.weights[keep]
    return PhysicalGraph(
        sensor_names=graph.sensor_names,
        edge_names=tuple(edge for edge, flag in zip(graph.edge_names, keep) if flag),
        relations=tuple(rel for rel, flag in zip(graph.relations, keep) if flag),
        incidence=incidence,
        weights=weights,
        laplacian=incidence.T @ (weights[:, None] * incidence),
    )


def geometry_knn_pairs(
    sensor_names: tuple[str, ...], coordinates: np.ndarray, neighbors: int
) -> list[tuple[str, str]]:
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    pairs = []
    for index in range(len(sensor_names)):
        for other in np.argsort(distance[index])[:neighbors]:
            pairs.append((sensor_names[index], sensor_names[int(other)]))
    return pairs


def topology_variants(physical: PhysicalGraph, edge_list: Path) -> dict[str, PhysicalGraph]:
    coordinates = read_coordinates(edge_list, physical.sensor_names)
    knn4 = geometry_knn_pairs(physical.sensor_names, coordinates, 4)
    no_interface = np.asarray(
        [relation != "deck_truss_interface" for relation in physical.relations]
    )
    dense_pairs = list(physical.edge_names) + geometry_knn_pairs(
        physical.sensor_names, coordinates, 4
    )
    return {
        "physical": physical,
        "physical_without_interface": subset_graph(physical, no_interface),
        "four_nearest_geometry": graph_from_pairs(
            physical.sensor_names, knn4, "geometry_knn4"
        ),
        "physical_plus_geometry": graph_from_pairs(
            physical.sensor_names, dense_pairs, "physical_plus_geometry"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-features", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    cache = np.load(args.sweep_features, allow_pickle=False)
    features = {name: cache[name] for name in cache.files}
    physical = load_physical_graph(
        args.edge_list, features["response_sensor_names"]
    )
    methods = (
        "normalized_edge_gradient",
        "raw_edge_difference",
        "endpoint_node_energy",
        "normalized_endpoint_node_energy",
    )
    baseline = {
        method: {
            position: evaluate_method(
                features, position, physical, method, args.permutations
            )
            for position in ("P1", "P2")
        }
        for method in methods
    }
    topology = {}
    for name, graph in topology_variants(physical, args.edge_list).items():
        represented = frozenset(GROUND_TRUTH_EDGE) in {
            frozenset(edge) for edge in graph.edge_names
        }
        topology[name] = {
            "candidate_edge_count": len(graph.edge_names),
            "target_edge_represented": represented,
        }
        if represented:
            topology[name]["positions"] = {
                position: evaluate_method(
                    features, position, graph, "normalized_edge_gradient", 0
                )["damage"]
                for position in ("P1", "P2")
            }

    result = {
        "status": "ds1_simple_baseline_and_graph_construction_audit",
        "permutation_count": args.permutations,
        "score_definitions": {
            "normalized_edge_gradient": "mean_m |b_e^H r_m|^2/(||b_e||^2 ||r_m||^2)",
            "raw_edge_difference": "mean_m |b_e^H r_m|^2",
            "endpoint_node_energy": "mean modal residual energy at the two edge endpoints",
            "normalized_endpoint_node_energy": "mean endpoint-node energy after unit total residual energy is imposed separately for every mode",
            "healthy_median_subtracted": "normalized edge score minus the edge-specific median LORO healthy score",
        },
        "simple_baselines_on_physical_graph": baseline,
        "graph_construction_sensitivity": topology,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    labels = ["edge gradient", "raw difference", "node raw", "node normalized", "healthy corrected"]
    values = {
        "P1": [
            baseline["normalized_edge_gradient"]["P1"]["damage"]["target_aggregate_rank"],
            baseline["raw_edge_difference"]["P1"]["damage"]["target_aggregate_rank"],
            baseline["endpoint_node_energy"]["P1"]["damage"]["target_aggregate_rank"],
            baseline["normalized_endpoint_node_energy"]["P1"]["damage"]["target_aggregate_rank"],
            baseline["normalized_edge_gradient"]["P1"]["healthy_median_subtracted"]["target_aggregate_rank"],
        ],
        "P2": [
            baseline["normalized_edge_gradient"]["P2"]["damage"]["target_aggregate_rank"],
            baseline["raw_edge_difference"]["P2"]["damage"]["target_aggregate_rank"],
            baseline["endpoint_node_energy"]["P2"]["damage"]["target_aggregate_rank"],
            baseline["normalized_endpoint_node_energy"]["P2"]["damage"]["target_aggregate_rank"],
            baseline["normalized_edge_gradient"]["P2"]["healthy_median_subtracted"]["target_aggregate_rank"],
        ],
    }
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, values["P1"], width, label="P1")
    axes[0].bar(x + width / 2, values["P2"], width, label="P2")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Aggregate rank of AL14-AL18")
    axes[0].set_title("Simple-baseline audit")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    topo_names = [name for name, item in topology.items() if item["target_edge_represented"]]
    p1 = [topology[name]["positions"]["P1"]["target_aggregate_rank"] for name in topo_names]
    p2 = [topology[name]["positions"]["P2"]["target_aggregate_rank"] for name in topo_names]
    x = np.arange(len(topo_names))
    axes[1].bar(x - width / 2, p1, width, label="P1")
    axes[1].bar(x + width / 2, p2, width, label="P2")
    axes[1].set_xticks(x, [name.replace("_", " ") for name in topo_names], rotation=18, ha="right")
    axes[1].set_ylabel("Aggregate rank of AL14-AL18")
    axes[1].set_title("Graph-construction sensitivity")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
