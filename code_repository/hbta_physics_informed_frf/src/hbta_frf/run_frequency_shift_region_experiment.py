"""Exploratory structural-family localization from modal-frequency shifts.

The first-order eigenvalue change for a stiffness loss at edge ``e`` is
proportional to ``-w_e |b_e^T phi_m|^2``.  This script compares that predicted
three-mode signature with measured frequency shifts, then aggregates the best
edge consistencies within each structural family.  It is a companion diagnostic
to the modal-vector multi-edge experiment, not a standalone confirmatory test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_damage_gp import fit_mode_shape_template, load_physical_graph, normalize_mode_shapes
from .run_multiedge_region_experiment import (
    AGGREGATORS,
    EXPECTED_RELATION,
    TYPE_GROUPS,
    aggregate_edge_families,
    permutation_test,
    summarize_states,
)


def frequency_consistency_matrix(
    shifts: np.ndarray,
    signatures: np.ndarray,
) -> np.ndarray:
    """Non-negative scale-fit improvement for blocks by candidate edge."""
    target_energy = np.sum(shifts**2, axis=1, keepdims=True)
    target_energy = np.maximum(target_energy, np.finfo(float).eps)
    signature_energy = np.sum(signatures**2, axis=1)
    signature_energy = np.maximum(signature_energy, np.finfo(float).eps)
    projection = shifts @ signatures.T
    positive = np.maximum(projection, 0.0)
    return np.clip(
        positive**2 / (target_energy * signature_energy[None, :]), 0.0, 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    shapes = features["mode_shapes"].astype(np.complex128)
    frequencies = features["modal_frequencies_hz"].astype(float)
    recording_ids = features["recording_ids"]
    states = features["states"].astype(str)
    reference_mask = recording_ids == 0

    complex_template = fit_mode_shape_template(shapes[reference_mask])
    real_template = normalize_mode_shapes(np.real(complex_template))
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    relation_order = tuple(sorted(set(graph.relations)))

    reference_lambda = (2.0 * np.pi * frequencies[reference_mask]) ** 2
    lambda_template = np.median(reference_lambda, axis=0)
    lambda_scale = np.std(reference_lambda, axis=0, ddof=1)
    lambda_scale = np.maximum(lambda_scale, 1e-8)
    contrasts = graph.incidence @ real_template.T
    signatures = -graph.weights[:, None] * contrasts**2 / lambda_scale[None, :]

    edge_matrices = {}
    for state in EXPECTED_RELATION:
        current_lambda = (2.0 * np.pi * frequencies[states == state]) ** 2
        shifts = (current_lambda - lambda_template[None, :]) / lambda_scale[None, :]
        edge_matrices[state] = frequency_consistency_matrix(shifts, signatures)
    external_lambda = (2.0 * np.pi * frequencies[recording_ids == 1]) ** 2
    external_shifts = (external_lambda - lambda_template[None, :]) / lambda_scale[None, :]
    edge_matrices["external_healthy"] = frequency_consistency_matrix(
        external_shifts, signatures
    )

    summaries = {}
    for aggregator_name, rule in AGGREGATORS.items():
        grouped = {
            state: aggregate_edge_families(
                edge_matrices[state], graph.relations, relation_order, rule
            )
            for state in EXPECTED_RELATION
        }
        summaries[aggregator_name] = summarize_states(grouped, relation_order)

    primary = summaries["primary_top20_mean"]
    result = {
        "status": "exploratory_first_order_frequency_shift_region_gate",
        "model": "delta lambda proportional to -w_e |b_e^T phi_m|^2",
        "expected_relation_by_state": EXPECTED_RELATION,
        "damage_type_groups": TYPE_GROUPS,
        "candidate_relations": relation_order,
        "healthy_lambda_scale": lambda_scale.tolist(),
        "primary": primary,
        "aggregator_sensitivity": summaries,
        "permutation": permutation_test(
            {state: edge_matrices[state] for state in EXPECTED_RELATION},
            graph.relations,
            relation_order,
            AGGREGATORS["primary_top20_mean"],
            primary,
            args.permutations,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    grouped = {
        state: aggregate_edge_families(
            edge_matrices[state],
            graph.relations,
            relation_order,
            AGGREGATORS["primary_top20_mean"],
        )
        for state in EXPECTED_RELATION
    }
    values = np.asarray(
        [np.median(grouped[state], axis=0) for state in EXPECTED_RELATION]
    )
    figure, axis = plt.subplots(figsize=(10.8, 4.4))
    x = np.arange(len(EXPECTED_RELATION))
    width = 0.12
    colors = plt.cm.Set2(np.linspace(0, 1, len(relation_order)))
    for relation_index, relation in enumerate(relation_order):
        axis.bar(
            x + (relation_index - (len(relation_order) - 1) / 2) * width,
            values[:, relation_index],
            width,
            label=relation.replace("_", " "),
            color=colors[relation_index],
        )
    for state_index, state in enumerate(EXPECTED_RELATION):
        target = relation_order.index(EXPECTED_RELATION[state])
        axis.scatter(
            x[state_index] + (target - (len(relation_order) - 1) / 2) * width,
            values[state_index, target],
            marker="*",
            s=85,
            color="black",
            zorder=5,
        )
    axis.set_xticks(x, list(EXPECTED_RELATION))
    axis.set_ylabel("Frequency-shift consistency")
    axis.set_xlabel("HBTA damage state (star: documented subsystem)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
