"""Exploratory DS1--DS8 structural-region localization.

The HBTA damage states do not all map to one measured sensor edge.  This
experiment therefore tests a coarser, documented target: the structural
subsystem affected by each intervention.  Single-edge scores are computed
without damage labels, then aggregated within drawing-derived edge families.

This is an exploratory gate, not confirmatory inference.  The primary
aggregation rule (mean of the largest 20 percent of edge scores in a family)
is declared here, and several fixed alternatives are reported as sensitivity
checks rather than selected on the damage results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_damage_gp import (
    aligned_residuals,
    damage_alignment_matrix,
    endpoint_contrast_alignment_matrix,
    fit_healthy_graph_model,
    fit_mode_shape_template,
    load_physical_graph,
)
from .run_graph_damage_gp_conformal import twofold_tune


# Mapping fixed from the HBTA intervention descriptions and the drawing-derived
# meaning of the sensor-projected graph edges.  It is deliberately a subsystem
# target, not an assertion that an uninstrumented joint is one measured edge.
EXPECTED_RELATION = {
    "DS1": "deck_longitudinal",
    "DS2": "deck_longitudinal",
    "DS3": "deck_transverse",
    "DS4": "deck_transverse",
    "DS5": "lateral_bracing",
    "DS6": "lateral_bracing",
    "DS7": "lateral_bracing",
    "DS8": "deck_truss_interface",
}

TYPE_GROUPS = {
    "DT1_stringer_floor_beam": ("DS1", "DS2"),
    "DT2_stringer_cross_beam": ("DS3", "DS4"),
    "DT3_lateral_bracing": ("DS5", "DS6", "DS7"),
    "DT4_floor_beam_main_member": ("DS8",),
}

AGGREGATORS = {
    "primary_top20_mean": ("top_fraction", 0.20),
    "top10_mean": ("top_fraction", 0.10),
    "top30_mean": ("top_fraction", 0.30),
    "top3_mean": ("top_count", 3),
    "maximum": ("top_count", 1),
    "all_edge_mean": ("top_fraction", 1.00),
}


def aggregate_edge_families(
    edge_scores: np.ndarray,
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    rule: tuple[str, float | int],
) -> np.ndarray:
    """Aggregate block-by-edge scores into equal-status structural families."""
    output = np.empty((len(edge_scores), len(relation_order)), dtype=float)
    relation_array = np.asarray(relations)
    kind, value = rule
    for relation_index, relation in enumerate(relation_order):
        current = edge_scores[:, relation_array == relation]
        if current.shape[1] == 0:
            raise ValueError(f"No graph edges for relation {relation}")
        if kind == "top_fraction":
            count = max(1, int(np.ceil(float(value) * current.shape[1])))
        elif kind == "top_count":
            count = min(int(value), current.shape[1])
        else:
            raise ValueError(f"Unknown aggregation rule {kind}")
        top = np.partition(current, current.shape[1] - count, axis=1)[:, -count:]
        output[:, relation_index] = np.mean(top, axis=1)
    return output


def descending_rank(values: np.ndarray, target_index: int) -> np.ndarray:
    return 1 + np.sum(values > values[:, [target_index]], axis=1)


def summarize_states(
    matrices: dict[str, np.ndarray], relation_order: tuple[str, ...]
) -> dict:
    states: dict[str, object] = {}
    reciprocal_ranks = []
    correct = []
    for state in sorted(EXPECTED_RELATION):
        matrix = matrices[state]
        target = relation_order.index(EXPECTED_RELATION[state])
        block_ranks = descending_rank(matrix, target)
        aggregate = np.median(matrix, axis=0)
        aggregate_rank = int(1 + np.sum(aggregate > aggregate[target]))
        predicted = relation_order[int(np.argmax(aggregate))]
        reciprocal_ranks.append(1.0 / aggregate_rank)
        correct.append(aggregate_rank == 1)
        order = np.argsort(-aggregate)
        states[state] = {
            "expected_relation": EXPECTED_RELATION[state],
            "predicted_relation": predicted,
            "aggregate_rank": aggregate_rank,
            "block_top1_fraction": float(np.mean(block_ranks == 1)),
            "block_median_rank": float(np.median(block_ranks)),
            "aggregate_relation_scores": {
                relation_order[index]: float(aggregate[index]) for index in order
            },
        }

    type_reciprocal_ranks = []
    type_correct = []
    types: dict[str, object] = {}
    for type_name, members in TYPE_GROUPS.items():
        matrix = np.concatenate([matrices[state] for state in members], axis=0)
        expected = EXPECTED_RELATION[members[0]]
        target = relation_order.index(expected)
        aggregate = np.median(matrix, axis=0)
        aggregate_rank = int(1 + np.sum(aggregate > aggregate[target]))
        predicted = relation_order[int(np.argmax(aggregate))]
        type_reciprocal_ranks.append(1.0 / aggregate_rank)
        type_correct.append(aggregate_rank == 1)
        types[type_name] = {
            "states": list(members),
            "expected_relation": expected,
            "predicted_relation": predicted,
            "aggregate_rank": aggregate_rank,
        }
    return {
        "state_top1_accuracy": float(np.mean(correct)),
        "state_mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "type_top1_accuracy": float(np.mean(type_correct)),
        "type_mean_reciprocal_rank": float(np.mean(type_reciprocal_ranks)),
        "states": states,
        "damage_types": types,
    }


def permutation_test(
    edge_matrices: dict[str, np.ndarray],
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    rule: tuple[str, float | int],
    observed: dict,
    count: int,
) -> dict:
    """Shuffle family labels over edges while preserving family sizes."""
    generator = np.random.default_rng(20260829)
    labels = np.asarray(relations)
    null_state_mrr = np.empty(count)
    null_type_mrr = np.empty(count)
    null_state_accuracy = np.empty(count)
    null_type_accuracy = np.empty(count)
    for index in range(count):
        shuffled = tuple(generator.permutation(labels).tolist())
        grouped = {
            state: aggregate_edge_families(matrix, shuffled, relation_order, rule)
            for state, matrix in edge_matrices.items()
        }
        summary = summarize_states(grouped, relation_order)
        null_state_mrr[index] = summary["state_mean_reciprocal_rank"]
        null_type_mrr[index] = summary["type_mean_reciprocal_rank"]
        null_state_accuracy[index] = summary["state_top1_accuracy"]
        null_type_accuracy[index] = summary["type_top1_accuracy"]

    def upper_p(null: np.ndarray, value: float) -> float:
        return float((1 + np.sum(null >= value)) / (1 + len(null)))

    return {
        "count": count,
        "scheme": "shuffle edge-family labels while preserving family sizes",
        "state_mrr_one_sided_p": upper_p(
            null_state_mrr, observed["state_mean_reciprocal_rank"]
        ),
        "type_mrr_one_sided_p": upper_p(
            null_type_mrr, observed["type_mean_reciprocal_rank"]
        ),
        "state_accuracy_one_sided_p": upper_p(
            null_state_accuracy, observed["state_top1_accuracy"]
        ),
        "type_accuracy_one_sided_p": upper_p(
            null_type_accuracy, observed["type_top1_accuracy"]
        ),
        "state_mrr_null_05_50_95": np.quantile(
            null_state_mrr, [0.05, 0.5, 0.95]
        ).tolist(),
        "type_mrr_null_05_50_95": np.quantile(
            null_type_mrr, [0.05, 0.5, 0.95]
        ).tolist(),
    }


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
    recording_ids = features["recording_ids"]
    states = features["states"].astype(str)
    reference = shapes[recording_ids == 0]
    external = shapes[recording_ids == 1]
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    relation_order = tuple(sorted(set(graph.relations)))

    selected = twofold_tune(graph, reference)["selected"]
    template = fit_mode_shape_template(reference)
    healthy_residuals = aligned_residuals(reference, template)
    model = fit_healthy_graph_model(
        graph,
        healthy_residuals,
        selected["alpha_factor"],
        selected["nugget_ratio"],
    )

    residuals = {
        state: aligned_residuals(shapes[states == state], template)
        for state in sorted(EXPECTED_RELATION)
    }
    residuals["external_healthy"] = aligned_residuals(external, template)
    physical_edges = {
        state: damage_alignment_matrix(value, graph, model)
        for state, value in residuals.items()
    }
    endpoint_edges = {
        state: endpoint_contrast_alignment_matrix(value, graph)
        for state, value in residuals.items()
    }

    result: dict[str, object] = {
        "status": "exploratory_structural_subsystem_localization_gate",
        "target_definition": {
            "scope": (
                "documented structural subsystem represented by a graph-edge family; "
                "not exact localization of an uninstrumented joint"
            ),
            "expected_relation_by_state": EXPECTED_RELATION,
            "damage_type_groups": TYPE_GROUPS,
            "candidate_relations": relation_order,
            "edge_count_by_relation": {
                relation: int(np.sum(np.asarray(graph.relations) == relation))
                for relation in relation_order
            },
        },
        "protocol": {
            "healthy_template_and_graph_hyperparameters": "14 UDS01 blocks only",
            "damage_test": "14 blocks from each DS1--DS8 record",
            "external_healthy_control": "14 UDS02 blocks",
            "primary_rule": "mean of largest 20 percent of edge scores per family",
            "permutation_unit": "complete edge-family label assignment",
            "block_warning": "blocks are repeated estimates within one record",
        },
        "healthy_only_hyperparameters": selected,
        "methods": {},
    }

    for method_name, edge_matrices in (
        ("graph_propagated", physical_edges),
        ("endpoint_contrast", endpoint_edges),
    ):
        method_result: dict[str, object] = {"aggregator_sensitivity": {}}
        for aggregator_name, rule in AGGREGATORS.items():
            grouped = {
                state: aggregate_edge_families(
                    edge_matrices[state], graph.relations, relation_order, rule
                )
                for state in EXPECTED_RELATION
            }
            summary = summarize_states(grouped, relation_order)
            method_result["aggregator_sensitivity"][aggregator_name] = summary
            if aggregator_name == "primary_top20_mean":
                method_result["primary"] = summary
                method_result["permutation"] = permutation_test(
                    {state: edge_matrices[state] for state in EXPECTED_RELATION},
                    graph.relations,
                    relation_order,
                    rule,
                    summary,
                    args.permutations,
                )

        healthy_grouped = aggregate_edge_families(
            edge_matrices["external_healthy"],
            graph.relations,
            relation_order,
            AGGREGATORS["primary_top20_mean"],
        )
        method_result["external_healthy_control"] = {
            "median_max_family_score": float(
                np.median(np.max(healthy_grouped, axis=1))
            ),
            "aggregate_dominant_relation": relation_order[
                int(np.argmax(np.median(healthy_grouped, axis=0)))
            ],
            "aggregate_relation_scores": {
                relation: float(value)
                for relation, value in zip(
                    relation_order, np.median(healthy_grouped, axis=0)
                )
            },
        }
        result["methods"][method_name] = method_result

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.3), sharex=True)
    display = {
        "deck_longitudinal": "deck longitudinal",
        "deck_transverse": "deck transverse",
        "deck_truss_interface": "deck--truss interface",
        "lateral_bracing": "lateral bracing",
        "truss_chord": "truss chord",
        "truss_diagonal": "truss diagonal",
    }
    colors = plt.cm.Set2(np.linspace(0, 1, len(relation_order)))
    x = np.arange(len(EXPECTED_RELATION))
    width = 0.12
    for axis, (method_name, edge_matrices) in zip(
        axes,
        (("Graph-propagated", physical_edges), ("Endpoint contrast", endpoint_edges)),
    ):
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
        for relation_index, relation in enumerate(relation_order):
            axis.bar(
                x + (relation_index - (len(relation_order) - 1) / 2) * width,
                values[:, relation_index],
                width,
                label=display[relation],
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
        axis.set_ylabel("Aggregated consistency score")
        axis.set_title(method_name)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, list(EXPECTED_RELATION))
    axes[-1].set_xlabel("HBTA damage state (star: documented subsystem)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
