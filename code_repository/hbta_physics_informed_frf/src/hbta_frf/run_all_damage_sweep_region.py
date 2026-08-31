"""Cross-excitation structural-family localization for HBTA DS1--DS8.

All raw swept-sine recordings are read in place.  Healthy records define one
modal template per shaker position.  Damage labels are used only after modal
vectors and edge scores have been computed.  Because DS2--DS8 do not have a
validated one-to-one sensor-edge mapping, the target is the documented
structural subsystem represented by a drawing-derived graph-edge family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import RAW_HDF5, RESPONSE_SENSORS
from .graph_damage_gp import (
    aligned_residuals,
    damage_alignment_matrix,
    endpoint_contrast_alignment_matrix,
    fit_healthy_graph_model,
    fit_mode_shape_template,
    load_physical_graph,
)
from .run_ds1_sweep_validation import (
    PRIMARY_NPERSEG,
    PRIMARY_WINDOWS_SECONDS,
    extract_sweep_features,
)
from .run_graph_damage_gp_conformal import twofold_tune
from .run_multiedge_region_experiment import (
    AGGREGATORS,
    EXPECTED_RELATION,
    aggregate_edge_families,
    summarize_states,
)


ALL_SWEEP_RECORDINGS = {
    position: tuple(
        [
            f"MVS_{position}_UDS_SM_Z_01",
            f"MVS_{position}_UDS_SM_Z_02",
        ]
        + [f"MVS_{position}_DS{i}_SM_Z_01" for i in range(1, 9)]
    )
    for position in ("P1", "P2")
}


def edge_score_matrices(
    features: dict,
    graph,
    noise_selected: dict,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    positions = features["positions"].astype(str)
    states = features["states"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    output = {"endpoint_contrast": {}, "graph_propagated": {}}
    for position in ("P1", "P2"):
        healthy = (positions == position) & (states == "UDS")
        template = fit_mode_shape_template(shapes[healthy])
        healthy_residual = aligned_residuals(shapes[healthy], template)
        model = fit_healthy_graph_model(
            graph,
            healthy_residual,
            noise_selected["alpha_factor"],
            noise_selected["nugget_ratio"],
        )
        output["endpoint_contrast"][position] = {}
        output["graph_propagated"][position] = {}
        for state in EXPECTED_RELATION:
            current = (positions == position) & (states == state)
            residual = aligned_residuals(shapes[current], template)
            output["endpoint_contrast"][position][state] = (
                endpoint_contrast_alignment_matrix(residual, graph)
            )
            output["graph_propagated"][position][state] = damage_alignment_matrix(
                residual, graph, model
            )
    return output


def summarize_method(
    edge_matrices: dict[str, dict[str, np.ndarray]],
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    rule: tuple[str, float | int],
) -> dict:
    grouped_by_position = {}
    position_summaries = {}
    for position in ("P1", "P2"):
        grouped = {
            state: aggregate_edge_families(
                edge_matrices[position][state], relations, relation_order, rule
            )
            for state in EXPECTED_RELATION
        }
        grouped_by_position[position] = grouped
        position_summaries[position] = summarize_states(grouped, relation_order)

    pooled = {
        state: np.concatenate(
            [grouped_by_position["P1"][state], grouped_by_position["P2"][state]],
            axis=0,
        )
        for state in EXPECTED_RELATION
    }
    pooled_summary = summarize_states(pooled, relation_order)
    correct_both = []
    reciprocal_ranks = []
    for state in EXPECTED_RELATION:
        ranks = [
            position_summaries[position]["states"][state]["aggregate_rank"]
            for position in ("P1", "P2")
        ]
        correct_both.append(all(rank == 1 for rank in ranks))
        reciprocal_ranks.extend(1.0 / np.asarray(ranks, dtype=float))
    return {
        "position_results": position_summaries,
        "pooled_positions": pooled_summary,
        "state_fraction_correct_at_both_positions": float(np.mean(correct_both)),
        "mean_reciprocal_rank_over_16_state_position_tests": float(
            np.mean(reciprocal_ranks)
        ),
    }


def permutation_test(
    edge_matrices: dict[str, dict[str, np.ndarray]],
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    rule: tuple[str, float | int],
    observed: dict,
    count: int,
) -> dict:
    generator = np.random.default_rng(20260829)
    labels = np.asarray(relations)
    null_mrr = np.empty(count)
    null_both = np.empty(count)
    for index in range(count):
        shuffled = tuple(generator.permutation(labels).tolist())
        summary = summarize_method(
            edge_matrices, shuffled, relation_order, rule
        )
        null_mrr[index] = summary[
            "mean_reciprocal_rank_over_16_state_position_tests"
        ]
        null_both[index] = summary["state_fraction_correct_at_both_positions"]

    def upper_p(null: np.ndarray, value: float) -> float:
        return float((1 + np.sum(null >= value)) / (1 + len(null)))

    return {
        "count": count,
        "scheme": (
            "one edge-family label permutation is applied jointly to P1, P2, "
            "and all DS1--DS8 records"
        ),
        "cross_position_mrr_one_sided_p": upper_p(
            null_mrr,
            observed["mean_reciprocal_rank_over_16_state_position_tests"],
        ),
        "both_position_accuracy_one_sided_p": upper_p(
            null_both, observed["state_fraction_correct_at_both_positions"]
        ),
        "mrr_null_05_50_95": np.quantile(
            null_mrr, [0.05, 0.5, 0.95]
        ).tolist(),
        "both_position_accuracy_null_05_50_95": np.quantile(
            null_both, [0.05, 0.5, 0.95]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_HDF5)
    parser.add_argument("--noise-features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    noise = np.load(args.noise_features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    selected_sensor_indices = noise["selected_sensor_indices"]
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    relation_order = tuple(sorted(set(graph.relations)))

    features = extract_sweep_features(
        args.source,
        selected_sensor_indices,
        PRIMARY_NPERSEG,
        PRIMARY_WINDOWS_SECONDS,
        recordings_by_position=ALL_SWEEP_RECORDINGS,
    )
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, **features)

    noise_reference = noise["mode_shapes"][noise["recording_ids"] == 0]
    noise_selected = twofold_tune(graph, noise_reference)["selected"]
    matrices = edge_score_matrices(features, graph, noise_selected)

    result = {
        "status": "exploratory_cross_excitation_structural_subsystem_localization",
        "target_scope": (
            "documented structural subsystem represented by a graph-edge family, "
            "not exact uninstrumented-joint localization"
        ),
        "expected_relation_by_state": EXPECTED_RELATION,
        "candidate_relations": relation_order,
        "protocol": {
            "positions": ["P1", "P2"],
            "healthy_records_per_position": 2,
            "damage_records_per_state_and_position": 1,
            "passes_per_record": 2,
            "primary_family_aggregation": (
                "mean of largest 20 percent of edge scores in each family"
            ),
            "raw_source_read_in_place": True,
            "block_warning": (
                "up/down passages are repeated estimates, not independent damage cases"
            ),
        },
        "feature_quality": {
            "fit_count": int(len(features["mode_shapes"])),
            "optimizer_success_fraction": float(
                np.mean(features["optimizer_success"])
            ),
            "median_coherence": float(np.median(features["median_coherence"])),
            "minimum_coherent_fraction": float(
                np.min(features["coherent_fraction"])
            ),
            "weighted_nmse_median_min_max": [
                float(np.median(features["weighted_nmse"])),
                float(np.min(features["weighted_nmse"])),
                float(np.max(features["weighted_nmse"])),
            ],
        },
        "healthy_only_graph_hyperparameters": noise_selected,
        "methods": {},
        "raw_source_integrity": {
            "path": str(args.source.resolve()),
            "size_bytes": int(features["source_size_bytes"]),
            "mtime_ns": int(features["source_mtime_ns"]),
            "copied_into_project": False,
        },
    }

    for method, edge_matrices in matrices.items():
        primary = summarize_method(
            edge_matrices,
            graph.relations,
            relation_order,
            AGGREGATORS["primary_top20_mean"],
        )
        sensitivity = {
            name: summarize_method(
                edge_matrices, graph.relations, relation_order, rule
            )
            for name, rule in AGGREGATORS.items()
        }
        result["methods"][method] = {
            "primary": primary,
            "aggregator_sensitivity": sensitivity,
            "permutation": permutation_test(
                edge_matrices,
                graph.relations,
                relation_order,
                AGGREGATORS["primary_top20_mean"],
                primary,
                args.permutations,
            ),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 6.5), sharex=True, sharey=True)
    for row, method in enumerate(("endpoint_contrast", "graph_propagated")):
        primary = result["methods"][method]["primary"]
        for column, position in enumerate(("P1", "P2")):
            summary = primary["position_results"][position]["states"]
            scores = np.asarray(
                [
                    [
                        summary[state]["aggregate_relation_scores"][relation]
                        for relation in relation_order
                    ]
                    for state in EXPECTED_RELATION
                ]
            ).T
            image = axes[row, column].imshow(scores, aspect="auto", cmap="viridis")
            for state_index, state in enumerate(EXPECTED_RELATION):
                target = relation_order.index(EXPECTED_RELATION[state])
                axes[row, column].scatter(
                    state_index, target, marker="*", s=75, color="white", edgecolor="black"
                )
            axes[row, column].set_title(f"{method.replace('_', ' ')} -- {position}")
            axes[row, column].set_yticks(
                np.arange(len(relation_order)),
                [relation.replace("_", " ") for relation in relation_order],
            )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    for axis in axes[-1]:
        axis.set_xticks(np.arange(len(EXPECTED_RELATION)), list(EXPECTED_RELATION))
        axis.set_xlabel("Damage state (star: documented subsystem)")
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_cache}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
