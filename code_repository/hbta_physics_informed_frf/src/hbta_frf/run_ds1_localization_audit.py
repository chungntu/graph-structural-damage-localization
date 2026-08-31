"""Locked, DS1-specific audit of physics-informed graph localization.

The known DS1 edge is used only after healthy-only fitting to evaluate its
locked rank among every candidate edge.  Graph-label permutations operate on the
whole DS1 record, so the fourteen blocks are never treated as independent
replicates for the primary p-value.
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
    shuffled_graph,
)
from .run_graph_damage_gp_conformal import fit_full, twofold_tune
from .run_graph_damage_gp_experiment import auc_pairwise


GROUND_TRUTH_EDGE = ("AL14", "AL18")
MODE_SUBSETS = (
    (0,),
    (1,),
    (2,),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1, 2),
)


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """Best ranks (one is best), with exact ties receiving the same rank."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Expected a block-by-edge score matrix")
    return 1 + np.sum(values[:, None, :] > values[:, :, None], axis=2)


def locate_edge_index(edge_names, requested: tuple[str, str]) -> int:
    target = frozenset(requested)
    matches = [
        index
        for index, edge in enumerate(edge_names)
        if frozenset(edge) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one ground-truth edge {requested}, got {matches}")
    return matches[0]


def summarize_localization(
    matrix: np.ndarray,
    target_index: int,
    edge_names,
) -> dict:
    ranks = descending_ranks(matrix)
    target_ranks = ranks[:, target_index]
    aggregate_scores = np.median(matrix, axis=0)
    aggregate_rank = int(1 + np.sum(aggregate_scores > aggregate_scores[target_index]))
    dominant_index = int(np.argmax(aggregate_scores))
    top_edges = np.argsort(-aggregate_scores)[:10]
    return {
        "block_count": int(len(matrix)),
        "target_block_ranks": target_ranks.astype(int).tolist(),
        "target_top1_block_fraction": float(np.mean(target_ranks == 1)),
        "target_top7_block_fraction": float(np.mean(target_ranks <= 7)),
        "target_median_block_rank": float(np.median(target_ranks)),
        "target_mean_reciprocal_rank": float(np.mean(1.0 / target_ranks)),
        "target_aggregate_rank": aggregate_rank,
        "target_aggregate_score": float(aggregate_scores[target_index]),
        "dominant_edge_index": dominant_index,
        "dominant_edge": list(edge_names[dominant_index]),
        "top_edges_by_median_score": [
            {
                "rank": rank,
                "edge_index": int(index),
                "edge": list(edge_names[index]),
                "median_score": float(aggregate_scores[index]),
            }
            for rank, index in enumerate(top_edges, start=1)
        ],
    }


def fit_and_matrix(graph, reference, test, selected):
    template, training_residuals, model = fit_full(graph, reference, selected)
    residuals = aligned_residuals(test, template)
    return (
        damage_alignment_matrix(residuals, graph, model),
        endpoint_contrast_alignment_matrix(residuals, graph),
        template,
        training_residuals,
        model,
    )


def mode_subset_audit(
    graph, reference, external, ds1, selected, target_index
) -> list[dict]:
    runs = []
    for subset in MODE_SUBSETS:
        reference_subset = reference[:, subset, :]
        external_subset = external[:, subset, :]
        ds1_subset = ds1[:, subset, :]
        template = fit_mode_shape_template(reference_subset)
        training = aligned_residuals(reference_subset, template)
        model = fit_healthy_graph_model(
            graph,
            training,
            selected["alpha_factor"],
            selected["nugget_ratio"],
        )
        residuals = aligned_residuals(ds1_subset, template)
        external_residuals = aligned_residuals(external_subset, template)
        matrix = damage_alignment_matrix(residuals, graph, model)
        summary = summarize_localization(matrix, target_index, graph.edge_names)
        endpoint_matrix = endpoint_contrast_alignment_matrix(residuals, graph)
        endpoint_external = endpoint_contrast_alignment_matrix(
            external_residuals, graph
        )
        endpoint_summary = summarize_localization(
            endpoint_matrix, target_index, graph.edge_names
        )
        endpoint_negative = summarize_localization(
            endpoint_external, target_index, graph.edge_names
        )
        runs.append(
            {
                "mode_indices_zero_based": list(subset),
                "mode_count": len(subset),
                "target_aggregate_rank": summary["target_aggregate_rank"],
                "target_median_block_rank": summary["target_median_block_rank"],
                "target_mean_reciprocal_rank": summary[
                    "target_mean_reciprocal_rank"
                ],
                "target_top1_block_fraction": summary[
                    "target_top1_block_fraction"
                ],
                "endpoint_target_aggregate_rank": endpoint_summary[
                    "target_aggregate_rank"
                ],
                "endpoint_target_top1_block_fraction": endpoint_summary[
                    "target_top1_block_fraction"
                ],
                "endpoint_external_healthy_aggregate_rank": endpoint_negative[
                    "target_aggregate_rank"
                ],
                "endpoint_target_ds1_vs_external_healthy_auc": auc_pairwise(
                    endpoint_external[:, target_index],
                    endpoint_matrix[:, target_index],
                ),
            }
        )
    return runs


def leave_one_block_out_audit(matrix, target_index, edge_names) -> list[dict]:
    runs = []
    for omitted in range(len(matrix)):
        summary = summarize_localization(
            np.delete(matrix, omitted, axis=0), target_index, edge_names
        )
        runs.append(
            {
                "omitted_block": omitted,
                "target_aggregate_rank": summary["target_aggregate_rank"],
                "target_mean_reciprocal_rank": summary[
                    "target_mean_reciprocal_rank"
                ],
                "target_top1_block_fraction": summary[
                    "target_top1_block_fraction"
                ],
            }
        )
    return runs


def grouped_decision_audit(matrix, target_index, edge_names) -> list[dict]:
    """Sensitivity to aggregating consecutive 60-second decisions."""
    runs = []
    for group_size in (1, 2, 7, 14):
        grouped = np.asarray(
            [
                np.mean(matrix[start : start + group_size], axis=0)
                for start in range(0, len(matrix), group_size)
                if start + group_size <= len(matrix)
            ]
        )
        summary = summarize_localization(grouped, target_index, edge_names)
        runs.append(
            {
                "consecutive_blocks_per_decision": group_size,
                "seconds_per_decision": 60 * group_size,
                "decision_count": len(grouped),
                "target_aggregate_rank": summary["target_aggregate_rank"],
                "target_median_decision_rank": summary[
                    "target_median_block_rank"
                ],
                "target_top1_decision_fraction": summary[
                    "target_top1_block_fraction"
                ],
            }
        )
    return runs


def empirical_upper_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def empirical_lower_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null <= observed)) / (1 + len(null)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--shuffle-count", type=int, default=1000)
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    shapes = features["mode_shapes"].astype(np.complex128)
    recording_ids = features["recording_ids"]
    states = features["states"].astype(str)
    reference = shapes[recording_ids == 0]
    external = shapes[recording_ids == 1]
    ds1 = shapes[states == "DS1"]
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    target_index = locate_edge_index(graph.edge_names, GROUND_TRUTH_EDGE)

    tuning = twofold_tune(graph, reference)
    selected = tuning["selected"]
    physical_matrix, endpoint_matrix, template, training, model = fit_and_matrix(
        graph, reference, ds1, selected
    )
    external_residuals = aligned_residuals(external, template)
    external_matrix = damage_alignment_matrix(external_residuals, graph, model)
    external_endpoint_matrix = endpoint_contrast_alignment_matrix(
        external_residuals, graph
    )

    physical = summarize_localization(
        physical_matrix, target_index, graph.edge_names
    )
    endpoint = summarize_localization(
        endpoint_matrix, target_index, graph.edge_names
    )
    negative = summarize_localization(
        external_matrix, target_index, graph.edge_names
    )
    endpoint_negative = summarize_localization(
        external_endpoint_matrix, target_index, graph.edge_names
    )

    shuffled_runs = []
    for seed in range(args.shuffle_count):
        control = shuffled_graph(graph, seed)
        control_selected = twofold_tune(control, reference)["selected"]
        control_matrix, control_endpoint_matrix, *_ = fit_and_matrix(
            control, reference, ds1, control_selected
        )
        summary = summarize_localization(
            control_matrix, target_index, control.edge_names
        )
        endpoint_summary = summarize_localization(
            control_endpoint_matrix, target_index, control.edge_names
        )
        shuffled_runs.append(
            {
                "seed": seed,
                "target_aggregate_rank": summary["target_aggregate_rank"],
                "target_median_block_rank": summary[
                    "target_median_block_rank"
                ],
                "target_mean_reciprocal_rank": summary[
                    "target_mean_reciprocal_rank"
                ],
                "target_top1_block_fraction": summary[
                    "target_top1_block_fraction"
                ],
                "endpoint_target_aggregate_rank": endpoint_summary[
                    "target_aggregate_rank"
                ],
                "endpoint_target_mean_reciprocal_rank": endpoint_summary[
                    "target_mean_reciprocal_rank"
                ],
                "endpoint_target_top1_block_fraction": endpoint_summary[
                    "target_top1_block_fraction"
                ],
                "selected_alpha_factor": control_selected["alpha_factor"],
                "selected_nugget_ratio": control_selected["nugget_ratio"],
            }
        )
        if (seed + 1) % 50 == 0:
            print(
                f"completed {seed + 1}/{args.shuffle_count} DS1 graph permutations",
                flush=True,
            )

    null_mrr = np.asarray(
        [item["target_mean_reciprocal_rank"] for item in shuffled_runs]
    )
    null_aggregate_rank = np.asarray(
        [item["target_aggregate_rank"] for item in shuffled_runs]
    )
    null_median_rank = np.asarray(
        [item["target_median_block_rank"] for item in shuffled_runs]
    )
    null_top1 = np.asarray(
        [item["target_top1_block_fraction"] for item in shuffled_runs]
    )
    endpoint_null_mrr = np.asarray(
        [item["endpoint_target_mean_reciprocal_rank"] for item in shuffled_runs]
    )
    endpoint_null_aggregate_rank = np.asarray(
        [item["endpoint_target_aggregate_rank"] for item in shuffled_runs]
    )
    endpoint_null_top1 = np.asarray(
        [item["endpoint_target_top1_block_fraction"] for item in shuffled_runs]
    )
    permutation = {
        "count": args.shuffle_count,
        "scheme": (
            "permute sensor labels within AG/AL families; preserve graph topology, "
            "edge weights, relation rows, degree sequence, and the complete DS1 record"
        ),
        "primary_statistic": "record-level mean reciprocal rank of AL14-AL18",
        "primary_one_sided_p_value": empirical_upper_p(
            null_mrr, physical["target_mean_reciprocal_rank"]
        ),
        "aggregate_rank_one_sided_p_value": empirical_lower_p(
            null_aggregate_rank, physical["target_aggregate_rank"]
        ),
        "median_block_rank_one_sided_p_value": empirical_lower_p(
            null_median_rank, physical["target_median_block_rank"]
        ),
        "top1_fraction_one_sided_p_value": empirical_upper_p(
            null_top1, physical["target_top1_block_fraction"]
        ),
        "endpoint_baseline_primary_one_sided_p_value": empirical_upper_p(
            endpoint_null_mrr, endpoint["target_mean_reciprocal_rank"]
        ),
        "endpoint_baseline_aggregate_rank_one_sided_p_value": empirical_lower_p(
            endpoint_null_aggregate_rank, endpoint["target_aggregate_rank"]
        ),
        "endpoint_baseline_top1_fraction_one_sided_p_value": empirical_upper_p(
            endpoint_null_top1, endpoint["target_top1_block_fraction"]
        ),
        "null_quantiles": {
            "mean_reciprocal_rank_05_50_95": np.quantile(
                null_mrr, [0.05, 0.5, 0.95]
            ).tolist(),
            "aggregate_rank_05_50_95": np.quantile(
                null_aggregate_rank, [0.05, 0.5, 0.95]
            ).tolist(),
            "top1_fraction_05_50_95": np.quantile(
                null_top1, [0.05, 0.5, 0.95]
            ).tolist(),
            "endpoint_mean_reciprocal_rank_05_50_95": np.quantile(
                endpoint_null_mrr, [0.05, 0.5, 0.95]
            ).tolist(),
            "endpoint_aggregate_rank_05_50_95": np.quantile(
                endpoint_null_aggregate_rank, [0.05, 0.5, 0.95]
            ).tolist(),
        },
        "runs": shuffled_runs,
    }

    leave_one_out = leave_one_block_out_audit(
        physical_matrix, target_index, graph.edge_names
    )
    endpoint_leave_one_out = leave_one_block_out_audit(
        endpoint_matrix, target_index, graph.edge_names
    )
    mode_subsets = mode_subset_audit(
        graph, reference, external, ds1, selected, target_index
    )
    grouped = grouped_decision_audit(
        physical_matrix, target_index, graph.edge_names
    )

    top_five_percent_rank = int(np.ceil(0.05 * len(graph.edge_names)))
    gates = {
        "blind_aggregate_edge_is_ground_truth": bool(
            physical["target_aggregate_rank"] == 1
        ),
        "aggregate_rank_within_top_5_percent": bool(
            physical["target_aggregate_rank"] <= top_five_percent_rank
        ),
        "primary_graph_permutation_p_le_0_05": bool(
            permutation["primary_one_sided_p_value"] <= 0.05
        ),
        "physics_mrr_exceeds_endpoint_baseline": bool(
            physical["target_mean_reciprocal_rank"]
            > endpoint["target_mean_reciprocal_rank"]
        ),
        "external_healthy_target_not_aggregate_top7": bool(
            negative["target_aggregate_rank"] > top_five_percent_rank
        ),
        "external_healthy_target_top1_fraction_le_1_over_14": bool(
            negative["target_top1_block_fraction"] <= 1.0 / 14.0
        ),
        "all_leave_one_block_out_aggregate_ranks_are_one": bool(
            all(item["target_aggregate_rank"] == 1 for item in leave_one_out)
        ),
        "at_least_five_of_seven_mode_subsets_are_top7": bool(
            sum(
                item["target_aggregate_rank"] <= top_five_percent_rank
                for item in mode_subsets
            )
            >= 5
        ),
    }
    gates["overall_locked_ds1_go"] = bool(all(gates.values()))

    endpoint_followup_gates = {
        "blind_aggregate_edge_is_ground_truth": bool(
            endpoint["target_aggregate_rank"] == 1
        ),
        "all_ds1_blocks_rank_ground_truth_first": bool(
            endpoint["target_top1_block_fraction"] == 1.0
        ),
        "graph_label_permutation_p_le_0_05": bool(
            permutation["endpoint_baseline_primary_one_sided_p_value"] <= 0.05
        ),
        "external_healthy_does_not_rank_ground_truth_first": bool(
            endpoint_negative["target_aggregate_rank"] > 1
        ),
        "target_edge_ds1_vs_external_healthy_auc_ge_0_95": bool(
            auc_pairwise(
                external_endpoint_matrix[:, target_index],
                endpoint_matrix[:, target_index],
            )
            >= 0.95
        ),
        "all_leave_one_block_out_aggregate_ranks_are_one": bool(
            all(
                item["target_aggregate_rank"] == 1
                for item in endpoint_leave_one_out
            )
        ),
    }
    endpoint_followup_gates["overall_simple_edge_contrast_go"] = bool(
        all(endpoint_followup_gates.values())
    )

    result = {
        "status": "locked_ds1_single_case_localization_audit",
        "interpretation_scope": (
            "Tests topology-specific localization information for the single DS1 "
            "record; it does not test generalization to unseen damage locations."
        ),
        "selection_warning": (
            "DS1 was selected after the broader DS1-DS8 exploratory analysis because "
            "it is the only state with a unique represented ground-truth edge."
        ),
        "protocol": {
            "healthy_fit_and_tuning": "all fourteen UDS01 blocks only",
            "external_negative_control": "all fourteen UDS02 blocks, unseen in fitting",
            "test_record": "all fourteen DS1 blocks, unseen in fitting and tuning",
            "candidate_edge_count": len(graph.edge_names),
            "ground_truth_edge": list(GROUND_TRUTH_EDGE),
            "ground_truth_edge_index": target_index,
            "primary_test": (
                "one record-level mean-reciprocal-rank statistic compared against "
                "whole-record graph-label permutations"
            ),
            "independence_claim": (
                "blocks are repeated estimates within one record, not independent "
                "damage experiments"
            ),
        },
        "selected_healthy_only_hyperparameters": selected,
        "physical_rank_one_graph": physical,
        "simple_endpoint_contrast_baseline": endpoint,
        "external_healthy_negative_control": negative,
        "endpoint_external_healthy_negative_control": endpoint_negative,
        "target_edge_ds1_vs_external_healthy_auc": {
            "rank_one_resolvent": auc_pairwise(
                external_matrix[:, target_index],
                physical_matrix[:, target_index],
            ),
            "endpoint_contrast": auc_pairwise(
                external_endpoint_matrix[:, target_index],
                endpoint_matrix[:, target_index],
            ),
        },
        "graph_label_permutation_control": permutation,
        "leave_one_ds1_block_out": leave_one_out,
        "endpoint_leave_one_ds1_block_out": endpoint_leave_one_out,
        "modal_subset_sensitivity": {
            "note": (
                "The cache contains three fitted modal bands; subsets test modal-band "
                "sensitivity, not re-estimation from different raw frequency windows."
            ),
            "runs": mode_subsets,
        },
        "consecutive_decision_aggregation_sensitivity": {
            "note": (
                "Aggregates existing 60-second decisions; it is not a raw-data "
                "block-length re-extraction audit."
            ),
            "runs": grouped,
        },
        "locked_go_no_go": gates,
        "simple_edge_contrast_followup_go_no_go": endpoint_followup_gates,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    aggregate = np.median(physical_matrix, axis=0)
    order = np.argsort(-aggregate)
    ordered_target = int(np.where(order == target_index)[0][0])
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    axes[0].plot(np.arange(1, len(order) + 1), aggregate[order], color="#345995")
    axes[0].scatter(
        [ordered_target + 1],
        [aggregate[target_index]],
        color="#d1495b",
        zorder=3,
        label="AL14-AL18",
    )
    axes[0].set_xlabel("Candidate connection rank")
    axes[0].set_ylabel("Median relative-motion score")
    axes[0].set_title("Locked ranking of 122 connections")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(null_mrr, bins=30, color="#9bb7d4", edgecolor="white")
    axes[1].axvline(
        physical["target_mean_reciprocal_rank"],
        color="#d1495b",
        linewidth=2,
        label="physical graph",
    )
    axes[1].set_xlabel("DS1 record mean reciprocal rank")
    axes[1].set_ylabel("Permuted graphs")
    axes[1].set_title(
        f"Graph-label null, p={permutation['primary_one_sided_p_value']:.4f}"
    )
    axes[1].legend()

    positions = np.arange(1, len(physical["target_block_ranks"]) + 1)
    axes[2].plot(
        positions,
        physical["target_block_ranks"],
        marker="o",
        label="DS1",
    )
    axes[2].plot(
        positions,
        negative["target_block_ranks"],
        marker="s",
        label="healthy UDS02",
    )
    axes[2].axhline(top_five_percent_rank, color="black", linestyle="--", linewidth=1)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("60-second block")
    axes[2].set_ylabel("Rank of AL14-AL18 (1 is best)")
    axes[2].set_title("Damage versus negative control")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)

    print(json.dumps(gates, indent=2))
    print(f"wrote {args.output_json} and {args.output_figure}")


if __name__ == "__main__":
    main()
