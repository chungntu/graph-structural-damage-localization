"""Exploratory multi-edge localization by first-order modal consistency.

For every measured graph edge, first-order eigenvector perturbation predicts a
spatial change direction from the healthy measured modes.  A structural family
is allowed to activate up to three non-negative edge amplitudes.  The primary
score is out-of-mode prediction: amplitudes are fitted on two modes and tested
on the third, then averaged over the three held-out modes.

The experiment is intentionally labelled exploratory because Euclidean-normalized
measured output modes are used in place of mass-normalized complete modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from .graph_damage_gp import (
    align_mode_shapes,
    fit_mode_shape_template,
    load_physical_graph,
    normalize_mode_shapes,
)
from .run_multiedge_region_experiment import (
    EXPECTED_RELATION,
    TYPE_GROUPS,
    summarize_states,
)


def real_unit_modes(shapes: np.ndarray, complex_template: np.ndarray) -> np.ndarray:
    """Align measured complex modes, retain their real structural directions."""
    aligned = align_mode_shapes(shapes, complex_template)
    real = np.real(aligned)
    norms = np.linalg.norm(real, axis=-1, keepdims=True)
    return real / np.maximum(norms, np.finfo(float).eps)


def perturbation_fields(
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    incidence: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return first-order fields [edge, mode, sensor] for unit stiffness loss."""
    edge_contrast = incidence @ modes.T  # [edge, mode]
    fields = np.zeros((len(incidence), len(modes), modes.shape[1]), dtype=float)
    for edge_index in range(len(incidence)):
        for mode_index in range(len(modes)):
            for other_index in range(len(modes)):
                if other_index == mode_index:
                    continue
                coefficient = (
                    weights[edge_index]
                    * edge_contrast[edge_index, other_index]
                    * edge_contrast[edge_index, mode_index]
                    / (eigenvalues[other_index] - eigenvalues[mode_index])
                )
                fields[edge_index, mode_index] += coefficient * modes[other_index]
    return fields


def greedy_nonnegative_fit(
    design: np.ndarray,
    target: np.ndarray,
    candidates: np.ndarray,
    maximum_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily select up to ``maximum_edges`` and refit non-negative amplitudes."""
    selected: list[int] = []
    coefficients = np.empty(0)
    current_error = float(target @ target)
    remaining = list(map(int, candidates))
    for _ in range(min(maximum_edges, len(remaining))):
        best = None
        for edge in remaining:
            trial = selected + [edge]
            values, error_norm = nnls(design[:, trial], target)
            error = float(error_norm**2)
            if best is None or error < best[0]:
                best = (error, edge, values)
        if best is None or best[0] >= current_error - 1e-12:
            break
        current_error, edge, coefficients = best
        selected.append(edge)
        remaining.remove(edge)
    return np.asarray(selected, dtype=int), coefficients


def family_scores(
    residual: np.ndarray,
    fields: np.ndarray,
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    maximum_edges: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, ...]]]]:
    """Compute in-sample and leave-one-mode-out family consistency scores."""
    mode_count, sensor_count = residual.shape
    all_design = fields.transpose(1, 2, 0)  # [mode, sensor, edge]
    relation_array = np.asarray(relations)
    in_sample = np.empty(len(relation_order))
    cross_mode = np.empty(len(relation_order))
    selected_by_relation: list[list[tuple[int, ...]]] = []

    flat_target = residual.reshape(-1)
    flat_design = all_design.reshape(mode_count * sensor_count, -1)
    total = max(float(flat_target @ flat_target), np.finfo(float).eps)

    for relation_index, relation in enumerate(relation_order):
        candidates = np.flatnonzero(relation_array == relation)
        selected, coefficients = greedy_nonnegative_fit(
            flat_design, flat_target, candidates, maximum_edges
        )
        prediction = (
            flat_design[:, selected] @ coefficients
            if len(selected)
            else np.zeros_like(flat_target)
        )
        in_sample[relation_index] = 1.0 - float(
            np.sum((flat_target - prediction) ** 2) / total
        )

        heldout_scores = []
        fold_selections: list[tuple[int, ...]] = []
        for heldout in range(mode_count):
            training_modes = np.asarray(
                [index for index in range(mode_count) if index != heldout]
            )
            train_target = residual[training_modes].reshape(-1)
            train_design = all_design[training_modes].reshape(
                len(training_modes) * sensor_count, -1
            )
            selected, coefficients = greedy_nonnegative_fit(
                train_design, train_target, candidates, maximum_edges
            )
            test_target = residual[heldout]
            test_design = all_design[heldout]
            prediction = (
                test_design[:, selected] @ coefficients
                if len(selected)
                else np.zeros_like(test_target)
            )
            denominator = max(
                float(test_target @ test_target), np.finfo(float).eps
            )
            heldout_scores.append(
                1.0 - float(np.sum((test_target - prediction) ** 2) / denominator)
            )
            fold_selections.append(tuple(map(int, selected)))
        cross_mode[relation_index] = float(np.mean(heldout_scores))
        selected_by_relation.append(fold_selections)
    return in_sample, cross_mode, selected_by_relation


def calculate_matrices(
    residuals: dict[str, np.ndarray],
    fields: np.ndarray,
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    maximum_edges: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    in_sample: dict[str, np.ndarray] = {}
    cross_mode: dict[str, np.ndarray] = {}
    for state, blocks in residuals.items():
        within = []
        heldout = []
        for block in blocks:
            current_within, current_heldout, _ = family_scores(
                block, fields, relations, relation_order, maximum_edges
            )
            within.append(current_within)
            heldout.append(current_heldout)
        in_sample[state] = np.asarray(within)
        cross_mode[state] = np.asarray(heldout)
    return in_sample, cross_mode


def permutation_test(
    residuals: dict[str, np.ndarray],
    fields: np.ndarray,
    relations: tuple[str, ...],
    relation_order: tuple[str, ...],
    observed: dict,
    maximum_edges: int,
    count: int,
) -> dict:
    if count <= 0:
        return {
            "count": 0,
            "status": "not_run_after_cross_mode_gate_failed",
        }
    generator = np.random.default_rng(20260829)
    labels = np.asarray(relations)
    null_state_mrr = np.empty(count)
    null_type_mrr = np.empty(count)
    for permutation_index in range(count):
        shuffled = tuple(generator.permutation(labels).tolist())
        _, cross_mode = calculate_matrices(
            residuals, fields, shuffled, relation_order, maximum_edges
        )
        summary = summarize_states(cross_mode, relation_order)
        null_state_mrr[permutation_index] = summary["state_mean_reciprocal_rank"]
        null_type_mrr[permutation_index] = summary["type_mean_reciprocal_rank"]

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
    parser.add_argument("--maximum-edges", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    shapes = features["mode_shapes"].astype(np.complex128)
    frequencies = features["modal_frequencies_hz"].astype(float)
    recording_ids = features["recording_ids"]
    states = features["states"].astype(str)
    reference_mask = recording_ids == 0
    reference_shapes = shapes[reference_mask]
    template_complex = fit_mode_shape_template(reference_shapes)
    template_real = normalize_mode_shapes(np.real(template_complex))
    reference_real = real_unit_modes(reference_shapes, template_complex)
    healthy_residual = reference_real - template_real[None, :, :]
    noise_scale = np.sqrt(np.mean(healthy_residual**2, axis=(0, 2)))
    noise_scale = np.maximum(noise_scale, 1e-6)

    healthy_frequencies = np.median(frequencies[reference_mask], axis=0)
    eigenvalues = (2.0 * np.pi * healthy_frequencies) ** 2
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    relation_order = tuple(sorted(set(graph.relations)))
    fields = perturbation_fields(
        template_real, eigenvalues, graph.incidence, graph.weights
    )
    fields = fields / noise_scale[None, :, None]

    residuals = {
        state: (
            real_unit_modes(shapes[states == state], template_complex)
            - template_real[None, :, :]
        )
        / noise_scale[None, :, None]
        for state in EXPECTED_RELATION
    }
    in_sample, cross_mode = calculate_matrices(
        residuals,
        fields,
        graph.relations,
        relation_order,
        args.maximum_edges,
    )
    in_sample_summary = summarize_states(in_sample, relation_order)
    cross_mode_summary = summarize_states(cross_mode, relation_order)

    result = {
        "status": "exploratory_first_order_multiedge_physics_consistency_gate",
        "physics_model": (
            "first-order eigenvector perturbation from measured healthy modes; "
            "non-negative amplitudes on at most three edges per family"
        ),
        "primary_validation": (
            "fit edge amplitudes on two modes and predict the held-out third mode"
        ),
        "scope_warning": (
            "measured Euclidean-normalized output modes replace unavailable "
            "mass-normalized complete modes"
        ),
        "expected_relation_by_state": EXPECTED_RELATION,
        "damage_type_groups": TYPE_GROUPS,
        "candidate_relations": relation_order,
        "maximum_active_edges": args.maximum_edges,
        "healthy_modal_frequencies_hz": healthy_frequencies.tolist(),
        "healthy_residual_noise_scale_by_mode": noise_scale.tolist(),
        "in_sample_fit": in_sample_summary,
        "cross_mode_prediction": cross_mode_summary,
        "permutation": permutation_test(
            residuals,
            fields,
            graph.relations,
            relation_order,
            cross_mode_summary,
            args.maximum_edges,
            args.permutations,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
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
    for axis, (title, matrices) in zip(
        axes,
        (("Fit to all three modes", in_sample), ("Held-out-mode prediction", cross_mode)),
    ):
        values = np.asarray(
            [np.median(matrices[state], axis=0) for state in EXPECTED_RELATION]
        )
        for relation_index, relation in enumerate(relation_order):
            axis.bar(
                x + (relation_index - (len(relation_order) - 1) / 2) * width,
                values[:, relation_index],
                width,
                color=colors[relation_index],
                label=display[relation],
            )
        for state_index, state in enumerate(EXPECTED_RELATION):
            target = relation_order.index(EXPECTED_RELATION[state])
            axis.scatter(
                x[state_index] + (target - (len(relation_order) - 1) / 2) * width,
                values[state_index, target],
                marker="*",
                s=80,
                color="black",
                zorder=5,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.set_xticks(x, list(EXPECTED_RELATION), rotation=45)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Explained residual energy")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
