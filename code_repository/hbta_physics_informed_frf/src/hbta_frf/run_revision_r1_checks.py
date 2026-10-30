"""Additional checks requested during the first SDHM revision round.

1. Scan-grid resolution sensitivity of the continuous top-K profile R(x)
   (Reviewer 2, comment 6).
2. Sensitivity of the discrete and continuous results to the modal
   normalization convention (Reviewer 1, comment 1).
3. Topology-permutation p-values for the fused (P1+P2) DS1 spectrum and a
   restatement of the per-position values under both test statistics
   (Reviewer 1, comment 2).

Nothing here changes the locked primary configuration; every result is written
to a new JSON file so the originally submitted outputs stay untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .graph_damage_gp import (
    fit_mode_shape_template,
    load_physical_graph,
    normalized_endpoint_node_energy_matrix,
    shuffled_graph,
)
from .run_ds1_localization_audit import GROUND_TRUTH_EDGE, locate_edge_index
from .run_spatial_region_localization import load_edge_midpoints, locked_ground_truth
from .run_topk_spatial_localization import (
    PRIMARY_BANDWIDTH_M,
    PRIMARY_K,
    PRIMARY_TOLERANCE_M,
)

SPAN_LIMIT_M = 14.0
GRID_STEPS_M = (0.05, 0.10, 0.25, 0.50, 1.00)
NORMALIZATIONS = ("unit_euclidean", "modal_scale_factor", "reference_component")


# ---------------------------------------------------------------------------
# Continuous profile with an explicit scan step
# ---------------------------------------------------------------------------
def scan_centers(step_m: float) -> np.ndarray:
    return np.arange(-SPAN_LIMIT_M, SPAN_LIMIT_M + step_m / 2.0, step_m)


def topk_curve(scores, edge_x, centers, bandwidth_m, k):
    order = np.argsort(-scores)[:k]
    kernel = np.exp(-0.5 * ((centers[:, None] - edge_x[order][None, :]) / bandwidth_m) ** 2)
    denom = np.maximum(kernel.sum(axis=1), np.finfo(float).eps)
    return (kernel @ scores[order]) / denom


def select_peaks(curve, centers, count, step_m, minimum_separation_m=3.5):
    """Same rule as run_spatial_region_localization.select_peaks, any step."""
    distance = max(1, int(round(minimum_separation_m / step_m)))
    candidates, _ = find_peaks(curve, distance=distance)
    candidates = list(candidates[np.argsort(-curve[candidates])])
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


def matched_errors(predicted, truth):
    cost = np.abs(predicted[:, None] - truth[None, :])
    rows, columns = linear_sum_assignment(cost)
    return cost[rows, columns]


def benchmark(scores_by_state, edge_x, truth, step_m, bandwidth_m=PRIMARY_BANDWIDTH_M, k=PRIMARY_K):
    centers = scan_centers(step_m)
    states, errors, passes = {}, [], []
    for state, values in truth.items():
        target = np.asarray(values, dtype=float)
        curve = topk_curve(scores_by_state[state], edge_x, centers, bandwidth_m, k)
        predicted = select_peaks(curve, centers, len(target), step_m)
        err = matched_errors(predicted, target)
        ok = bool(np.all(err <= PRIMARY_TOLERANCE_M))
        passes.append(ok)
        errors.extend(err.tolist())
        states[state] = {
            "predicted_x_m": [round(float(v), 4) for v in predicted],
            "errors_m": [round(float(v), 4) for v in err],
            "pass": ok,
        }
    return {
        "scan_step_m": step_m,
        "states_passed": int(np.sum(passes)),
        "state_count": len(passes),
        "mean_absolute_error_m": float(np.mean(errors)),
        "states": states,
    }


# ---------------------------------------------------------------------------
# Normalization conventions
# ---------------------------------------------------------------------------
def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), np.finfo(float).eps)


def residual_under(shape, template, convention, reference_index):
    """Residual r_m for one mode under a given scaling convention.

    ``shape`` and ``template`` are (modes, sensors) complex arrays already
    fused and unit-Euclidean normalized by the primary pipeline.
    """
    out = np.empty_like(shape)
    for m in range(shape.shape[0]):
        phi, ref = shape[m], template[m]
        if convention == "unit_euclidean":
            inner = np.vdot(ref, phi)
            out[m] = phi * np.exp(-1j * np.angle(inner)) - ref
        elif convention == "modal_scale_factor":
            # Least-squares complex scale to the baseline (classical MSF).
            alpha = np.vdot(phi, ref) / np.vdot(phi, phi)
            out[m] = alpha * phi - ref
        elif convention == "reference_component":
            # Unit modal displacement at the largest-amplitude baseline sensor.
            k = reference_index[m]
            out[m] = phi / phi[k] - ref / ref[k]
        else:
            raise ValueError(convention)
    return out


def edge_contrast_score(residual, incidence):
    """s_e of Eq. (11): mean_m |b_e^H r_m|^2 / (2 ||r_m||^2)."""
    energy = np.maximum(np.sum(np.abs(residual) ** 2, axis=1), np.finfo(float).eps)
    projection = residual @ incidence.T  # (modes, edges)
    per_mode = np.abs(projection) ** 2 / (2.0 * energy[:, None])
    return per_mode.mean(axis=0)


def rank_of(scores, index):
    return int(1 + np.sum(scores > scores[index]))


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--ds1-sweep-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    cached = np.load(args.sweep_features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    edge_x = load_edge_midpoints(args.edge_list, graph)
    truth = locked_ground_truth()
    target = locate_edge_index(graph.edge_names, GROUND_TRUTH_EDGE)

    states = cached["states"].astype(str)
    recordings = cached["recordings"].astype(str)
    directions = cached["sweep_directions"].astype(str)
    shapes = cached["mode_shapes"].astype(np.complex128)

    # Fused healthy template, exactly as make_edge_scores builds it.
    healthy_joint = np.asarray(
        [
            fit_mode_shape_template(
                shapes[(states == "UDS") & (directions == d) & np.char.endswith(recordings, s)]
            )
            for s in ("01", "02")
            for d in ("up", "down")
        ]
    )
    template = fit_mode_shape_template(healthy_joint)
    reference_index = np.argmax(np.abs(template), axis=1)
    fused = {
        state: np.asarray(
            [fit_mode_shape_template(shapes[(states == state) & (directions == d)]) for d in ("up", "down")]
        )
        for state in (f"DS{i}" for i in range(1, 9))
    }

    # Euclidean auto-MAC of the healthy template (orthogonality in R^N).
    automac = np.abs(template.conj() @ template.T) ** 2

    # ---------------- 1. Grid resolution ----------------
    primary_scores = {}
    for state, joint in fused.items():
        residual = np.asarray([residual_under(j, template, "unit_euclidean", reference_index) for j in joint])
        primary_scores[state] = np.median(normalized_endpoint_node_energy_matrix(residual, graph), axis=0)
    grid = {f"{step:.2f}": benchmark(primary_scores, edge_x, truth, step) for step in GRID_STEPS_M}

    # ---------------- 2. Normalization conventions ----------------
    normalization = {}
    for convention in NORMALIZATIONS:
        cont_scores, parallel_fraction = {}, {}
        for state, joint in fused.items():
            residual = np.asarray([residual_under(j, template, convention, reference_index) for j in joint])
            cont_scores[state] = np.median(normalized_endpoint_node_energy_matrix(residual, graph), axis=0)
            # Share of residual energy parallel to the baseline shape.
            par = np.abs(np.einsum("mn,bmn->bm", template.conj(), residual)) ** 2
            tot = np.sum(np.abs(residual) ** 2, axis=2)
            parallel_fraction[state] = float(np.median(par / tot))
        ds1_res = np.asarray([residual_under(j, template, convention, reference_index) for j in fused["DS1"]])
        ds1_scores = np.median(
            np.asarray([edge_contrast_score(r, graph.incidence) for r in ds1_res]), axis=0
        )
        cont = benchmark(cont_scores, edge_x, truth, 0.25)
        normalization[convention] = {
            "ds1_fused_target_rank": rank_of(ds1_scores, target),
            "ds1_fused_target_score": float(ds1_scores[target]),
            "ds1_fused_median_score": float(np.median(ds1_scores)),
            "continuous_states_passed": cont["states_passed"],
            "continuous_mean_absolute_error_m": cont["mean_absolute_error_m"],
            "continuous_predicted_x_m": {s: v["predicted_x_m"] for s, v in cont["states"].items()},
            "median_parallel_residual_energy_fraction": parallel_fraction,
        }

    # ---------------- 3. Fused DS1 topology permutation ----------------
    ds1_res = np.asarray([residual_under(j, template, "unit_euclidean", reference_index) for j in fused["DS1"]])
    observed = np.median(np.asarray([edge_contrast_score(r, graph.incidence) for r in ds1_res]), axis=0)
    observed_rank = rank_of(observed, target)
    null_ranks = []
    for seed in range(args.permutations):
        control = shuffled_graph(graph, seed)
        null = np.median(np.asarray([edge_contrast_score(r, control.incidence) for r in ds1_res]), axis=0)
        null_ranks.append(rank_of(null, target))
    null_ranks = np.asarray(null_ranks)
    fused_perm = {
        "observed_rank": observed_rank,
        "observed_target_score": float(observed[target]),
        "count_null_rank_le_observed": int(np.sum(null_ranks <= observed_rank)),
        "p_value_eq20": float((1 + np.sum(null_ranks <= observed_rank)) / (1 + args.permutations)),
    }

    per_position = {}
    sweep = json.loads(args.ds1_sweep_json.read_text(encoding="utf-8"))
    for position in ("P1", "P2"):
        pv = sweep["primary_validations"][position]
        pc = pv["permutation_control"]
        obs_rank = pv["ds1_localization"]["target_aggregate_rank"]
        obs_mrr = pv["ds1_localization"]["target_mean_reciprocal_rank"]
        ranks = np.asarray([r["target_aggregate_rank"] for r in pc["runs"]])
        mrr = np.asarray([r["target_mean_reciprocal_rank"] for r in pc["runs"]])
        per_position[position] = {
            "observed_aggregate_rank": obs_rank,
            "observed_pass_ranks": pv["ds1_localization"]["target_block_ranks"],
            "aggregate_rank_count_le_observed": int(np.sum(ranks <= obs_rank)),
            "aggregate_rank_p_value_eq20": float((1 + np.sum(ranks <= obs_rank)) / (1 + len(ranks))),
            "mrr_count_ge_observed": int(np.sum(mrr >= obs_mrr - 1e-12)),
            "mrr_p_value": float((1 + np.sum(mrr >= obs_mrr - 1e-12)) / (1 + len(mrr))),
            "uniform_expectation_rank_le_observed": len(ranks) * obs_rank / len(graph.edge_names),
        }

    result = {
        "status": "sdhm_revision_r1_checks",
        "primary_configuration": {
            "bandwidth_m": PRIMARY_BANDWIDTH_M,
            "k": PRIMARY_K,
            "tolerance_m": PRIMARY_TOLERANCE_M,
        },
        "healthy_template_euclidean_automac": automac.round(4).tolist(),
        "grid_resolution_sensitivity": grid,
        "normalization_sensitivity": normalization,
        "fused_ds1_permutation": fused_perm,
        "per_position_ds1_permutation": per_position,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
