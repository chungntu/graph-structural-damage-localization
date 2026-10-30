"""Top-K restricted continuous spatial localization for HBTA DS1--DS8.

Pooling the endpoint node-energy score over the full 122-edge candidate set
with a Gaussian kernel (as in ``run_multireference_spatial_localization.py``)
is unstable: a broad cluster of moderately elevated, non-damaged candidates
can outweigh the few sharply elevated candidates near the true damage and
displace the pooled peak. This module restricts pooling at every scan
coordinate to the ``K`` highest-scoring candidate connections, evaluated
under the ``normalized_endpoint_node_energy`` score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_damage_gp import load_physical_graph
from .run_graph_damage_gp_conformal import twofold_tune
from .run_multireference_spatial_localization import make_edge_scores
from .run_spatial_region_localization import (
    SCAN_CENTERS_M,
    load_edge_midpoints,
    locked_ground_truth,
    matched_errors,
    select_peaks,
)

PRIMARY_METHOD = "normalized_endpoint_node_energy"
PRIMARY_BANDWIDTH_M = 1.75
PRIMARY_K = 8
PRIMARY_TOLERANCE_M = 3.5


def topk_region_curve(
    scores: np.ndarray,
    edge_x: np.ndarray,
    centers: np.ndarray,
    bandwidth_m: float,
    k: int,
) -> np.ndarray:
    """Gaussian-kernel spatial profile restricted to the top-k scoring edges."""
    order = np.argsort(-scores)[:k]
    sub_x = edge_x[order]
    sub_scores = scores[order]
    kernel = np.exp(-0.5 * ((centers[:, None] - sub_x[None, :]) / bandwidth_m) ** 2)
    denom = np.maximum(np.sum(kernel, axis=1), np.finfo(float).eps)
    return (kernel @ sub_scores) / denom


def summarize(
    scores_by_state: dict[str, np.ndarray],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    bandwidth_m: float,
    k: int,
) -> dict:
    states = {}
    all_errors: list[float] = []
    state_pass = []
    for state, truth_values in truth.items():
        target = np.asarray(truth_values)
        curve = topk_region_curve(
            scores_by_state[state], edge_x, SCAN_CENTERS_M, bandwidth_m, k
        )
        predicted = select_peaks(curve, SCAN_CENTERS_M, len(target))
        errors = matched_errors(predicted, target)
        ok = bool(np.all(errors <= PRIMARY_TOLERANCE_M))
        state_pass.append(ok)
        all_errors.extend(errors.tolist())
        states[state] = {
            "ground_truth_x_m": target.tolist(),
            "predicted_x_m": predicted.tolist(),
            "matched_absolute_errors_m": errors.tolist(),
            "all_targets_within_one_panel": ok,
        }
    return {
        "bandwidth_m": bandwidth_m,
        "k": k,
        "mean_absolute_error_m": float(np.mean(all_errors)),
        "target_fraction_within_one_panel": float(
            np.mean(np.asarray(all_errors) <= PRIMARY_TOLERANCE_M)
        ),
        "state_fraction_within_one_panel": float(np.mean(state_pass)),
        "states": states,
    }


def permutation(
    scores_by_state: dict[str, np.ndarray],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    observed: dict,
    bandwidth_m: float,
    k: int,
    count: int,
    seed: int = 20260829,
) -> dict:
    generator = np.random.default_rng(seed)
    null_error = np.empty(count)
    null_state_frac = np.empty(count)
    for index in range(count):
        shuffled_x = generator.permutation(edge_x)
        current = summarize(scores_by_state, shuffled_x, truth, bandwidth_m, k)
        null_error[index] = current["mean_absolute_error_m"]
        null_state_frac[index] = current["state_fraction_within_one_panel"]
    return {
        "count": count,
        "mean_error_one_sided_p": float(
            (1 + np.sum(null_error <= observed["mean_absolute_error_m"]))
            / (1 + count)
        ),
        "state_fraction_one_sided_p": float(
            (
                1
                + np.sum(
                    null_state_frac >= observed["state_fraction_within_one_panel"]
                )
            )
            / (1 + count)
        ),
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
    edge_scores = make_edge_scores(features, graph, selected)
    edge_x = load_edge_midpoints(args.edge_list, graph)
    truth = locked_ground_truth()

    scores_by_state = edge_scores[PRIMARY_METHOD]
    primary = summarize(
        scores_by_state, edge_x, truth, PRIMARY_BANDWIDTH_M, PRIMARY_K
    )
    perm = permutation(
        scores_by_state,
        edge_x,
        truth,
        primary,
        PRIMARY_BANDWIDTH_M,
        PRIMARY_K,
        args.permutations,
    )
    bandwidth_sensitivity = {
        str(bandwidth): summarize(scores_by_state, edge_x, truth, bandwidth, PRIMARY_K)
        for bandwidth in (0.75, 1.0, 1.25, 1.75, 2.625, 3.5, 5.25, 7.0, 8.75, 10.5)
    }
    k_sensitivity = {
        str(k): summarize(scores_by_state, edge_x, truth, PRIMARY_BANDWIDTH_M, k)
        for k in (5, 6, 7, 8, 9, 10, 12, 15)
    }

    result = {
        "status": "topk_restricted_spatial_localization",
        "definition": (
            "Continuous spatial profile R(x) pooled with a Gaussian kernel "
            "restricted to the K highest-scoring candidate connections, "
            "scored by the normalized endpoint node-energy sigma_e, after "
            "P1/P2 multi-reference modal fusion. Introduced because pooling "
            "over the full 122-edge candidate set lets a broad, moderately "
            "elevated non-damaged cluster outweigh a sparse, sharply "
            "elevated true signal."
        ),
        "primary_method": PRIMARY_METHOD,
        "primary_bandwidth_m": PRIMARY_BANDWIDTH_M,
        "primary_k": PRIMARY_K,
        "success_tolerance_m": PRIMARY_TOLERANCE_M,
        "ground_truth_x_m": truth,
        "primary": primary,
        "permutation": perm,
        "bandwidth_sensitivity_at_primary_k": bandwidth_sensitivity,
        "k_sensitivity_at_primary_bandwidth": k_sensitivity,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    states_order = [f"DS{i}" for i in range(1, 9)]
    labels: list[str] = []
    errors: list[float] = []
    for state in states_order:
        entry = primary["states"][state]
        if len(entry["matched_absolute_errors_m"]) == 1:
            labels.append(state)
            errors.append(entry["matched_absolute_errors_m"][0])
        else:
            for site_index, error in enumerate(entry["matched_absolute_errors_m"], start=1):
                labels.append(f"{state}\n(Site {site_index})")
                errors.append(error)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(labels))
    colors = ["#2ca02c" if e <= PRIMARY_TOLERANCE_M else "#d62728" for e in errors]
    ax.bar(x, errors, 0.6, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.85)
    ax.axhline(PRIMARY_TOLERANCE_M, color="#2ca02c", linestyle="--", linewidth=1.6)
    ax.axhspan(0, PRIMARY_TOLERANCE_M, color="#2ca02c", alpha=0.08, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Localization Error (m)", fontweight="bold")
    ax.set_title(
        f"Top-K Restricted Spatial Smoothing (K={PRIMARY_K}, h={PRIMARY_BANDWIDTH_M} m)",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")
    print(
        f"states_pass={primary['state_fraction_within_one_panel']*8:.0f}/8  "
        f"targets_pass={primary['target_fraction_within_one_panel']*9:.0f}/9  "
        f"p_state={perm['state_fraction_one_sided_p']:.4f}  "
        f"p_error={perm['mean_error_one_sided_p']:.4f}"
    )


if __name__ == "__main__":
    main()
