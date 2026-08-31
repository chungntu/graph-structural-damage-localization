"""Frozen cross-excitation validation of the DS1 modal relative-motion score.

The endpoint-contrast formula and modal centers come from the preceding P2
noise study.  This script estimates two independent swept-sine passes for P1
and P2, fits healthy templates without seeing DS1, and tests the known edge
against all physical candidates and whole-record topology-label permutations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
    RAW_HDF5,
    RESPONSE_SENSORS,
)
from .frf import estimate_h1_frf, load_vertical_recording
from .graph_damage_gp import (
    aligned_residuals,
    endpoint_contrast_alignment_matrix,
    fit_mode_shape_template,
    load_physical_graph,
    shuffled_graph,
)
from .modal_model import fit_complex_residues, fit_shared_pole_model, modal_basis
from .run_ds1_localization_audit import (
    GROUND_TRUTH_EDGE,
    empirical_upper_p,
    locate_edge_index,
    summarize_localization,
)
from .run_graph_damage_gp_experiment import auc_pairwise


RECORDINGS = {
    "P1": (
        "MVS_P1_UDS_SM_Z_01",
        "MVS_P1_UDS_SM_Z_02",
        "MVS_P1_DS1_SM_Z_01",
    ),
    "P2": (
        "MVS_P2_UDS_SM_Z_01",
        "MVS_P2_UDS_SM_Z_02",
        "MVS_P2_DS1_SM_Z_01",
    ),
}
PRIMARY_WINDOWS_SECONDS = {
    "up": (45.0, 285.0),
    "down": (380.0, 615.0),
}
PRIMARY_NPERSEG = 2048
OVERLAP_FRACTION = 0.75
COHERENCE_THRESHOLD = 0.5
MODEL_BAND_HZ = (5.0, 26.0)

ROBUSTNESS_VARIANTS = (
    ("short_spectral_window", 1024, PRIMARY_WINDOWS_SECONDS),
    ("long_spectral_window", 4096, PRIMARY_WINDOWS_SECONDS),
    (
        "pass_windows_inward_5s",
        PRIMARY_NPERSEG,
        {"up": (50.0, 280.0), "down": (385.0, 610.0)},
    ),
    (
        "pass_windows_outward_5s",
        PRIMARY_NPERSEG,
        {"up": (40.0, 290.0), "down": (375.0, 620.0)},
    ),
)


def extract_sweep_features(
    source: Path,
    selected_sensor_indices: np.ndarray,
    nperseg: int,
    windows_seconds: dict[str, tuple[float, float]],
    recordings_by_position: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    """Read sweep records in place and return derived modal-residue shapes."""
    if recordings_by_position is None:
        recordings_by_position = RECORDINGS
    source_before = source.stat()
    shapes = []
    modal_frequencies = []
    damping_ratios = []
    fit_nmse = []
    optimizer_success = []
    median_coherence = []
    coherent_fraction = []
    sample_positions = []
    sample_recordings = []
    sample_states = []
    sample_directions = []

    for position, recordings in recordings_by_position.items():
        for recording in recordings:
            response, shaker, sample_rate = load_vertical_recording(
                source, recording, RESPONSE_SENSORS
            )
            for direction, (start_seconds, stop_seconds) in windows_seconds.items():
                start = int(round(start_seconds * sample_rate))
                stop = min(int(round(stop_seconds * sample_rate)), len(shaker))
                if stop <= start:
                    raise ValueError(f"Empty sweep pass {recording} {direction}")
                frequencies, frf, coherence, _ = estimate_h1_frf(
                    response[None, :, start:stop],
                    shaker[None, start:stop],
                    sample_rate,
                    nperseg,
                    OVERLAP_FRACTION,
                    1.0,
                    30.0,
                )
                band = (
                    (frequencies >= MODEL_BAND_HZ[0])
                    & (frequencies <= MODEL_BAND_HZ[1])
                )
                band_frequencies = frequencies[band]
                # Apply the frequency mask after selecting the block so NumPy
                # keeps the intended [sensor, frequency] axis order.
                band_frf = frf[0][:, band]
                band_coherence = coherence[0][:, band]
                selected_weights = np.where(
                    band_coherence[selected_sensor_indices] >= COHERENCE_THRESHOLD,
                    band_coherence[selected_sensor_indices],
                    0.0,
                )
                fit = fit_shared_pole_model(
                    band_frequencies,
                    band_frf[selected_sensor_indices],
                    selected_weights,
                    INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
                )
                basis = modal_basis(
                    band_frequencies,
                    fit.modal_frequencies_hz,
                    fit.damping_ratios,
                )
                full_weights = np.where(
                    band_coherence >= COHERENCE_THRESHOLD,
                    band_coherence,
                    0.0,
                )
                residues, _ = fit_complex_residues(
                    basis, band_frf, full_weights
                )
                mode_count = len(fit.modal_frequencies_hz)
                shapes.append(residues[:, :mode_count].T)
                modal_frequencies.append(fit.modal_frequencies_hz)
                damping_ratios.append(fit.damping_ratios)
                fit_nmse.append(fit.weighted_nmse)
                optimizer_success.append(fit.optimizer_success)
                median_coherence.append(float(np.median(band_coherence)))
                coherent_fraction.append(
                    float(np.mean(band_coherence >= COHERENCE_THRESHOLD))
                )
                sample_positions.append(position)
                sample_recordings.append(recording)
                if "_UDS_" in recording:
                    state = "UDS"
                else:
                    state = next(
                        label for label in (f"DS{i}" for i in range(1, 9))
                        if f"_{label}_" in recording
                    )
                sample_states.append(state)
                sample_directions.append(direction)
                print(
                    f"{recording} {direction}: modes "
                    f"{np.round(fit.modal_frequencies_hz, 3).tolist()}, "
                    f"NMSE={fit.weighted_nmse:.4f}",
                    flush=True,
                )

    source_after = source.stat()
    if (
        source_before.st_size != source_after.st_size
        or source_before.st_mtime_ns != source_after.st_mtime_ns
    ):
        raise RuntimeError("Raw HDF5 source changed during sweep extraction")

    return {
        "mode_shapes": np.asarray(shapes),
        "modal_frequencies_hz": np.asarray(modal_frequencies),
        "damping_ratios": np.asarray(damping_ratios),
        "weighted_nmse": np.asarray(fit_nmse),
        "optimizer_success": np.asarray(optimizer_success),
        "median_coherence": np.asarray(median_coherence),
        "coherent_fraction": np.asarray(coherent_fraction),
        "positions": np.asarray(sample_positions),
        "recordings": np.asarray(sample_recordings),
        "states": np.asarray(sample_states),
        "sweep_directions": np.asarray(sample_directions),
        "response_sensor_names": np.asarray(RESPONSE_SENSORS),
        "selected_sensor_indices": np.asarray(selected_sensor_indices),
        "nperseg": np.asarray(nperseg),
        "window_names": np.asarray(tuple(windows_seconds)),
        "window_start_stop_seconds": np.asarray(tuple(windows_seconds.values())),
        "source_path": np.asarray(str(source.resolve())),
        "source_size_bytes": np.asarray(source_before.st_size, dtype=np.int64),
        "source_mtime_ns": np.asarray(source_before.st_mtime_ns, dtype=np.int64),
    }


def leave_one_recording_out_healthy_matrix(
    healthy_shapes: np.ndarray,
    healthy_recordings: np.ndarray,
    graph,
) -> np.ndarray:
    """External-style healthy residuals, holding out both passes of a record."""
    matrices = []
    for recording in np.unique(healthy_recordings):
        heldout = healthy_recordings == recording
        template = fit_mode_shape_template(healthy_shapes[~heldout])
        residuals = aligned_residuals(healthy_shapes[heldout], template)
        matrices.append(endpoint_contrast_alignment_matrix(residuals, graph))
    return np.concatenate(matrices)


def validate_position(
    features: dict,
    position: str,
    graph,
    target_index: int,
    shuffle_count: int,
) -> dict:
    positions = features["positions"].astype(str)
    states = features["states"].astype(str)
    recordings = features["recordings"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    current = positions == position
    healthy = current & (states == "UDS")
    damaged = current & (states == "DS1")

    healthy_shapes = shapes[healthy]
    damaged_shapes = shapes[damaged]
    template = fit_mode_shape_template(healthy_shapes)
    damaged_residuals = aligned_residuals(damaged_shapes, template)
    damaged_matrix = endpoint_contrast_alignment_matrix(damaged_residuals, graph)
    healthy_matrix = leave_one_recording_out_healthy_matrix(
        healthy_shapes, recordings[healthy], graph
    )
    damaged_summary = summarize_localization(
        damaged_matrix, target_index, graph.edge_names
    )
    healthy_summary = summarize_localization(
        healthy_matrix, target_index, graph.edge_names
    )

    null_mrr = []
    null_aggregate_rank = []
    null_top1_fraction = []
    permutation_runs = []
    for seed in range(shuffle_count):
        control = shuffled_graph(graph, seed)
        control_matrix = endpoint_contrast_alignment_matrix(
            damaged_residuals, control
        )
        summary = summarize_localization(
            control_matrix, target_index, control.edge_names
        )
        null_mrr.append(summary["target_mean_reciprocal_rank"])
        null_aggregate_rank.append(summary["target_aggregate_rank"])
        null_top1_fraction.append(summary["target_top1_block_fraction"])
        permutation_runs.append(
            {
                "seed": seed,
                "target_mean_reciprocal_rank": summary[
                    "target_mean_reciprocal_rank"
                ],
                "target_aggregate_rank": summary["target_aggregate_rank"],
                "target_top1_pass_fraction": summary[
                    "target_top1_block_fraction"
                ],
            }
        )
    null_mrr = np.asarray(null_mrr)
    null_aggregate_rank = np.asarray(null_aggregate_rank)
    null_top1_fraction = np.asarray(null_top1_fraction)
    primary_p = empirical_upper_p(
        null_mrr, damaged_summary["target_mean_reciprocal_rank"]
    )
    target_auc = auc_pairwise(
        healthy_matrix[:, target_index], damaged_matrix[:, target_index]
    )
    complete_separation = bool(
        np.min(damaged_matrix[:, target_index])
        > np.max(healthy_matrix[:, target_index])
    )
    top_five_percent_rank = int(np.ceil(0.05 * len(graph.edge_names)))
    gates = {
        "aggregate_ground_truth_rank_within_top_5_percent": bool(
            damaged_summary["target_aggregate_rank"] <= top_five_percent_rank
        ),
        "both_sweep_passes_rank_ground_truth_within_top_5_percent": bool(
            all(rank <= top_five_percent_rank for rank in damaged_summary[
                "target_block_ranks"
            ])
        ),
        "graph_label_permutation_p_le_0_05": bool(primary_p <= 0.05),
        "target_edge_score_completely_separates_ds1_from_healthy": complete_separation,
    }
    gates["position_go"] = bool(all(gates.values()))
    return {
        "position": position,
        "healthy_template_pass_count": int(len(healthy_shapes)),
        "ds1_test_pass_count": int(len(damaged_shapes)),
        "ds1_localization": damaged_summary,
        "leave_one_recording_out_healthy_control": healthy_summary,
        "target_edge_ds1_vs_healthy_auc": target_auc,
        "target_edge_complete_score_separation": complete_separation,
        "permutation_control": {
            "count": shuffle_count,
            "primary_statistic": "mean reciprocal rank across the up/down DS1 passes",
            "primary_one_sided_p_value": primary_p,
            "aggregate_rank_one_sided_p_value": float(
                (1 + np.sum(
                    null_aggregate_rank
                    <= damaged_summary["target_aggregate_rank"]
                ))
                / (1 + shuffle_count)
            ),
            "top1_fraction_one_sided_p_value": empirical_upper_p(
                null_top1_fraction,
                damaged_summary["target_top1_block_fraction"],
            ),
            "null_mrr_quantiles_05_50_95": (
                np.quantile(null_mrr, [0.05, 0.5, 0.95]).tolist()
                if shuffle_count
                else [float("nan")] * 3
            ),
            "runs": permutation_runs,
        },
        "gates": gates,
        "_damaged_matrix": damaged_matrix,
        "_healthy_matrix": healthy_matrix,
        "_null_mrr": null_mrr,
    }


def public_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def robustness_audit(
    source: Path,
    selected_sensor_indices: np.ndarray,
    graph,
    target_index: int,
) -> list[dict]:
    runs = []
    for name, nperseg, windows in ROBUSTNESS_VARIANTS:
        print(f"robustness variant: {name}", flush=True)
        features = extract_sweep_features(
            source, selected_sensor_indices, nperseg, windows
        )
        positions = []
        for position in ("P1", "P2"):
            current = validate_position(
                features, position, graph, target_index, shuffle_count=0
            )
            positions.append(
                {
                    "position": position,
                    "target_aggregate_rank": current["ds1_localization"][
                        "target_aggregate_rank"
                    ],
                    "target_pass_ranks": current["ds1_localization"][
                        "target_block_ranks"
                    ],
                    "target_edge_ds1_vs_healthy_auc": current[
                        "target_edge_ds1_vs_healthy_auc"
                    ],
                    "complete_score_separation": current[
                        "target_edge_complete_score_separation"
                    ],
                }
            )
        runs.append(
            {
                "name": name,
                "nperseg": nperseg,
                "windows_seconds": windows,
                "positions": positions,
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_HDF5)
    parser.add_argument("--noise-features", type=Path, required=True)
    parser.add_argument("--noise-audit", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--shuffle-count", type=int, default=1000)
    args = parser.parse_args()

    noise_features = np.load(args.noise_features, allow_pickle=False)
    selected_sensor_indices = noise_features["selected_sensor_indices"]
    graph = load_physical_graph(
        args.edge_list, np.asarray(RESPONSE_SENSORS)
    )
    target_index = locate_edge_index(graph.edge_names, GROUND_TRUTH_EDGE)

    primary_features = extract_sweep_features(
        args.source,
        selected_sensor_indices,
        PRIMARY_NPERSEG,
        PRIMARY_WINDOWS_SECONDS,
    )
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, **primary_features)

    validations = {
        position: validate_position(
            primary_features,
            position,
            graph,
            target_index,
            args.shuffle_count,
        )
        for position in ("P1", "P2")
    }
    robustness = robustness_audit(
        args.source, selected_sensor_indices, graph, target_index
    )
    noise_audit = json.loads(args.noise_audit.read_text(encoding="utf-8"))
    noise_p = noise_audit["graph_label_permutation_control"][
        "endpoint_baseline_primary_one_sided_p_value"
    ]
    significant_configurations = int(noise_p <= 0.05) + sum(
        validation["permutation_control"]["primary_one_sided_p_value"] <= 0.05
        for validation in validations.values()
    )
    combined_gates = {
        "both_sweep_positions_pass_locked_position_gates": bool(
            all(validation["gates"]["position_go"] for validation in validations.values())
        ),
        "at_least_two_of_noise_p1_p2_permutation_tests_significant": bool(
            significant_configurations >= 2
        ),
        "all_robustness_variants_keep_both_positions_top5_percent": bool(
            all(
                position["target_aggregate_rank"] <= 7
                for run in robustness
                for position in run["positions"]
            )
        ),
        "all_modal_optimizations_succeeded": bool(
            np.all(primary_features["optimizer_success"])
        ),
        "primary_median_coherence_ge_0_95": bool(
            np.median(primary_features["median_coherence"]) >= 0.95
        ),
    }
    combined_gates["overall_cross_excitation_paper_go"] = bool(
        all(combined_gates.values())
    )

    result = {
        "status": "frozen_ds1_cross_excitation_validation",
        "protocol": {
            "development_configuration": "P2 noise; score fixed before sweep localization",
            "validation_configurations": ["P1 vertical sine sweep", "P2 vertical sine sweep"],
            "healthy_records_per_position": 2,
            "passes_per_record": 2,
            "sweep_windows_seconds": PRIMARY_WINDOWS_SECONDS,
            "welch_nperseg": PRIMARY_NPERSEG,
            "welch_overlap_fraction": OVERLAP_FRACTION,
            "modal_centers_hz": INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
            "candidate_edge_count": len(graph.edge_names),
            "ground_truth_edge": list(GROUND_TRUTH_EDGE),
            "selected_pole_fit_sensors": noise_features[
                "selected_sensor_names"
            ].astype(str).tolist(),
            "selection_statement": (
                "Sweep windows were fixed from the measured shaker ridge and modal-fit "
                "quality before inspecting sweep localization ranks."
            ),
        },
        "feature_quality": {
            "median_coherence_across_passes": float(
                np.median(primary_features["median_coherence"])
            ),
            "minimum_coherent_fraction": float(
                np.min(primary_features["coherent_fraction"])
            ),
            "weighted_nmse_median_min_max": [
                float(np.median(primary_features["weighted_nmse"])),
                float(np.min(primary_features["weighted_nmse"])),
                float(np.max(primary_features["weighted_nmse"])),
            ],
            "modal_frequencies_hz_by_pass": primary_features[
                "modal_frequencies_hz"
            ].tolist(),
        },
        "noise_development_permutation_p": noise_p,
        "primary_validations": {
            position: public_result(validation)
            for position, validation in validations.items()
        },
        "robustness_audit": robustness,
        "combined_go_no_go": combined_gates,
        "raw_source_integrity": {
            "source_path": str(args.source.resolve()),
            "size_bytes": int(primary_features["source_size_bytes"]),
            "mtime_ns": int(primary_features["source_mtime_ns"]),
            "copied_into_workspace": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for axis, position in zip(axes[:2], ("P1", "P2")):
        validation = validations[position]
        damaged = validation["_damaged_matrix"]
        healthy = validation["_healthy_matrix"]
        axis.boxplot(
            [healthy[:, target_index], damaged[:, target_index]],
            tick_labels=["healthy LORO", "DS1 sweep"],
            showfliers=False,
        )
        ranks = validation["ds1_localization"]["target_block_ranks"]
        axis.set_title(
            f"{position}: AL14-AL18 ranks {ranks[0]}, {ranks[1]}"
        )
        axis.set_ylabel("Normalized relative-motion score")
        axis.grid(alpha=0.25)
    colors = {"P1": "#345995", "P2": "#d1495b"}
    for position in ("P1", "P2"):
        validation = validations[position]
        axes[2].hist(
            validation["_null_mrr"],
            bins=30,
            alpha=0.45,
            color=colors[position],
            label=f"{position} permutations",
        )
        axes[2].axvline(
            validation["ds1_localization"]["target_mean_reciprocal_rank"],
            color=colors[position],
            linewidth=2,
        )
    axes[2].set_xlabel("Mean reciprocal rank of AL14-AL18")
    axes[2].set_ylabel("Permuted topologies")
    axes[2].set_title("Whole-record topology controls")
    axes[2].legend()
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=180)
    plt.close(figure)

    print(json.dumps(combined_gates, indent=2))
    print(
        f"wrote {args.output_cache}, {args.output_json}, and {args.output_figure}"
    )


if __name__ == "__main__":
    main()
