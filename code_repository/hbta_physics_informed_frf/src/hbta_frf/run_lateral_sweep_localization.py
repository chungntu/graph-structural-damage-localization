"""Multi-reference lateral-mode localization for HBTA bracing damage.

The vertical three-mode view is weak for DS5--DS7, whose documented
interventions lie in the lateral bracing.  This experiment reads the dedicated
Y-direction swept-sine records in place, estimates six lateral modal-response
vectors on the 17 available AG channels, fuses P1 and P2 because structural
mode shapes are input-location invariant, and localizes longitudinal regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .config import RAW_HDF5
from .frf import estimate_h1_frf
from .graph_damage_gp import (
    aligned_residuals,
    damage_alignment_matrix,
    endpoint_contrast_alignment_matrix,
    fit_healthy_graph_model,
    fit_mode_shape_template,
    load_physical_graph,
)
from .modal_model import fit_complex_residues, fit_shared_pole_model, modal_basis
from .run_all_damage_sweep_region import ALL_SWEEP_RECORDINGS
from .run_ds1_sweep_validation import (
    COHERENCE_THRESHOLD,
    OVERLAP_FRACTION,
    PRIMARY_NPERSEG,
    PRIMARY_WINDOWS_SECONDS,
)
from .run_graph_damage_gp_conformal import twofold_tune
from .run_spatial_region_localization import (
    PRIMARY_TOLERANCE_M,
    SCAN_CENTERS_M,
    gaussian_region_matrix,
    load_edge_midpoints,
    locked_ground_truth,
    matched_errors,
    select_peaks,
)


LATERAL_SENSORS = tuple(f"AG{i:02d}" for i in range(1, 19) if i != 9)
LATERAL_MODAL_CENTERS_HZ = (3.1, 6.1, 7.25, 8.65, 12.75, 16.1)
LATERAL_MODEL_BAND_HZ = (2.0, 18.0)


def lateral_recordings() -> dict[str, tuple[str, ...]]:
    return {
        position: tuple(name.replace("_Z_", "_Y_") for name in recordings)
        for position, recordings in ALL_SWEEP_RECORDINGS.items()
    }


def load_lateral_recording(
    source: Path, recording: str
) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(source, "r") as handle:
        group = handle[recording]
        sample_rate = float(group.attrs["sample_rate"])
        acceleration = group["acceleration"]
        response = np.stack(
            [
                np.asarray(acceleration[sensor]["y"], dtype=float)
                for sensor in LATERAL_SENSORS
            ]
        )
        shaker = np.asarray(acceleration["AS"]["y"], dtype=float)
    common = min(response.shape[1], len(shaker))
    return response[:, :common], shaker[:common], sample_rate


def extract_features(source: Path) -> dict:
    before = source.stat()
    output: dict[str, list] = {
        "mode_shapes": [],
        "modal_frequencies_hz": [],
        "damping_ratios": [],
        "weighted_nmse": [],
        "optimizer_success": [],
        "median_coherence": [],
        "coherent_fraction": [],
        "positions": [],
        "recordings": [],
        "states": [],
        "sweep_directions": [],
    }
    for position, recordings in lateral_recordings().items():
        for recording in recordings:
            response, shaker, sample_rate = load_lateral_recording(
                source, recording
            )
            for direction, (start_s, stop_s) in PRIMARY_WINDOWS_SECONDS.items():
                start = int(round(start_s * sample_rate))
                stop = min(int(round(stop_s * sample_rate)), len(shaker))
                frequencies, frf, coherence, _ = estimate_h1_frf(
                    response[None, :, start:stop],
                    shaker[None, start:stop],
                    sample_rate,
                    PRIMARY_NPERSEG,
                    OVERLAP_FRACTION,
                    1.0,
                    30.0,
                )
                band = (
                    (frequencies >= LATERAL_MODEL_BAND_HZ[0])
                    & (frequencies <= LATERAL_MODEL_BAND_HZ[1])
                )
                current_frequencies = frequencies[band]
                current_frf = frf[0][:, band]
                current_coherence = coherence[0][:, band]
                weights = np.where(
                    current_coherence >= COHERENCE_THRESHOLD,
                    current_coherence,
                    0.0,
                )
                fit = fit_shared_pole_model(
                    current_frequencies,
                    current_frf,
                    weights,
                    LATERAL_MODAL_CENTERS_HZ,
                )
                basis = modal_basis(
                    current_frequencies,
                    fit.modal_frequencies_hz,
                    fit.damping_ratios,
                )
                residues, _ = fit_complex_residues(
                    basis, current_frf, weights
                )
                output["mode_shapes"].append(
                    residues[:, : len(LATERAL_MODAL_CENTERS_HZ)].T
                )
                output["modal_frequencies_hz"].append(fit.modal_frequencies_hz)
                output["damping_ratios"].append(fit.damping_ratios)
                output["weighted_nmse"].append(fit.weighted_nmse)
                output["optimizer_success"].append(fit.optimizer_success)
                output["median_coherence"].append(
                    float(np.median(current_coherence))
                )
                output["coherent_fraction"].append(
                    float(np.mean(current_coherence >= COHERENCE_THRESHOLD))
                )
                output["positions"].append(position)
                output["recordings"].append(recording)
                output["states"].append(
                    "UDS"
                    if "_UDS_" in recording
                    else next(
                        state
                        for state in (f"DS{i}" for i in range(1, 9))
                        if f"_{state}_" in recording
                    )
                )
                output["sweep_directions"].append(direction)
                print(
                    f"{recording} {direction}: "
                    f"{np.round(fit.modal_frequencies_hz, 3).tolist()}, "
                    f"NMSE={fit.weighted_nmse:.4f}",
                    flush=True,
                )
    after = source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("Raw HDF5 source changed during lateral extraction")
    return {
        **{key: np.asarray(value) for key, value in output.items()},
        "response_sensor_names": np.asarray(LATERAL_SENSORS),
        "source_path": np.asarray(str(source.resolve())),
        "source_size_bytes": np.asarray(before.st_size, dtype=np.int64),
        "source_mtime_ns": np.asarray(before.st_mtime_ns, dtype=np.int64),
    }


def joint_shapes(features: dict, state: str) -> np.ndarray:
    states = features["states"].astype(str)
    directions = features["sweep_directions"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    return np.asarray(
        [
            fit_mode_shape_template(shapes[(states == state) & (directions == direction)])
            for direction in ("up", "down")
        ]
    )


def healthy_joint_shapes(features: dict) -> np.ndarray:
    states = features["states"].astype(str)
    directions = features["sweep_directions"].astype(str)
    recordings = features["recordings"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    return np.asarray(
        [
            fit_mode_shape_template(
                shapes[
                    (states == "UDS")
                    & (directions == direction)
                    & np.char.endswith(recordings, suffix)
                ]
            )
            for suffix in ("01", "02")
            for direction in ("up", "down")
        ]
    )


def permutation_p(
    edge_scores: dict[str, np.ndarray],
    edge_x: np.ndarray,
    truth: dict[str, tuple[float, ...]],
    observed_errors: np.ndarray,
    count: int,
) -> dict:
    generator = np.random.default_rng(20260829)
    null_mean = np.empty(count)
    for index in range(count):
        matrix = gaussian_region_matrix(
            generator.permutation(edge_x), SCAN_CENTERS_M, 3.5
        )
        errors = []
        for state in ("DS5", "DS6", "DS7"):
            target = np.asarray(truth[state])
            predicted = select_peaks(
                matrix @ edge_scores[state], SCAN_CENTERS_M, len(target)
            )
            errors.extend(matched_errors(predicted, target).tolist())
        null_mean[index] = np.mean(errors)
    observed_mean = float(np.mean(observed_errors))
    return {
        "count": count,
        "mean_error_one_sided_p": float(
            (1 + np.sum(null_mean <= observed_mean)) / (1 + count)
        ),
        "mean_error_null_05_50_95": np.quantile(
            null_mean, [0.05, 0.5, 0.95]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_HDF5)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    if args.reuse_cache and args.output_cache.exists():
        cached = np.load(args.output_cache, allow_pickle=False)
        features = {key: cached[key] for key in cached.files}
        print(f"reused {args.output_cache}")
    else:
        features = extract_features(args.source)
        args.output_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_cache, **features)
    graph = load_physical_graph(
        args.edge_list, np.asarray(LATERAL_SENSORS), skip_unknown=True
    )
    healthy = healthy_joint_shapes(features)
    template = fit_mode_shape_template(healthy)
    healthy_residual = aligned_residuals(healthy, template)
    selected = twofold_tune(graph, healthy)["selected"]
    model = fit_healthy_graph_model(
        graph,
        healthy_residual,
        selected["alpha_factor"],
        selected["nugget_ratio"],
    )
    edge_x = load_edge_midpoints(args.edge_list, graph)
    region_matrix = gaussian_region_matrix(edge_x, SCAN_CENTERS_M, 3.5)
    truth = locked_ground_truth()

    result = {
        "status": "exploratory_multireference_lateral_region_localization",
        "lateral_sensors": list(LATERAL_SENSORS),
        "modal_centers_hz": list(LATERAL_MODAL_CENTERS_HZ),
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
        "healthy_only_graph_hyperparameters": selected,
        "methods": {},
        "raw_source_integrity": {
            "path": str(args.source.resolve()),
            "size_bytes": int(features["source_size_bytes"]),
            "mtime_ns": int(features["source_mtime_ns"]),
            "copied_into_project": False,
        },
    }

    damage_residuals = {
        state: aligned_residuals(joint_shapes(features, state), template)
        for state in ("DS5", "DS6", "DS7")
    }
    matrices = {
        "endpoint_contrast": {
            state: endpoint_contrast_alignment_matrix(residual, graph)
            for state, residual in damage_residuals.items()
        },
        "graph_propagated": {
            state: damage_alignment_matrix(residual, graph, model)
            for state, residual in damage_residuals.items()
        },
    }
    for method, state_matrices in matrices.items():
        edge_scores = {
            state: np.median(matrix, axis=0)
            for state, matrix in state_matrices.items()
        }
        errors = []
        states_result = {}
        for state, score in edge_scores.items():
            target = np.asarray(truth[state])
            curve = region_matrix @ score
            predicted = select_peaks(
                curve, SCAN_CENTERS_M, len(target)
            )
            current_errors = matched_errors(predicted, target)
            errors.extend(current_errors.tolist())
            states_result[state] = {
                "ground_truth_x_m": target.tolist(),
                "predicted_x_m": predicted.tolist(),
                "matched_absolute_errors_m": current_errors.tolist(),
                "all_targets_within_one_panel": bool(
                    np.all(current_errors <= PRIMARY_TOLERANCE_M)
                ),
            }
        result["methods"][method] = {
            "mean_absolute_error_m": float(np.mean(errors)),
            "target_fraction_within_one_panel": float(
                np.mean(np.asarray(errors) <= PRIMARY_TOLERANCE_M)
            ),
            "states": states_result,
            "permutation": permutation_p(
                edge_scores,
                edge_x,
                truth,
                np.asarray(errors),
                args.permutations,
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 5.7), sharex=True)
    y = np.arange(3)
    for axis, method in zip(axes, ("endpoint_contrast", "graph_propagated")):
        for index, state in enumerate(("DS5", "DS6", "DS7")):
            current = result["methods"][method]["states"][state]
            axis.scatter(
                current["ground_truth_x_m"],
                np.full(len(current["ground_truth_x_m"]), y[index]),
                marker="*",
                s=130,
                color="black",
            )
            axis.scatter(
                current["predicted_x_m"],
                np.full(len(current["predicted_x_m"]), y[index]),
                marker="o",
                s=50,
                color="#d26a3a",
            )
        axis.set_yticks(y, ["DS5", "DS6", "DS7"])
        axis.set_title(method.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Longitudinal coordinate x (m); star: damage, circle: inferred")
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output_cache}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
