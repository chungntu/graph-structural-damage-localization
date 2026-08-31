"""Audit healthy P2-noise shaker-to-response FRF feasibility on HBTA."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .config import (
    BLOCK_SECONDS,
    MAXIMUM_FREQUENCY_HZ,
    MINIMUM_FREQUENCY_HZ,
    PRIMARY_HEALTHY_NOISE_RECORDINGS,
    RAW_HDF5,
    RESPONSE_SENSORS,
    TRIM_SECONDS,
    WELCH_NPERSEG,
    WELCH_OVERLAP_FRACTION,
)
from .frf import estimate_h1_frf, load_vertical_recording, nonoverlapping_blocks


def _recording_frf(source: Path, name: str) -> dict:
    response, shaker, sample_rate = load_vertical_recording(
        source, name, RESPONSE_SENSORS
    )
    response_blocks = nonoverlapping_blocks(
        response, sample_rate, BLOCK_SECONDS, TRIM_SECONDS
    ).transpose(1, 0, 2)
    shaker_blocks = nonoverlapping_blocks(
        shaker, sample_rate, BLOCK_SECONDS, TRIM_SECONDS
    )
    frequencies, frf, coherence, input_psd = estimate_h1_frf(
        response_blocks,
        shaker_blocks,
        sample_rate,
        WELCH_NPERSEG,
        WELCH_OVERLAP_FRACTION,
        MINIMUM_FREQUENCY_HZ,
        MAXIMUM_FREQUENCY_HZ,
    )
    return {
        "frequencies": frequencies,
        "frf": frf,
        "coherence": coherence,
        "input_psd": input_psd,
        "sample_rate": sample_rate,
    }


def _modal_profile(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coherence = result["coherence"]
    frf = result["frf"]
    median_coherence_by_sensor = np.median(coherence, axis=0)
    sensor_quality = np.median(median_coherence_by_sensor, axis=1)
    selected_sensors = sensor_quality >= np.quantile(sensor_quality, 0.5)
    coherent = median_coherence_by_sensor[selected_sensors] >= 0.35
    log_magnitude = 20.0 * np.log10(np.abs(frf[:, selected_sensors]) + 1e-15)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        profile = np.nanmedian(
            np.where(coherent[None, ...], log_magnitude, np.nan), axis=(0, 1)
        )
    if np.isnan(profile).any():
        fallback = np.median(log_magnitude, axis=(0, 1))
        profile = np.where(np.isnan(profile), fallback, profile)
    smoothing_window = min(11, len(profile) if len(profile) % 2 else len(profile) - 1)
    smoothed_profile = savgol_filter(profile, smoothing_window, 3)
    prominence = 1.0
    frequency_step = float(np.median(np.diff(result["frequencies"])))
    minimum_distance = max(1, int(round(0.45 / frequency_step)))
    peaks, properties = find_peaks(
        smoothed_profile, prominence=prominence, distance=minimum_distance
    )
    order = np.argsort(properties["prominences"])[::-1]
    peaks = peaks[order[:12]]
    peaks = np.sort(peaks)
    return smoothed_profile, result["frequencies"][peaks], sensor_quality


def _match_peaks(first: np.ndarray, second: np.ndarray, tolerance: float = 0.25) -> list:
    matches = []
    available = set(range(len(second)))
    for value in first:
        if not available:
            break
        candidate = min(available, key=lambda index: abs(second[index] - value))
        difference = abs(second[candidate] - value)
        if difference <= tolerance:
            matches.append(
                {
                    "recording_1_hz": float(value),
                    "recording_2_hz": float(second[candidate]),
                    "absolute_difference_hz": float(difference),
                }
            )
            available.remove(candidate)
    return matches


def run(source: Path, cache_path: Path, output_json: Path, output_figure: Path) -> dict:
    source_stat_before = source.stat()
    results = {
        name: _recording_frf(source, name)
        for name in PRIMARY_HEALTHY_NOISE_RECORDINGS
    }
    source_stat_after = source.stat()
    if (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
    ):
        raise RuntimeError("Raw HDF5 source changed during FRF audit")

    first_name, second_name = PRIMARY_HEALTHY_NOISE_RECORDINGS
    first = results[first_name]
    second = results[second_name]
    np.testing.assert_allclose(first["frequencies"], second["frequencies"])
    profile_1, peaks_1, quality_1 = _modal_profile(first)
    profile_2, peaks_2, quality_2 = _modal_profile(second)
    matches = _match_peaks(peaks_1, peaks_2)

    median_coherence_1 = np.median(first["coherence"], axis=(0, 1))
    median_coherence_2 = np.median(second["coherence"], axis=(0, 1))
    per_sensor_coherence_1 = np.median(first["coherence"], axis=0)
    per_sensor_coherence_2 = np.median(second["coherence"], axis=0)
    usable_sensor_fraction_by_frequency = np.mean(
        (per_sensor_coherence_1 >= 0.5) & (per_sensor_coherence_2 >= 0.5), axis=0
    )
    usable_bins = usable_sensor_fraction_by_frequency >= 0.25

    median_frf_1 = np.median(first["frf"], axis=0)
    median_frf_2 = np.median(second["frf"], axis=0)
    common_coherent = (
        (per_sensor_coherence_1 >= 0.5)
        & (per_sensor_coherence_2 >= 0.5)
        & usable_bins[None, :]
    )
    log_1 = 20.0 * np.log10(np.abs(median_frf_1) + 1e-15)
    log_2 = 20.0 * np.log10(np.abs(median_frf_2) + 1e-15)
    if np.count_nonzero(common_coherent) >= 3:
        log_magnitude_correlation = float(
            np.corrcoef(log_1[common_coherent], log_2[common_coherent])[0, 1]
        )
        median_absolute_difference_db = float(
            np.median(np.abs(log_1[common_coherent] - log_2[common_coherent]))
        )
        phase_difference = np.angle(
            median_frf_1[common_coherent]
            * np.conjugate(median_frf_2[common_coherent])
        )
        median_absolute_phase_difference_degrees = float(
            np.degrees(np.median(np.abs(phase_difference)))
        )
    else:
        log_magnitude_correlation = float("nan")
        median_absolute_difference_db = float("nan")
        median_absolute_phase_difference_degrees = float("nan")

    match_fraction = len(matches) / max(1, min(len(peaks_1), len(peaks_2)))
    preliminary_go = bool(
        np.mean(usable_bins) >= 0.10
        and len(peaks_1) >= 3
        and len(peaks_2) >= 3
        and match_fraction >= 0.5
        and log_magnitude_correlation >= 0.65
    )
    summary = {
        "source": str(source.resolve()),
        "source_unchanged_during_audit": True,
        "recordings": list(PRIMARY_HEALTHY_NOISE_RECORDINGS),
        "block_seconds": BLOCK_SECONDS,
        "trim_seconds": TRIM_SECONDS,
        "welch_nperseg": WELCH_NPERSEG,
        "frequency_resolution_hz": float(np.median(np.diff(first["frequencies"]))),
        "frequency_band_hz": [
            float(first["frequencies"][0]),
            float(first["frequencies"][-1]),
        ],
        "block_counts": {
            name: int(result["frf"].shape[0]) for name, result in results.items()
        },
        "median_coherence_all_sensor_frequency": {
            first_name: float(np.median(first["coherence"])),
            second_name: float(np.median(second["coherence"])),
        },
        "usable_frequency_bin_fraction": float(np.mean(usable_bins)),
        "median_usable_sensor_fraction": float(
            np.median(usable_sensor_fraction_by_frequency)
        ),
        "modal_peaks_hz": {
            first_name: peaks_1.tolist(),
            second_name: peaks_2.tolist(),
        },
        "matched_modal_peaks": matches,
        "matched_peak_fraction": float(match_fraction),
        "healthy_log_magnitude_correlation": log_magnitude_correlation,
        "healthy_median_absolute_difference_db": median_absolute_difference_db,
        "healthy_median_absolute_phase_difference_degrees": (
            median_absolute_phase_difference_degrees
        ),
        "preliminary_go": preliminary_go,
        "preliminary_go_criteria": {
            "usable_frequency_bin_fraction_min": 0.10,
            "modal_peaks_each_min": 3,
            "matched_peak_fraction_min": 0.5,
            "healthy_log_magnitude_correlation_min": 0.65,
        },
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        frequencies_hz=first["frequencies"].astype(np.float32),
        recording_names=np.asarray(PRIMARY_HEALTHY_NOISE_RECORDINGS),
        frf_recording_1=first["frf"].astype(np.complex64),
        frf_recording_2=second["frf"].astype(np.complex64),
        coherence_recording_1=first["coherence"].astype(np.float32),
        coherence_recording_2=second["coherence"].astype(np.float32),
        response_sensor_names=np.asarray(RESPONSE_SENSORS),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    frequencies = first["frequencies"]
    axes[0].plot(frequencies, median_coherence_1, label=first_name)
    axes[0].plot(frequencies, median_coherence_2, label=second_name)
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Median coherence")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(frequencies, profile_1, label=first_name)
    axes[1].plot(frequencies, profile_2, label=second_name)
    axes[1].scatter(peaks_1, np.interp(peaks_1, frequencies, profile_1), s=18)
    axes[1].scatter(peaks_2, np.interp(peaks_2, frequencies, profile_2), s=18)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Coherent median FRF magnitude (dB)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.suptitle("HBTA healthy P2-noise FRF feasibility audit")
    figure.tight_layout()
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_figure, dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_HDF5)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.source, args.cache, args.output_json, args.output_figure)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
