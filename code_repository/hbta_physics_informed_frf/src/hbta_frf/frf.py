"""Read-only HBTA loading and H1 frequency-response estimation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import csd, welch


def load_vertical_recording(
    source: Path,
    recording_name: str,
    response_sensors: tuple[str, ...],
    shaker_sensor: str = "AS",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load response and shaker acceleration without modifying the HDF5 file."""
    with h5py.File(source, "r") as handle:
        group = handle[recording_name]
        sample_rate = float(group.attrs["sample_rate"])
        acceleration = group["acceleration"]
        response = np.stack(
            [
                np.asarray(acceleration[sensor]["z"], dtype=np.float64)
                for sensor in response_sensors
            ]
        )
        shaker = np.asarray(acceleration[shaker_sensor]["z"], dtype=np.float64)

    common_length = min(response.shape[1], shaker.shape[0])
    response = response[:, :common_length]
    shaker = shaker[:common_length]
    if not np.isfinite(response).all() or not np.isfinite(shaker).all():
        raise ValueError(f"Non-finite acceleration data in {recording_name}")
    return response, shaker, sample_rate


def nonoverlapping_blocks(
    signal: np.ndarray,
    sample_rate: float,
    block_seconds: float,
    trim_seconds: float,
) -> np.ndarray:
    """Return complete non-overlapping blocks along the last axis."""
    block_samples = int(round(block_seconds * sample_rate))
    trim_samples = int(round(trim_seconds * sample_rate))
    if block_samples <= 0 or trim_samples < 0:
        raise ValueError("Block duration must be positive and trim non-negative")
    stop = signal.shape[-1] - trim_samples
    usable = stop - trim_samples
    block_count = usable // block_samples
    if block_count < 1:
        raise ValueError("No complete block after edge trimming")
    selected = signal[..., trim_samples : trim_samples + block_count * block_samples]
    return selected.reshape(*signal.shape[:-1], block_count, block_samples)


def estimate_h1_frf(
    response_blocks: np.ndarray,
    shaker_blocks: np.ndarray,
    sample_rate: float,
    nperseg: int,
    overlap_fraction: float,
    minimum_hz: float,
    maximum_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate H1 FRF and coherence for [block, sensor, sample] responses."""
    if response_blocks.ndim != 3 or shaker_blocks.ndim != 2:
        raise ValueError("Expected response [block,sensor,sample] and shaker [block,sample]")
    if response_blocks.shape[0] != shaker_blocks.shape[0]:
        raise ValueError("Response and shaker block counts differ")
    if response_blocks.shape[-1] != shaker_blocks.shape[-1]:
        raise ValueError("Response and shaker sample counts differ")

    segment = min(int(nperseg), response_blocks.shape[-1])
    noverlap = int(round(segment * overlap_fraction))
    noverlap = min(max(noverlap, 0), segment - 1)
    shaker_expanded = np.broadcast_to(
        shaker_blocks[:, None, :], response_blocks.shape
    )

    frequencies, input_psd = welch(
        shaker_blocks,
        fs=sample_rate,
        window="hann",
        nperseg=segment,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    _, output_psd = welch(
        response_blocks,
        fs=sample_rate,
        window="hann",
        nperseg=segment,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    _, cross_psd = csd(
        shaker_expanded,
        response_blocks,
        fs=sample_rate,
        window="hann",
        nperseg=segment,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )

    epsilon = np.finfo(np.float64).tiny
    frf = cross_psd / (input_psd[:, None, :] + epsilon)
    coherence = np.abs(cross_psd) ** 2 / (
        input_psd[:, None, :] * output_psd + epsilon
    )
    coherence = np.clip(coherence.real, 0.0, 1.0)
    selected = (frequencies >= minimum_hz) & (frequencies <= maximum_hz)
    return (
        frequencies[selected],
        frf[..., selected],
        coherence[..., selected],
        input_psd[..., selected],
    )
