"""Prepare non-overlapping, sensor-wise spectral features for HBTA."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


RECORDINGS = [
    "MVS_P2_UDS_NM_Z_01",
    "MVS_P2_UDS_NM_Z_02",
    *[f"MVS_P2_DS{index}_NM_Z_01" for index in range(1, 9)],
]
SENSORS = [
    *[f"AG{index:02d}" for index in range(1, 19)],
    *[f"AL{index:02d}" for index in range(1, 41)],
]
BANDS_HZ = np.asarray(
    [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0),
     (20.0, 35.0), (35.0, 45.0)],
    dtype=np.float64,
)
FEATURE_NAMES = np.asarray(
    [
        "log_power_0.5_2Hz",
        "log_power_2_5Hz",
        "log_power_5_10Hz",
        "log_power_10_20Hz",
        "log_power_20_35Hz",
        "log_power_35_45Hz",
        "log_rms",
        "kurtosis",
        "spectral_centroid_hz",
    ]
)


def block_features(blocks: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return features with shape [block, sensor, feature]."""
    frequencies, psd = welch(
        blocks,
        fs=sample_rate,
        window="hann",
        nperseg=min(512, blocks.shape[-1]),
        noverlap=min(256, blocks.shape[-1] // 2),
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    epsilon = np.finfo(np.float64).tiny
    features = []
    for low, high in BANDS_HZ:
        selected = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(psd[..., selected], frequencies[selected], axis=-1)
        features.append(np.log(power + epsilon))

    centered = blocks - blocks.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=-1))
    features.append(np.log(rms + epsilon))
    features.append(kurtosis(centered, axis=-1, fisher=False, bias=False))
    centroid = np.sum(psd * frequencies, axis=-1) / (np.sum(psd, axis=-1) + epsilon)
    features.append(centroid)
    return np.stack(features, axis=-1).astype(np.float32)


def prepare(hdf5_path: Path, block_seconds: float, trim_seconds: float) -> dict:
    all_features = []
    recording_ids = []
    states = []
    block_starts = []
    positions = None

    with h5py.File(hdf5_path, "r") as data:
        for recording_index, recording_name in enumerate(RECORDINGS):
            group = data[recording_name]
            sample_rate = int(group.attrs["sample_rate"])
            block_samples = int(round(block_seconds * sample_rate))
            trim_samples = int(round(trim_seconds * sample_rate))
            acceleration = group["acceleration"]

            if positions is None:
                positions = np.concatenate(
                    [np.asarray(acceleration[name].attrs["position"]) for name in SENSORS],
                    axis=0,
                ).astype(np.float32)

            signals = np.stack(
                [np.asarray(acceleration[name]["z"], dtype=np.float64) for name in SENSORS]
            )
            stop = signals.shape[1] - trim_samples
            usable_samples = stop - trim_samples
            block_count = usable_samples // block_samples
            if block_count < 1:
                raise ValueError(f"No complete blocks in {recording_name}")

            signals = signals[
                :, trim_samples : trim_samples + block_count * block_samples
            ]
            blocks = signals.reshape(len(SENSORS), block_count, block_samples).transpose(1, 0, 2)
            features = block_features(blocks, sample_rate)
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite features in {recording_name}")

            all_features.append(features)
            recording_ids.extend([recording_index] * block_count)
            states.extend([str(group.attrs["damage_state"])] * block_count)
            block_starts.extend(
                trim_seconds + np.arange(block_count, dtype=np.float64) * block_seconds
            )
            print(f"{recording_name}: {block_count} blocks")

    return {
        "features": np.concatenate(all_features),
        "recording_ids": np.asarray(recording_ids, dtype=np.int16),
        "recording_names": np.asarray(RECORDINGS),
        "states": np.asarray(states),
        "block_starts_seconds": np.asarray(block_starts, dtype=np.float32),
        "sensor_names": np.asarray(SENSORS),
        "sensor_positions": positions,
        "feature_names": FEATURE_NAMES,
        "bands_hz": BANDS_HZ.astype(np.float32),
        "block_seconds": np.asarray(block_seconds, dtype=np.float32),
        "trim_seconds": np.asarray(trim_seconds, dtype=np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output", type=Path, default=Path("HBTA/prepared/features.npz"))
    parser.add_argument("--block-seconds", type=float, default=10.0)
    parser.add_argument("--trim-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hdf5.is_file():
        raise FileNotFoundError(args.hdf5)
    if args.block_seconds <= 0 or args.trim_seconds < 0:
        raise ValueError("Block duration must be positive and trim non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare(args.hdf5, args.block_seconds, args.trim_seconds)
    np.savez_compressed(args.output, **prepared)
    print(f"Wrote {args.output}: {prepared['features'].shape}")


if __name__ == "__main__":
    main()
