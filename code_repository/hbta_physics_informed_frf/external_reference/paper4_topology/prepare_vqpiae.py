"""Prepare HBTA descriptors tailored to the VQ-PIAE-A mechanism."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import csd, welch
from scipy.stats import kurtosis

from hbta_pipeline.prepare import RECORDINGS, SENSORS


BANDS_HZ = np.asarray(
    [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 15.0),
     (15.0, 20.0), (20.0, 30.0)],
    dtype=np.float64,
)
FEATURE_NAMES = np.asarray(
    [
        *[f"response_log_power_{low:g}_{high:g}Hz" for low, high in BANDS_HZ],
        "response_log_rms",
        "response_kurtosis",
        "response_spectral_centroid_hz",
        *[f"shaker_transmissibility_{low:g}_{high:g}Hz" for low, high in BANDS_HZ],
        *[f"shaker_coherence_{low:g}_{high:g}Hz" for low, high in BANDS_HZ],
    ]
)
SHAKER_EOV_FEATURE_NAMES = np.asarray(
    [
        *[f"shaker_log_power_{low:g}_{high:g}Hz" for low, high in BANDS_HZ],
        "shaker_log_rms",
        "shaker_kurtosis",
        "shaker_spectral_centroid_hz",
    ]
)
MULTICONDITION_RECORDINGS = [
    "MVS_P1_UDS_SM_Z_01",
    "MVS_P1_UDS_SM_Z_02",
    "MVS_P2_UDS_SM_Z_01",
    "MVS_P2_UDS_SM_Z_02",
    "MVS_P2_UDS_NM_Z_01",
    "MVS_P2_UDS_NM_Z_02",
    *[f"MVS_P2_DS{index}_NM_Z_01" for index in range(1, 9)],
]


def descriptor_features(
    response: np.ndarray,
    shaker: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Compute response, shaker-transmissibility and coherence descriptors."""
    nperseg = min(512, response.shape[-1])
    noverlap = min(256, response.shape[-1] // 2)
    frequencies, response_psd = welch(
        response,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    _, shaker_psd = welch(
        shaker,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    _, cross_psd = csd(
        shaker[:, None, :],
        response,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    epsilon = np.finfo(np.float64).tiny
    transfer_power = np.abs(cross_psd / (shaker_psd[:, None, :] + epsilon)) ** 2
    coherence = np.abs(cross_psd) ** 2 / (
        response_psd * shaker_psd[:, None, :] + epsilon
    )
    coherence = np.clip(coherence, 0.0, 1.0)

    response_features = []
    transfer_features = []
    coherence_features = []
    for low, high in BANDS_HZ:
        selected = (frequencies >= low) & (frequencies < high)
        response_power = np.trapezoid(
            response_psd[..., selected], frequencies[selected], axis=-1
        )
        response_features.append(np.log(response_power + epsilon))
        transfer_features.append(
            np.log(np.mean(transfer_power[..., selected], axis=-1) + epsilon)
        )
        coherence_features.append(np.mean(coherence[..., selected], axis=-1))

    centered = response - response.mean(axis=-1, keepdims=True)
    response_features.append(
        np.log(np.sqrt(np.mean(centered**2, axis=-1)) + epsilon)
    )
    response_features.append(
        kurtosis(centered, axis=-1, fisher=False, bias=False)
    )
    selected_spectrum = (frequencies >= 1.0) & (frequencies < 30.0)
    centroid = np.sum(
        response_psd[..., selected_spectrum] * frequencies[selected_spectrum], axis=-1
    ) / (np.sum(response_psd[..., selected_spectrum], axis=-1) + epsilon)
    response_features.append(centroid)
    return np.stack(
        [*response_features, *transfer_features, *coherence_features], axis=-1
    ).astype(np.float32)


def shaker_eov_features(shaker: np.ndarray, sample_rate: int) -> np.ndarray:
    """Describe the measured excitation for healthy-only EOV adjustment."""
    nperseg = min(512, shaker.shape[-1])
    noverlap = min(256, shaker.shape[-1] // 2)
    frequencies, shaker_psd = welch(
        shaker,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    epsilon = np.finfo(np.float64).tiny
    features = []
    for low, high in BANDS_HZ:
        selected = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(
            shaker_psd[..., selected], frequencies[selected], axis=-1
        )
        features.append(np.log(power + epsilon))
    centered = shaker - shaker.mean(axis=-1, keepdims=True)
    features.append(np.log(np.sqrt(np.mean(centered**2, axis=-1)) + epsilon))
    features.append(kurtosis(centered, axis=-1, fisher=False, bias=False))
    selected = (frequencies >= 1.0) & (frequencies < 30.0)
    centroid = np.sum(
        shaker_psd[..., selected] * frequencies[selected], axis=-1
    ) / (np.sum(shaker_psd[..., selected], axis=-1) + epsilon)
    features.append(centroid)
    result = np.stack(features, axis=-1).astype(np.float32)
    if result.shape[-1] != SHAKER_EOV_FEATURE_NAMES.size:
        raise AssertionError(
            f"Expected {SHAKER_EOV_FEATURE_NAMES.size} shaker EOV descriptors"
        )
    if not np.isfinite(result).all():
        raise ValueError("Shaker EOV descriptors contain non-finite values")
    return result


def prepare(
    hdf5_path: Path,
    block_seconds: float,
    trim_seconds: float,
    recordings: list[str],
) -> dict:
    all_features = []
    all_shaker_eov_features = []
    recording_ids = []
    states = []
    block_starts = []
    positions = None

    with h5py.File(hdf5_path, "r") as data:
        for recording_index, recording_name in enumerate(recordings):
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

            response = np.stack(
                [np.asarray(acceleration[name]["z"], dtype=np.float64) for name in SENSORS]
            )
            shaker = np.asarray(acceleration["AS"]["z"], dtype=np.float64)
            stop = response.shape[1] - trim_samples
            block_count = (stop - trim_samples) // block_samples
            sample_stop = trim_samples + block_count * block_samples
            response = response[:, trim_samples:sample_stop]
            shaker = shaker[trim_samples:sample_stop]
            response_blocks = response.reshape(
                len(SENSORS), block_count, block_samples
            ).transpose(1, 0, 2)
            shaker_blocks = shaker.reshape(block_count, block_samples)
            features = descriptor_features(response_blocks, shaker_blocks, sample_rate)
            eov_features = shaker_eov_features(shaker_blocks, sample_rate)
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite descriptors in {recording_name}")

            all_features.append(features)
            all_shaker_eov_features.append(eov_features)
            recording_ids.extend([recording_index] * block_count)
            states.extend([str(group.attrs["damage_state"])] * block_count)
            block_starts.extend(
                trim_seconds + np.arange(block_count, dtype=np.float64) * block_seconds
            )
            print(f"{recording_name}: {block_count} blocks", flush=True)

    return {
        "features": np.concatenate(all_features),
        "shaker_eov_features": np.concatenate(all_shaker_eov_features),
        "shaker_eov_feature_names": SHAKER_EOV_FEATURE_NAMES,
        "recording_ids": np.asarray(recording_ids, dtype=np.int16),
        "recording_names": np.asarray(recordings),
        "states": np.asarray(states),
        "block_starts_seconds": np.asarray(block_starts, dtype=np.float32),
        "sensor_names": np.asarray(SENSORS),
        "sensor_positions": positions,
        "feature_names": FEATURE_NAMES,
        "bands_hz": BANDS_HZ.astype(np.float32),
        "block_seconds": np.asarray(block_seconds, dtype=np.float32),
        "trim_seconds": np.asarray(trim_seconds, dtype=np.float32),
        "descriptor_groups": np.asarray(
            ["response"] * 9 + ["transmissibility"] * 6 + ["coherence"] * 6
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("HBTA/prepared/vqpiae_features.npz")
    )
    parser.add_argument("--block-seconds", type=float, default=10.0)
    parser.add_argument("--trim-seconds", type=float, default=30.0)
    parser.add_argument(
        "--multicondition-healthy",
        action="store_true",
        help="Include additional vertical healthy recordings before the controlled test set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hdf5.is_file():
        raise FileNotFoundError(args.hdf5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recordings = MULTICONDITION_RECORDINGS if args.multicondition_healthy else RECORDINGS
    prepared = prepare(args.hdf5, args.block_seconds, args.trim_seconds, recordings)
    np.savez_compressed(args.output, **prepared)
    print(f"Wrote {args.output}: {prepared['features'].shape}")


if __name__ == "__main__":
    main()
