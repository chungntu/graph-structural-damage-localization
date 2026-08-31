"""Prepare block-wise complex sensor spectra for arbitrary HBTA graph edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import stft

from hbta_pipeline.prepare import RECORDINGS, SENSORS
from hbta_pipeline.prepare_vqpiae import MULTICONDITION_RECORDINGS


def block_fourier_coefficients(
    blocks: np.ndarray,
    sample_rate: int,
    minimum_hz: float = 1.0,
    maximum_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complex Hann-window coefficients as [block,sensor,segment,freq]."""
    nperseg = min(256, blocks.shape[-1])
    noverlap = nperseg // 2
    frequencies, _, coefficients = stft(
        blocks,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )
    selected = (frequencies >= minimum_hz) & (frequencies < maximum_hz)
    coefficients = np.moveaxis(coefficients[..., selected, :], -1, -2)
    return (
        frequencies[selected].astype(np.float32),
        coefficients.astype(np.complex64),
    )


def prepare(
    hdf5_path: Path,
    block_seconds: float,
    trim_seconds: float,
    recordings: list[str] = RECORDINGS,
) -> dict:
    all_coefficients = []
    recording_ids = []
    states = []
    block_starts = []
    positions = None
    selected_frequencies = None

    with h5py.File(hdf5_path, "r") as data:
        for recording_index, recording_name in enumerate(recordings):
            group = data[recording_name]
            sample_rate = int(group.attrs["sample_rate"])
            block_samples = int(round(block_seconds * sample_rate))
            trim_samples = int(round(trim_seconds * sample_rate))
            acceleration = group["acceleration"]
            if positions is None:
                positions = np.concatenate(
                    [
                        np.asarray(acceleration[name].attrs["position"])
                        for name in SENSORS
                    ],
                    axis=0,
                ).astype(np.float32)

            response = np.stack(
                [
                    np.asarray(acceleration[name]["z"], dtype=np.float64)
                    for name in SENSORS
                ]
            )
            stop = response.shape[1] - trim_samples
            block_count = (stop - trim_samples) // block_samples
            sample_stop = trim_samples + block_count * block_samples
            response = response[:, trim_samples:sample_stop]
            blocks = response.reshape(
                len(SENSORS), block_count, block_samples
            ).transpose(1, 0, 2)
            frequencies, coefficients = block_fourier_coefficients(
                blocks, sample_rate
            )
            if selected_frequencies is None:
                selected_frequencies = frequencies
            else:
                np.testing.assert_allclose(
                    selected_frequencies, frequencies
                )
            if not np.isfinite(coefficients).all():
                raise ValueError(
                    f"Non-finite Fourier coefficients in {recording_name}"
                )

            all_coefficients.append(coefficients)
            recording_ids.extend([recording_index] * block_count)
            states.extend([str(group.attrs["damage_state"])] * block_count)
            block_starts.extend(
                trim_seconds
                + np.arange(block_count, dtype=np.float64) * block_seconds
            )
            print(
                f"{recording_name}: {block_count} blocks, "
                f"{coefficients.shape[2]} segments",
                flush=True,
            )

    return {
        "fourier_coefficients": np.concatenate(all_coefficients),
        "frequencies_hz": selected_frequencies,
        "recording_ids": np.asarray(recording_ids, dtype=np.int16),
        "recording_names": np.asarray(recordings),
        "states": np.asarray(states),
        "block_starts_seconds": np.asarray(
            block_starts, dtype=np.float32
        ),
        "sensor_names": np.asarray(SENSORS),
        "sensor_positions": positions,
        "block_seconds": np.asarray(block_seconds, dtype=np.float32),
        "trim_seconds": np.asarray(trim_seconds, dtype=np.float32),
        "nperseg": np.asarray(
            min(256, int(round(block_seconds * 100))), dtype=np.int16
        ),
        "noverlap": np.asarray(
            min(256, int(round(block_seconds * 100))) // 2,
            dtype=np.int16,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HBTA/prepared/edge_spectra.npz"),
    )
    parser.add_argument("--block-seconds", type=float, default=10.0)
    parser.add_argument("--trim-seconds", type=float, default=30.0)
    parser.add_argument("--multicondition-healthy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hdf5.is_file():
        raise FileNotFoundError(args.hdf5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recordings = (
        MULTICONDITION_RECORDINGS
        if args.multicondition_healthy
        else RECORDINGS
    )
    result = prepare(
        args.hdf5, args.block_seconds, args.trim_seconds, recordings
    )
    np.savez_compressed(args.output, **result)
    print(
        f"Wrote {args.output}: "
        f"{result['fourier_coefficients'].shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
