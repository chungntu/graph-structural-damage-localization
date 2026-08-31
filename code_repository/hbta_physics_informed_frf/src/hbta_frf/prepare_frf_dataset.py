"""Prepare derived FRFs in the isolated workspace from read-only HBTA data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import (
    BLOCK_SECONDS,
    MAXIMUM_FREQUENCY_HZ,
    MINIMUM_FREQUENCY_HZ,
    PRIMARY_DAMAGE_NOISE_RECORDINGS,
    PRIMARY_HEALTHY_NOISE_RECORDINGS,
    RAW_HDF5,
    RESPONSE_SENSORS,
    TRIM_SECONDS,
    WELCH_NPERSEG,
    WELCH_OVERLAP_FRACTION,
)
from .frf import estimate_h1_frf, load_vertical_recording, nonoverlapping_blocks


def prepare(source: Path) -> dict:
    recordings = PRIMARY_HEALTHY_NOISE_RECORDINGS + PRIMARY_DAMAGE_NOISE_RECORDINGS
    all_frf = []
    all_coherence = []
    recording_ids = []
    states = []
    frequencies = None
    source_before = source.stat()

    for recording_id, name in enumerate(recordings):
        response, shaker, sample_rate = load_vertical_recording(
            source, name, RESPONSE_SENSORS
        )
        response_blocks = nonoverlapping_blocks(
            response, sample_rate, BLOCK_SECONDS, TRIM_SECONDS
        ).transpose(1, 0, 2)
        shaker_blocks = nonoverlapping_blocks(
            shaker, sample_rate, BLOCK_SECONDS, TRIM_SECONDS
        )
        current_frequencies, frf, coherence, _ = estimate_h1_frf(
            response_blocks,
            shaker_blocks,
            sample_rate,
            WELCH_NPERSEG,
            WELCH_OVERLAP_FRACTION,
            MINIMUM_FREQUENCY_HZ,
            MAXIMUM_FREQUENCY_HZ,
        )
        if frequencies is None:
            frequencies = current_frequencies
        else:
            np.testing.assert_allclose(frequencies, current_frequencies)
        all_frf.append(frf.astype(np.complex64))
        all_coherence.append(coherence.astype(np.float32))
        recording_ids.extend([recording_id] * len(frf))
        state = "UDS" if "_UDS_" in name else name.split("_")[2]
        states.extend([state] * len(frf))
        print(f"{name}: {len(frf)} non-overlapping blocks", flush=True)

    source_after = source.stat()
    if (
        source_before.st_size != source_after.st_size
        or source_before.st_mtime_ns != source_after.st_mtime_ns
    ):
        raise RuntimeError("Raw HDF5 source changed during FRF preparation")

    return {
        "frf": np.concatenate(all_frf),
        "coherence": np.concatenate(all_coherence),
        "frequencies_hz": frequencies.astype(np.float32),
        "recording_ids": np.asarray(recording_ids, dtype=np.int16),
        "recording_names": np.asarray(recordings),
        "states": np.asarray(states),
        "response_sensor_names": np.asarray(RESPONSE_SENSORS),
        "block_seconds": np.asarray(BLOCK_SECONDS, dtype=np.float32),
        "trim_seconds": np.asarray(TRIM_SECONDS, dtype=np.float32),
        "source_path": np.asarray(str(source.resolve())),
        "source_size_bytes": np.asarray(source_before.st_size, dtype=np.int64),
        "source_mtime_ns": np.asarray(source_before.st_mtime_ns, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_HDF5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Wrote {args.output}: {result['frf'].shape}")


if __name__ == "__main__":
    main()
