"""Read-only metadata audit for the external HBTA HDF5 source."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py

from .config import DAMAGE_RECORDINGS, HEALTHY_RECORDINGS, RAW_HDF5


def _json_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def audit(source: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)

    source_stat_before = source.stat()
    expected = set(HEALTHY_RECORDINGS + DAMAGE_RECORDINGS)
    result = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "source_size_bytes": source_stat_before.st_size,
        "source_mtime_ns_before": source_stat_before.st_mtime_ns,
        "expected_recordings": sorted(expected),
        "recordings": {},
    }

    with h5py.File(source, "r") as handle:
        result["all_recording_names"] = sorted(handle.keys())
        result["missing_expected_recordings"] = sorted(expected - set(handle.keys()))
        for name in sorted(expected & set(handle.keys())):
            group = handle[name]
            acceleration = group["acceleration"]
            sensor_names = sorted(acceleration.keys())
            channel_lengths = {}
            for sensor in sensor_names:
                channel_lengths[sensor] = {
                    channel: int(acceleration[sensor][channel].shape[0])
                    for channel in acceleration[sensor].keys()
                }
            sample_rate = int(group.attrs["sample_rate"])
            lengths = [
                length
                for channels in channel_lengths.values()
                for length in channels.values()
            ]
            result["recordings"][name] = {
                "attributes": {
                    key: _json_value(value) for key, value in group.attrs.items()
                },
                "sample_rate_hz": sample_rate,
                "duration_seconds_min": min(lengths) / sample_rate,
                "duration_seconds_max": max(lengths) / sample_rate,
                "acceleration_sensors": sensor_names,
                "channel_lengths": channel_lengths,
            }

    source_stat_after = source.stat()
    result["source_mtime_ns_after"] = source_stat_after.st_mtime_ns
    result["source_unchanged_during_audit"] = (
        source_stat_before.st_size == source_stat_after.st_size
        and source_stat_before.st_mtime_ns == source_stat_after.st_mtime_ns
    )
    if not result["source_unchanged_during_audit"]:
        raise RuntimeError("Raw HDF5 source changed during read-only audit")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_HDF5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

