"""Extract recording-block modal poles while allowing free sensor residues."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ
from .modal_model import (
    fit_complex_residues,
    fit_shared_pole_model,
    modal_basis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-count", type=int, default=12)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    frequencies = data["frequencies_hz"].astype(float)
    frf = data["frf"].astype(np.complex128)
    coherence = data["coherence"].astype(float)
    recording_ids = data["recording_ids"]
    band = (frequencies >= 5.0) & (frequencies <= 26.0)
    frequencies = frequencies[band]
    frf = frf[:, :, band]
    coherence = coherence[:, :, band]

    development = recording_ids == 0
    development_frf = np.median(frf[development][:8], axis=0)
    development_coherence = np.median(coherence[development][:8], axis=0)
    sensor_energy = np.sum(
        development_coherence * np.abs(development_frf) ** 2, axis=1
    )
    selected_sensors = np.argsort(sensor_energy)[-args.sensor_count :]

    modal_frequencies = []
    damping_ratios = []
    fit_nmse = []
    optimizer_success = []
    mode_shapes = []
    for block_index in range(len(frf)):
        weights = coherence[block_index, selected_sensors]
        weights = np.where(weights >= 0.5, weights, 0.0)
        fit = fit_shared_pole_model(
            frequencies,
            frf[block_index, selected_sensors],
            weights,
            INITIAL_VERTICAL_MODAL_FREQUENCIES_HZ,
        )
        modal_frequencies.append(fit.modal_frequencies_hz)
        damping_ratios.append(fit.damping_ratios)
        fit_nmse.append(fit.weighted_nmse)
        optimizer_success.append(fit.optimizer_success)
        full_weights = np.where(coherence[block_index] >= 0.5, coherence[block_index], 0.0)
        basis = modal_basis(
            frequencies,
            fit.modal_frequencies_hz,
            fit.damping_ratios,
        )
        full_residues, _ = fit_complex_residues(
            basis, frf[block_index], full_weights
        )
        # For a fixed input, each modal-residue column is a mode shape times
        # one unknown complex participation scalar. MAC removes that scalar.
        mode_shapes.append(full_residues[:, : len(fit.modal_frequencies_hz)].T)
        if (block_index + 1) % 10 == 0 or block_index + 1 == len(frf):
            print(f"fitted {block_index + 1}/{len(frf)} blocks", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        modal_frequencies_hz=np.asarray(modal_frequencies),
        damping_ratios=np.asarray(damping_ratios),
        weighted_nmse=np.asarray(fit_nmse),
        optimizer_success=np.asarray(optimizer_success),
        mode_shapes=np.asarray(mode_shapes),
        selected_sensor_indices=selected_sensors,
        selected_sensor_names=data["response_sensor_names"][selected_sensors],
        recording_ids=recording_ids,
        recording_names=data["recording_names"],
        states=data["states"],
        frequency_resolution_hz=np.asarray(float(np.diff(frequencies).mean())),
        source_dataset=np.asarray(str(args.dataset.resolve())),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
