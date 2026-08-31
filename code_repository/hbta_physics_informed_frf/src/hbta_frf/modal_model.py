"""Shared-pole modal acceleration-FRF model with variable projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def modal_basis(
    frequencies_hz: np.ndarray,
    modal_frequencies_hz: np.ndarray,
    damping_ratios: np.ndarray,
    include_constant: bool = True,
) -> np.ndarray:
    """Return complex modal acceleration-FRF basis [frequency, component]."""
    omega = 2.0 * np.pi * np.asarray(frequencies_hz, dtype=np.float64)
    modal_omega = 2.0 * np.pi * np.asarray(
        modal_frequencies_hz, dtype=np.float64
    )
    damping = np.asarray(damping_ratios, dtype=np.float64)
    denominator = (
        modal_omega[None, :] ** 2
        - omega[:, None] ** 2
        + 2j * damping[None, :] * modal_omega[None, :] * omega[:, None]
    )
    basis = -(omega[:, None] ** 2) / denominator
    if include_constant:
        basis = np.column_stack([basis, np.ones(len(omega), dtype=np.complex128)])
    return basis


def fit_complex_residues(
    basis: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit independent complex residues for targets [sensor, frequency]."""
    residues = np.empty((targets.shape[0], basis.shape[1]), dtype=np.complex128)
    predictions = np.empty_like(targets, dtype=np.complex128)
    for sensor_index in range(targets.shape[0]):
        root_weight = np.sqrt(np.clip(weights[sensor_index], 1e-6, 1.0))
        weighted_basis = basis * root_weight[:, None]
        weighted_target = targets[sensor_index] * root_weight
        residues[sensor_index], *_ = np.linalg.lstsq(
            weighted_basis, weighted_target, rcond=None
        )
        predictions[sensor_index] = basis @ residues[sensor_index]
    return residues, predictions


@dataclass
class ModalFit:
    modal_frequencies_hz: np.ndarray
    damping_ratios: np.ndarray
    residues: np.ndarray
    prediction: np.ndarray
    weighted_nmse: float
    optimizer_success: bool
    optimizer_message: str


def fit_shared_pole_model(
    frequencies_hz: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    initial_modal_frequencies_hz: tuple[float, ...],
) -> ModalFit:
    """Fit shared modal frequencies/damping and sensor-specific residues."""
    initial_frequencies = np.asarray(initial_modal_frequencies_hz, dtype=np.float64)
    mode_count = len(initial_frequencies)
    frequency_bounds = [
        (float(value - 0.6), float(value + 0.6)) for value in initial_frequencies
    ]
    damping_bounds = [(0.001, 0.08)] * mode_count
    initial = np.concatenate([initial_frequencies, np.full(mode_count, 0.01)])
    target_energy = np.sum(weights * np.abs(targets) ** 2) + 1e-18

    def objective(parameters: np.ndarray) -> float:
        basis = modal_basis(
            frequencies_hz,
            parameters[:mode_count],
            parameters[mode_count:],
        )
        _, prediction = fit_complex_residues(basis, targets, weights)
        error = np.sum(weights * np.abs(targets - prediction) ** 2)
        return float(error / target_energy)

    optimization = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=frequency_bounds + damping_bounds,
        options={"maxiter": 250, "ftol": 1e-11},
    )
    parameters = optimization.x
    basis = modal_basis(
        frequencies_hz,
        parameters[:mode_count],
        parameters[mode_count:],
    )
    residues, prediction = fit_complex_residues(basis, targets, weights)
    return ModalFit(
        modal_frequencies_hz=parameters[:mode_count],
        damping_ratios=parameters[mode_count:],
        residues=residues,
        prediction=prediction,
        weighted_nmse=objective(parameters),
        optimizer_success=bool(optimization.success),
        optimizer_message=str(optimization.message),
    )
