"""Small shared-frequency Gaussian-process smoother for complex FRFs."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def matern32_kernel(frequencies_hz: np.ndarray, length_scale_hz: float) -> np.ndarray:
    """Unit-amplitude Matérn-3/2 covariance on a one-dimensional frequency grid."""
    distance = np.abs(frequencies_hz[:, None] - frequencies_hz[None, :])
    scaled = np.sqrt(3.0) * distance / length_scale_hz
    return (1.0 + scaled) * np.exp(-scaled)


def smooth_complex_residual(
    frequencies_hz: np.ndarray,
    targets: np.ndarray,
    prior_mean: np.ndarray,
    length_scale_hz: float,
    nugget_ratio: float,
) -> np.ndarray:
    """Return GP posterior mean at observed frequencies for all sensors.

    The same frequency covariance is shared by sensors, while every sensor and
    complex component has its own output coefficients. ``nugget_ratio`` is the
    observation-noise variance divided by the unit kernel variance.
    """
    kernel = matern32_kernel(frequencies_hz, length_scale_hz)
    factor = cho_factor(
        kernel + nugget_ratio * np.eye(len(frequencies_hz)),
        lower=True,
        check_finite=False,
    )
    residual = targets - prior_mean
    coefficients = cho_solve(factor, residual.T, check_finite=False)
    return prior_mean + (kernel @ coefficients).T
