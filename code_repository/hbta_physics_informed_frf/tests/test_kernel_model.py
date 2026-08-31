import numpy as np

from hbta_frf.kernel_model import smooth_complex_residual


def test_physical_prior_reduces_sparse_smoothing_bias() -> None:
    frequencies = np.linspace(0.0, 10.0, 101)
    physical = (1.0 / (4.0 - frequencies + 0.2j))[None, :]
    discrepancy = (0.05 * np.sin(frequencies))[None, :]
    target = physical + discrepancy
    black = smooth_complex_residual(
        frequencies, target, np.zeros_like(target), 1.0, 1.0
    )
    grey = smooth_complex_residual(
        frequencies, target, physical, 1.0, 1.0
    )
    assert np.mean(np.abs(grey - target) ** 2) < np.mean(np.abs(black - target) ** 2)
