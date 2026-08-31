import numpy as np

from hbta_frf.modal_model import fit_shared_pole_model, modal_basis


def test_shared_pole_fit_recovers_synthetic_modal_frequencies():
    rng = np.random.default_rng(5)
    frequencies = np.linspace(3.0, 25.0, 300)
    true_modes = np.asarray([6.5, 16.0, 22.5])
    damping = np.asarray([0.01, 0.015, 0.02])
    basis = modal_basis(frequencies, true_modes, damping)
    residues = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    targets = residues @ basis.T
    targets += 0.002 * (
        rng.normal(size=targets.shape) + 1j * rng.normal(size=targets.shape)
    )
    weights = np.ones_like(targets.real)

    fitted = fit_shared_pole_model(
        frequencies,
        targets,
        weights,
        initial_modal_frequencies_hz=(6.4, 16.2, 22.4),
    )

    np.testing.assert_allclose(fitted.modal_frequencies_hz, true_modes, atol=0.08)
    assert fitted.weighted_nmse < 1e-3
