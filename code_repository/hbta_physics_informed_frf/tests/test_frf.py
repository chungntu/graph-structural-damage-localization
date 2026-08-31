import numpy as np
from scipy.signal import butter, lfilter

from hbta_frf.frf import estimate_h1_frf


def test_h1_frf_recovers_linear_filter_shape():
    rng = np.random.default_rng(7)
    sample_rate = 100.0
    block_count = 8
    block_samples = 6000
    shaker = rng.normal(size=(block_count, block_samples))
    numerator, denominator = butter(2, 8.0, fs=sample_rate)
    response = lfilter(numerator, denominator, shaker, axis=-1)
    response = response[:, None, :]

    frequencies, frf, coherence, _ = estimate_h1_frf(
        response,
        shaker,
        sample_rate=sample_rate,
        nperseg=1024,
        overlap_fraction=0.5,
        minimum_hz=1.0,
        maximum_hz=30.0,
    )

    assert frf.shape[:2] == (block_count, 1)
    assert np.median(coherence) > 0.99
    low = np.argmin(np.abs(frequencies - 2.0))
    high = np.argmin(np.abs(frequencies - 20.0))
    assert np.median(np.abs(frf[:, 0, low])) > np.median(
        np.abs(frf[:, 0, high])
    )
