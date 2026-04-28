"""
Integration tests: real tomogram + logR + probability with actual detector data.
No GPU or MPI required.  Slower than unit tests (~10-30 s for small config).

Run with:
    pytest test_integration.py -v
"""

import numpy as np
import pytest
import h5py
from pathlib import Path

TEST_DIR = Path(__file__).parent


@pytest.fixture(scope='module')
def emc_config():
    """
    One-class config with small parameters, model initialised, data loaded.
    Written to the testing/ directory (same cwd as run_emc_maxwell.py).
    """
    import runpy, os

    os.chdir(TEST_DIR)

    config = runpy.run_path(str(TEST_DIR / 'config.py'))['make_pickle_file'](
        shape=(64, 64, 64),  # must match detector q-range; smaller shapes produce empty masks
        rotation_order=2,    # fewest rotations (91) for speed
        classes=1,
        classes_2D=0,
    )

    D = config['classes'][0]['P_data'].shape[0]

    for c in config['classes']:
        c['model'].init_random()
        c['fluence'] = np.ones(D, dtype=float)
        c['P_data'].load_from_file()
        c['data'].load_from_file()

    return config


@pytest.fixture(scope='module')
def logR_result(emc_config):
    """Run CPU logR on the small config and return (config, logR array per class)."""
    from emc3.likelihood import calculate_logR

    calculate_logR(emc_config)

    # each class stores logR in c['logR_dr']
    return emc_config


# ── tomograms ─────────────────────────────────────────────────────────────────

class TestTomograms:
    def test_tomogram_shape(self, emc_config):
        from emc3.tomograms import Tomograms
        c = emc_config['classes'][0]
        c['mapper'].load_coords(c['P_data'].mask)
        tomo = Tomograms(c['mapper'], c['model'], c['P_data'].C_i, c['fluence'])
        W = tomo.calculate_tomogram(0)
        I = c['P_data'].C_i.shape[0]
        assert W.shape == (I,), f"expected ({I},), got {W.shape}"

    def test_tomogram_non_negative(self, emc_config):
        from emc3.tomograms import Tomograms
        c = emc_config['classes'][0]
        c['mapper'].load_coords(c['P_data'].mask)
        tomo = Tomograms(c['mapper'], c['model'], c['P_data'].C_i, c['fluence'])
        W = tomo.calculate_tomogram(0)
        assert np.all(W >= 0), "tomogram has negative values"

    def test_wsums_positive(self, emc_config):
        from emc3.tomograms import Tomograms
        c = emc_config['classes'][0]
        c['mapper'].load_coords(c['P_data'].mask)
        tomo = Tomograms(c['mapper'], c['model'], c['P_data'].C_i, c['fluence'])
        wsums = tomo.calculate_wsums(chunksize=32)
        R = c['mapper'].shape[1]
        assert wsums.shape == (R,)
        assert np.all(wsums > 0), "wsums_r contains non-positive values"


# ── logR ──────────────────────────────────────────────────────────────────────

class TestLogR:
    def test_logR_shape(self, logR_result):
        for c in logR_result['classes']:
            D = c['P_data'].shape[0]
            R = c['mapper'].shape[1]
            assert c['logR_dr'].shape == (D, R), \
                f"expected ({D}, {R}), got {c['logR_dr'].shape}"

    def test_logR_finite(self, logR_result):
        for c in logR_result['classes']:
            assert np.all(np.isfinite(c['logR_dr'])), \
                f"class {c['class_id']}: logR_dr has non-finite values"

    def test_logR_varies_across_orientations(self, logR_result):
        """logR should not be constant — different orientations should score differently."""
        for c in logR_result['classes']:
            logR = c['logR_dr']
            std_per_frame = logR.std(axis=1)
            assert np.mean(std_per_frame) > 0, \
                "logR is identical for all orientations (suspicious)"


# ── probability from real logR ────────────────────────────────────────────────

class TestProbabilityFromLogR:
    def test_P_sums_to_one(self, logR_result):
        from emc3.probability import Probability

        config = logR_result
        R = config['R']
        D = config['classes'][0]['P_data'].shape[0]

        logR_dr = np.empty((D, R), dtype=np.float64, order='C')
        for c in config['classes']:
            r0 = c['r_offset']
            R_c = c['mapper'].shape[1]
            logR_dr[:, r0:r0 + R_c] = c['logR_dr'].astype(np.float64)

        P_dr = np.zeros((D, R), dtype=np.float64, order='C')
        Probability(D, R, beta=1.0).calculate(P_dr, logR_dr, 0, D)

        np.testing.assert_allclose(P_dr.sum(axis=1), 1.0, atol=1e-5)

    def test_P_non_negative(self, logR_result):
        from emc3.probability import Probability

        config = logR_result
        R = config['R']
        D = config['classes'][0]['P_data'].shape[0]

        logR_dr = np.empty((D, R), dtype=np.float64, order='C')
        for c in config['classes']:
            r0 = c['r_offset']
            R_c = c['mapper'].shape[1]
            logR_dr[:, r0:r0 + R_c] = c['logR_dr'].astype(np.float64)

        P_dr = np.zeros((D, R), dtype=np.float64, order='C')
        Probability(D, R, beta=1.0).calculate(P_dr, logR_dr, 0, D)

        assert np.all(P_dr >= 0)

    def test_higher_beta_gives_sharper_P(self, logR_result):
        from emc3.probability import Probability

        config = logR_result
        R = config['R']
        D = config['classes'][0]['P_data'].shape[0]

        logR_dr = np.empty((D, R), dtype=np.float64, order='C')
        for c in config['classes']:
            r0 = c['r_offset']
            R_c = c['mapper'].shape[1]
            logR_dr[:, r0:r0 + R_c] = c['logR_dr'].astype(np.float64)

        def entropy(P):
            return -np.sum(P * np.log(P + 1e-300), axis=1)

        P_low  = np.zeros((D, R), dtype=np.float64, order='C')
        P_high = np.zeros((D, R), dtype=np.float64, order='C')
        Probability(D, R, beta=0.1).calculate(P_low,  logR_dr.copy(), 0, D)
        Probability(D, R, beta=5.0).calculate(P_high, logR_dr.copy(), 0, D)

        assert entropy(P_high).mean() < entropy(P_low).mean()
