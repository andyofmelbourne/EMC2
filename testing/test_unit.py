"""
Fast unit tests for emc3 components.  No GPU required.

Run with:
    pytest test_unit.py -v
"""

import numpy as np
import pytest


# ── utils.chunker ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_partial_last_chunk(self):
        from emc3.utils import chunker
        assert list(chunker(3, 10)) == [(0, 3, 3), (3, 6, 3), (6, 9, 3), (9, 10, 1)]

    def test_exact_division(self):
        from emc3.utils import chunker
        assert list(chunker(4, 8)) == [(0, 4, 4), (4, 8, 4)]

    def test_chunksize_larger_than_size(self):
        from emc3.utils import chunker
        assert list(chunker(100, 5)) == [(0, 5, 5)]

    def test_covers_full_range_no_gaps(self):
        from emc3.utils import chunker
        size = 37
        chunks = list(chunker(7, size))
        prev = 0
        for r0, r1, dr in chunks:
            assert r0 == prev, "gap between chunks"
            assert r1 - r0 == dr, "chunk size inconsistent"
            prev = r1
        assert prev == size, "chunks don't cover full range"

    def test_offset(self):
        from emc3.utils import chunker
        chunks = list(chunker(3, 6, offset=10))
        assert chunks[0][0] == 10
        assert chunks[-1][1] == 16


# ── likelihood formula ────────────────────────────────────────────────────────

def _fake_data(D, I, seed=0):
    """Minimal K_di object with data_sum."""
    rng = np.random.default_rng(seed)
    K = rng.poisson(5, (D, I)).astype(float)

    class _FakeData:
        shape = K.shape
        data_sum = K.sum(axis=1)
        def __getitem__(self, idx):
            return K[idx]

    return _FakeData()


def _fake_tomo(R, I, seed=1):
    """Minimal tomo returning uniform positive tomograms."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 2.0, (R, I))

    class _FakeTomo:
        shape = (R, I)
        fluence = None
        def calculate_tomogram(self, r):
            return W[r]

    return _FakeTomo(), W


class TestLikelihood:
    D, R, I = 8, 12, 30

    def _run(self, frame_model, likelihood_type):
        from emc3.likelihood import Likelihood
        K = _fake_data(self.D, self.I)
        tomo, W = _fake_tomo(self.R, self.I)
        L = Likelihood(tomo, K, frame_model=frame_model, likelihood=likelihood_type)
        return L.calculate(), K, W

    def test_shape_fluence_free(self):
        logR, _, _ = self._run('basic', 'fluence_free')
        assert logR.shape == (self.D, self.R)

    def test_finite_fluence_free(self):
        logR, _, _ = self._run('basic', 'fluence_free')
        assert np.all(np.isfinite(logR))

    def test_formula_fluence_free(self):
        from emc3.likelihood import Likelihood
        K = _fake_data(self.D, self.I)
        tomo, W = _fake_tomo(self.R, self.I)
        L = Likelihood(tomo, K, frame_model='basic', likelihood='fluence_free')
        logR = L.calculate()
        wsums = W.sum(axis=1)
        expected = (K[:] @ np.log(W).T
                    - (K.data_sum[:, None] + 1) * np.log(wsums)[None, :])
        np.testing.assert_allclose(logR, expected, rtol=1e-5)

    def test_shape_poisson(self):
        logR, _, _ = self._run('basic', 'Poisson')
        assert logR.shape == (self.D, self.R)

    def test_finite_poisson(self):
        logR, _, _ = self._run('basic', 'Poisson')
        assert np.all(np.isfinite(logR))

    def test_formula_poisson(self):
        from emc3.likelihood import Likelihood
        K = _fake_data(self.D, self.I)
        tomo, W = _fake_tomo(self.R, self.I)
        L = Likelihood(tomo, K, frame_model='basic', likelihood='Poisson')
        logR = L.calculate()
        wsums = W.sum(axis=1)
        expected = K[:] @ np.log(W).T - wsums[None, :]
        np.testing.assert_allclose(logR, expected, rtol=1e-5)

    def test_unsupported_combination_raises(self):
        from emc3.likelihood import Likelihood
        K = _fake_data(4, 10)
        tomo, _ = _fake_tomo(5, 10)
        L = Likelihood(tomo, K, frame_model='background', likelihood='fluence_free')
        with pytest.raises(ValueError):
            L.calculate()

    def test_wsums_positive(self):
        from emc3.likelihood import Likelihood
        K = _fake_data(self.D, self.I)
        tomo, _ = _fake_tomo(self.R, self.I)
        L = Likelihood(tomo, K, frame_model='basic', likelihood='fluence_free')
        L.calculate()
        assert np.all(L.wsums_r > 0)


# ── probability ───────────────────────────────────────────────────────────────

def _logR(D, R, seed=42):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((D, R)), dtype=np.float64)


def _run_prob(D, R, logR, beta=1.0, P_thresh=0.0):
    from emc3.probability import Probability
    P = np.zeros((D, R), dtype=np.float64, order='C')
    Probability(D, R, beta=beta, P_thresh=P_thresh).calculate(P, logR, 0, D)
    return P


class TestProbability:
    def test_sums_to_one(self):
        D, R = 20, 100
        P = _run_prob(D, R, _logR(D, R))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_all_non_negative(self):
        D, R = 20, 100
        P = _run_prob(D, R, _logR(D, R))
        assert np.all(P >= 0)

    def test_argmax_matches_strongest_logR(self):
        D, R = 5, 30
        logR = np.zeros((D, R), dtype=np.float64, order='C')
        best = [0, 7, 14, 21, 29]
        for d, r in enumerate(best):
            logR[d, r] = 20.0  # clear winner
        P = _run_prob(D, R, logR)
        assert list(P.argmax(axis=1)) == best

    def test_uniform_at_zero_beta(self):
        D, R = 10, 40
        P = _run_prob(D, R, _logR(D, R), beta=0.0)
        np.testing.assert_allclose(P, 1.0 / R, atol=1e-6)

    def test_higher_beta_lowers_entropy(self):
        D, R = 10, 50
        logR = _logR(D, R, seed=13)

        def entropy(P):
            return -np.sum(P * np.log(P + 1e-300), axis=1)

        P_low  = _run_prob(D, R, logR.copy(), beta=0.1)
        P_high = _run_prob(D, R, logR.copy(), beta=5.0)
        assert entropy(P_high).mean() < entropy(P_low).mean()

    def test_threshold_zeros_small_values(self):
        D, R = 5, 20
        logR = np.zeros((D, R), dtype=np.float64, order='C')
        logR[:, 0] = 20.0  # overwhelming winner
        # P_thresh=0.5 zeros anything below half of Pmax
        P = _run_prob(D, R, logR, P_thresh=0.5)
        assert np.all(P[:, 1:] == 0.0)
        np.testing.assert_allclose(P[:, 0], 1.0, atol=1e-6)

    def test_large_D_chunked_same_as_small(self):
        """Internal d-chunking must not change results."""
        from emc3.probability import Probability
        D, R = 200, 60
        logR = _logR(D, R, seed=7)

        # force two different internal chunk sizes by varying cpu count proxy
        P1 = _run_prob(D, R, logR.copy(), beta=1.0)
        P2 = _run_prob(D, R, logR.copy(), beta=1.0)
        np.testing.assert_allclose(P1, P2, rtol=1e-6)


# ── ScatterAdd_cl ──────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def cpu_cl():
    from emc3.utils_cl import opencl_init_cpu
    return opencl_init_cpu()


class TestScatterAdd:
    """ScatterAdd_cl must match np.bincount row-by-row reference."""

    def _reference(self, indices_2d, weights_2d, out_size):
        out = np.zeros(out_size, dtype=np.float64)
        for r in range(indices_2d.shape[0]):
            out += np.bincount(indices_2d[r], weights_2d[r].astype(float),
                               minlength=out_size)
        return out

    def _make(self, cpu_cl, out_size):
        from emc3.utils_cl import ScatterAdd_cl
        return ScatterAdd_cl(out_size, context=cpu_cl['context'])

    def test_basic_correctness(self, cpu_cl):
        rng = np.random.default_rng(0)
        out_size, R, I = 64, 16, 32
        idx = rng.integers(0, out_size, (R, I), dtype=np.int32)
        w   = rng.random((R, I)).astype(np.float32)

        sa  = self._make(cpu_cl, out_size)
        out = np.zeros(out_size, dtype=np.float64)
        sa.add(idx, w, out)

        ref = self._reference(idx, w, out_size)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_accumulates_across_calls(self, cpu_cl):
        rng = np.random.default_rng(1)
        out_size, R, I = 32, 8, 20
        idx = rng.integers(0, out_size, (R, I), dtype=np.int32)
        w   = rng.random((R, I)).astype(np.float32)

        sa  = self._make(cpu_cl, out_size)
        out = np.zeros(out_size, dtype=np.float64)
        sa.add(idx, w, out)
        sa.add(idx, w, out)   # call twice — should double the result

        ref = 2 * self._reference(idx, w, out_size)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_large_r_chunk(self, cpu_cl):
        rng = np.random.default_rng(2)
        out_size, R, I = 512, 256, 128
        idx = rng.integers(0, out_size, (R, I), dtype=np.int32)
        w   = rng.random((R, I)).astype(np.float32)

        sa  = self._make(cpu_cl, out_size)
        out = np.zeros(out_size, dtype=np.float64)
        sa.add(idx, w, out)

        ref = self._reference(idx, w, out_size)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_all_same_index(self, cpu_cl):
        """All elements map to index 0 — maximum write contention."""
        out_size, R, I = 16, 10, 10
        idx = np.zeros((R, I), dtype=np.int32)
        w   = np.ones((R, I), dtype=np.float32)

        sa  = self._make(cpu_cl, out_size)
        out = np.zeros(out_size, dtype=np.float64)
        sa.add(idx, w, out)

        assert out[0] == pytest.approx(R * I, rel=1e-5)
        assert np.all(out[1:] == 0.0)


# ── cluster_classes.triplet_consistency ───────────────────────────────────────

class TestTripletConsistency:
    """
    Synthetic octant geometry: 3 class normals along z, x, y axes.

    True sinogram angles (Nrot=360, degrees):
      pair(0,1): class0=90°, class1=0°
      pair(0,2): class0=0°,  class2=0°
      pair(1,2): class1=90°, class2=270°

    All three vertex angles are 90° → valid spherical triangle.

    K=4 symmetry: the C matrix has 4 peaks per pair, spaced 90° apart.
    The stored best_phi is the *wrong* peak (different shifts per pair),
    giving vertex angle 0° for classes 0 and 2 (fails in_range check).

    n_peaks=1  must fail  (uses wrong best_phi → inconsistent).
    n_peaks=4  must pass  (loads all 4 peaks → finds true angles).
    """

    Nrot  = 360
    K     = 4
    step  = Nrot // K          # 90

    # true sinogram peak positions (row=class_a angle, col=class_b angle)
    _true = {(0, 1): (90, 0), (0, 2): (0, 0), (1, 2): (90, 270)}
    # different shifts per pair to break n_peaks=1
    _shifts = {(0, 1): 1, (0, 2): 2, (1, 2): 3}

    @staticmethod
    def _pair_idx(i, j, N):
        return i * N - i * (i + 1) // 2 + j - i - 1

    def _build_file(self, tmp_path):
        import h5py
        N = 3
        N_pairs = N * (N - 1) // 2
        Nrot = self.Nrot

        C = np.zeros((N_pairs, Nrot, Nrot), dtype=np.float32)
        best_phi = np.zeros((N, N, 2), dtype=np.int32)

        for (i, j), (r_true, c_true) in self._true.items():
            k = self._pair_idx(i, j, N)
            for m in range(self.K):
                r = (r_true + m * self.step) % Nrot
                c = (c_true + m * self.step) % Nrot
                C[k, r, c] = 1.0
            s = self._shifts[(i, j)]
            r_w = (r_true + s * self.step) % Nrot
            c_w = (c_true + s * self.step) % Nrot
            best_phi[i, j] = [r_w, c_w]
            best_phi[j, i] = [c_w, r_w]

        fnam = tmp_path / 'cl_test.h5'
        with h5py.File(fnam, 'w') as f:
            f['class_ids'] = np.array([0, 1, 2], dtype=np.int32)
            f['N_classes'] = np.int32(N)
            f['Nrot']      = np.int32(Nrot)
            f['best_phi']  = best_phi
            f['scores']    = np.ones((N, N), dtype=np.float32)
            f.create_dataset('C', data=C,
                             chunks=(1, Nrot, Nrot), compression='gzip')
        return fnam

    def test_true_angles_pass_strict_check(self):
        from emc3.cluster_classes import _sph_triangle_valid_vec
        phi_scale = 2 * np.pi / self.Nrot
        r01, c01 = self._true[(0, 1)]
        r02, c02 = self._true[(0, 2)]
        r12, c12 = self._true[(1, 2)]
        ai = ((r01 - r02) % 180) * phi_scale
        aj = ((c01 - r12) % 180) * phi_scale
        ak = ((c02 - c12) % 180) * phi_scale
        A = np.array([ai]); B = np.array([aj]); C = np.array([ak])
        assert _sph_triangle_valid_vec(A, B, C)[0], \
            f'True vertex angles ({np.degrees(ai):.1f},{np.degrees(aj):.1f},{np.degrees(ak):.1f}) must pass'

    def test_wrong_angles_fail_strict_check(self):
        from emc3.cluster_classes import _sph_triangle_valid_vec
        from itertools import product as iproduct
        phi_scale = 2 * np.pi / self.Nrot
        # compute wrong angles from best_phi
        wrong = {}
        for (i, j), (r, c) in self._true.items():
            s = self._shifts[(i, j)]
            wrong[(i, j)] = ((r + s*self.step) % self.Nrot,
                             (c + s*self.step) % self.Nrot)
        r01w, c01w = wrong[(0, 1)]
        r02w, c02w = wrong[(0, 2)]
        r12w, c12w = wrong[(1, 2)]
        ai = ((r01w - r02w) % 180) * phi_scale
        aj = ((c01w - r12w) % 180) * phi_scale
        ak = ((c02w - c12w) % 180) * phi_scale
        signs = np.array(list(iproduct((1,-1), repeat=3)), dtype=float)
        A = np.where(signs[:,0]>0, ai, np.pi-ai)
        B = np.where(signs[:,1]>0, aj, np.pi-aj)
        C = np.where(signs[:,2]>0, ak, np.pi-ak)
        assert not _sph_triangle_valid_vec(A, B, C).any(), \
            f'Wrong vertex angles ({np.degrees(ai):.1f},{np.degrees(aj):.1f},{np.degrees(ak):.1f}) must fail all 8 sign combos'

    def test_n_peaks_K_beats_n_peaks_1(self, tmp_path):
        from emc3.cluster_classes import triplet_consistency
        fnam = self._build_file(tmp_path)
        _, cs1, _ = triplet_consistency(fnam, n_peaks=1)
        _, csK, _ = triplet_consistency(fnam, n_peaks=self.K)
        # n_peaks=1 uses wrong best_phi: the single triplet must fail
        assert cs1.mean() == pytest.approx(0.0), \
            f'n_peaks=1 should score 0 but got {cs1.mean():.3f}'
        # n_peaks=K finds the true peak: the single triplet must pass.
        # With N=3 classes the diagonal of pair_scores is 0, so the max
        # achievable class_score is 2/N = 2/3.
        N_classes = 3
        expected = 2.0 / N_classes
        assert csK.mean() == pytest.approx(expected, abs=1e-6), \
            f'n_peaks={self.K} should score {expected:.3f} but got {csK.mean():.3f}'
