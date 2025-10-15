"""
Wsum_r = sum_i C_i W_ri

Basic:
    N_ri  = sum_d P_dr K_di
    D_ri  = C_i sum_d P_dr

Fluence:
    N_ri  = sum_d P_dr K_di
    D_r   = C_i sum_d w_d P_dr

    a_d   = sum_i K_di
    b_d   = sum_r P_dr wsum_r
    w'_d  = a_d / b_d

Fluence free:
    N_ri = wsum_r sum_d P_dr K_di
    D_ri = C_i sum_d P_dr K_d

------------------------------------

merge tomograms (then I):
    W'_ri = N_ri / D_ri

    A_n = sum_ri M^-1(W'_ri, r, i)_n
    B_n = sum_ri M^-1(1,     r, i)_n
    I_n = A_n / B_n

merge I:
    A_n = sum_ri M^-1(N_ri, r, i)_n
    B_n = sum_ri M^-1(D_ri, r, i)_n
    I_n = A_n / B_n

Loop over model class (this reduces out-of-order memory operations)
Would be nice to skip frames with low P_dr values but this complicates
dot product which is about 40 times faster

Perform coordinate mappings on gpu
nearest: i --> n
    n0 = round(i0 + (R_r . q_i)_0 / dq)
    n1 = round(i0 + (R_r . q_i)_1 / dq)
    n2 = round(i0 + (R_r . q_i)_2 / dq)

    n = N^2 n0 + N n1 + n2

    M^-1(x, r, i)_m = x delta(n - m)

Perform sum in I on cpu (out-of-order memory operations)

There are three main computational steps:
    1. calculate D_ri
    2. calculate P . K
    3. calculate voxel mapping n_sri
    4. merge N and D (in tomos or model space)
    5. apply symmetry

If I had a faster way to share / pipe I would split
these into separate processes for optimal load balancing

Inputs:
------
    P_dr normalised probability matrix
    K_di photon counts
    C_i  polarisation & solid angle correction

    Depending on algorithm:
        w_d  relative fluence per frame
        wsums_r tomogram sums within data mask

    input or calculate:
        M_sri mapping between pixel and model voxel
"""

import numpy as np
import h5py
from tqdm import tqdm
from time import time

from . import symmetry
from . import utils
from . import utils_cl
from .tomograms import Tomograms, Tomograms_cl
from .mapper import Mapper_cl
from .dot import AdotB


class Update_model_class():
    """
    chunked:
    3. calculate D_ri
    4. calculate P . K
    5. calculate voxel mapping n_sri
    6. merge N and D (in tomos or model space)
    7. apply symmetry
    """

    def __init__(self, w_d, c):
        self.wsums_r = c['wsums_r']
        self.w_d = w_d
        self.C_i = c['data'].C_i
        self.P_dr = c['P_dr']
        self.K_d = c['data'].data_sum
        self.K_di = c['data']
        self.maximise = c['maximise']
        self.model = c['model']

        # determine offset calculation
        # ----------------------------
        op = (c['likelihood'], c['frame_model'])

        if op == ('Poisson', 'basic'):
            self._calc_D = self._calc_CP

        elif op == ('Poisson', 'fluence'):
            self._calc_D = self._calc_CwP

        elif op == ('Poisson_fluence_free', 'basic'):
            self._calc_D = self._calc_CkP

        else:
            err = f"{c['likelihood']=} {c['frame_model']=} not supported"
            raise ValueError(err)

        # set up dot product calculation N_ri = sum_d P_dr K_di
        # N_ri = np.dot(P_dr.T, c['data'][:])
        # N_ri = np.dot(P_dr.T, c['data'][:])
        self.PdotK = AdotB(self.P_dr.T, self.K_di)

        self.r_chunk_size = self.PdotK.M_chunksize
        self.R = self.P_dr.shape[1]

        # prepare mapping operations
        cl = utils_cl.opencl_init()

        self.mapper_cl = Mapper_cl(
                c['mapper'],
                cl['context'],
                cl['queue'])

        self.mapper_cl.load_buffers(
                r_chunk_size=self.r_chunk_size,
                ravel=True)


    def calculate(self):
        # ------------------
        # 3. calculate P . K
        # ------------------
        t0 = time()
        self.N_ri = self.PdotK()
        print(f'P dot K time:', time() - t0)
        # return self.N_ri

        # -----------------
        # 4. calculate D_ri
        # -----------------
        t0 = time()
        D_ri = self._calc_D()
        # print(f'D_ri time:', time() - t0)

        if self.maximise == 'W':
            D_ri[D_ri == 0] = 1.
            self.N_ri /= D_ri
            D_ri[:] = 1.

        # --------------------------------
        # 5. calculate voxel mapping n_sri
        # --------------------------------
        R = self.R
        N_n = np.zeros(self.model.size, dtype=float)
        D_n = np.zeros(self.model.size, dtype=float)

        mtime = 0.
        btime = 0.

        for s in range(self.mapper_cl.shape[0]):
            t0 = time()
            n_sri = self.mapper_cl.calculate_mapping(
                    s, 0, self.R, ravel=True, cpu=True)
            mtime += time() - t0

            t0 = time()
            for s in range(n_sri.shape[0]):
                for r in range(n_sri.shape[1]):
                    N_n += np.bincount(
                        n_sri[s, r],
                        self.N_ri[r],
                        minlength=self.model.size
                    )

                    D_n += np.bincount(
                        n_sri[s, r],
                        D_ri[r],
                        minlength=self.model.size
                    )
            btime += time() - t0

        # print(f'mapping time:', mtime)
        # print(f'bincount time:', btime)

        sym = symmetry.Symmetry(
                self.model.i0,
                self.model.shape,
                self.model.symmetry
        )

        N_n = sym.apply_symmetry(
            N_n.reshape(self.model.shape),
        )

        D_n = sym.apply_symmetry(
            D_n.reshape(self.model.shape),
        )

        # I = N / D
        D_n[D_n == 0] = 1.
        N_n /= D_n
        return N_n


    def _calc_CP(self):
        D_ri = self.C_i[None, :] * \
                np.sum(self.P_dr, axis=0)[:, None]
        return D_ri


    def _calc_CwP(self):
        D_ri = self.C_i[None, :] * np.dot(self.w_d, self.P_dr)[:, None]
        return D_ri


    def _calc_CkP(self):
        self.N_ri *= self.wsums_r[:, None]
        D_ri = self.C_i[None, :] * \
            np.dot(self.K_d, self.P_dr)[:, None]
        return D_ri



def update_model_basic(config):
    """
    cpu all in memory update

    1. calculate wsums_r (if needed)
    2. calculate w_d     (if needed)

    chunked:
    3. calculate D_ri
    4. calculate P . K
    5. calculate voxel mapping n_sri
    6. merge N and D (in tomos or model space)
    7. apply symmetry
    """
    t0 = time()
    for c in config['classes']:
        # calculate tomogram sums if needed
        c['mapper'].load_coords(c['data'].mask)
        c['wsums_r'] = calculate_wsums_cl(**c)
    print(f'tomo time:', time() - t0)

    # calculate fluence if needed
    t0 = time()
    w_d = calculate_fluence(config)
    print(f'fluence time:', time() - t0)

    t0 = time()
    for c in config['classes']:
        mupdate = Update_model_class(w_d, c)

        I = mupdate.calculate()

        c['model'].data = I
    print(f'update time:', time() - t0)


def calculate_fluence(config):
    D = config['classes'][0]['data'].shape[0]
    K_d = config['classes'][0]['data'].data_sum

    w_d = np.zeros((D,), dtype=float)

    for c in config['classes']:
        w_d += np.dot(c['P_dr'], c['wsums_r'])

    w_d = K_d / w_d
    return w_d



def calculate_wsums(
        likelihood=None,
        frame_model=None,
        mapper=None,
        model=None,
        data=None,
        **kwargs
        ):

    if not (likelihood == 'Poisson' and frame_model == 'basic'):
        tomos = Tomograms(
                mapper,
                model,
                data.C_i)

        wsums_r = tomos.calculate_wsums()

    else:
        wsums_r = None

    return wsums_r


def calculate_wsums_cl(
        likelihood=None,
        frame_model=None,
        mapper=None,
        model=None,
        data=None,
        **kwargs
        ):

    cl = utils_cl.opencl_init()


    if not (likelihood == 'Poisson' and frame_model == 'basic'):
        tomos = Tomograms(
                mapper,
                model,
                data.C_i)

        tomos_cl = Tomograms_cl(tomos, cl['context'], cl['queue'])

        wsums_r = tomos_cl.calculate_wsums()

    else:
        wsums_r = None

    return wsums_r
