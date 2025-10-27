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
from pathlib import Path

from . import symmetry
from . import utils
from . import utils_cl
from .tomograms import Tomograms, Tomograms_cl
from .mapper import Mapper_cl
from .dot import AdotB

from .update_models import *

import pyopencl as cl
import pyopencl.array as cl_array
import pyclblast

import warnings, os
# warnings.filterwarnings("ignore", category=cl.CompilerWarning)
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


class Update_model_class():
    """
    chunked:
    3. calculate D_ri
    4. calculate P . K
    5. calculate voxel mapping n_sri
    6. merge N and D (in tomos or model space)
    7. apply symmetry
    """

    def __init__(self, w_d, c, cl=None,
            r_chunk_size=1024, d_chunk_size=1024,
            r0=None, r1=None):

        self.R = c['wsums_r'].shape[0]
        self.R0 = self.R

        if r0 is None:
            r0 = 0

        if r1 is None:
            r1 = self.R

        r1 = min(self.R, r1)
        r0 = max(0, r0)
        self.r0 = r0
        self.r1 = r1
        self.R = r1 - r0

        self.wsums_r = c['wsums_r'][r0: r1]
        self.w_d = w_d
        self.C_i = c['data'].C_i
        self.P_dr = c['P_dr'].astype(np.float32)
        self.K_d = c['data'].data_sum
        self.K_di = c['data']
        self.maximise = c['maximise']
        self.model = c['model']
        self.class_id = c['class_id']
        self.filter = c.get('filter_model', None)

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
        # self.PdotK = AdotB(self.P_dr.T, self.K_di)
        # self.r_chunk_size = self.PdotK.M_chunksize
        self.D = self.K_di.shape[0]
        self.I = self.K_di.shape[1]
        self.r_chunk_size = min(r_chunk_size, self.R)
        self.d_chunk_size = min(d_chunk_size, self.D)

        self.cl = cl
        self.mapper = c['mapper']

    def calculate(self, Pmin=1e-5):
        """
        ignore slices (r's) with no frames above Pmin
        """
        # prepare mapping operations
        if self.cl is None:
            self.cl = utils_cl.opencl_init()

        self.queue = self.cl['queue']

        self.mapper_cl = Mapper_cl(
                self.mapper,
                self.cl['context'],
                self.cl['queue'])

        self.mapper_cl.load_buffers(
                r_chunk_size=self.r_chunk_size,
                ravel=True)

        N_n = np.zeros(self.model.size, dtype=float)
        D_n = np.zeros(self.model.size, dtype=float)
        mtime = 0.
        btime = 0.
        for r0, r1, dr in utils.chunker(self.r_chunk_size, self.R):
            # ------------------
            # 3. calculate P . K
            # ------------------
            self.N_ri = np.zeros((dr, self.I), dtype=np.float32)
            for d0, d1, dr in utils.chunker(self.d_chunk_size, self.D):
                # t0 = time()
                # self.N_ri = self.PdotK()
                P_dr = self.P_dr[d0:d1, r0:r1].astype(np.float32)
                K_di = self.K_di[d0:d1, :].astype(np.float32)

                N_ri_dev, evt = gpu_dot(
                    P_dr,
                    K_di,
                    self.queue, a_transp=True, b_transp=False)
                evt.wait()
                self.N_ri += N_ri_dev.get()
                # assert(np.allclose(N_ri, self.N_ri))
                # print(f'P dot K time:', time() - t0)
                # return self.N_ri

            # -----------------
            # 4. calculate D_ri
            # -----------------
            # t0 = time()
            D_ri = self._calc_D(r0, r1)
            # print(f'D_ri time:', time() - t0)

            if self.maximise == 'W':
                D_ri[D_ri == 0] = 1.
                self.N_ri /= D_ri
                D_ri[:] = 1.

            # --------------------------------
            # 5. calculate voxel mapping n_sri
            # --------------------------------
            for s in range(self.mapper_cl.shape[0]):
                t0 = time()
                n_sri = self.mapper_cl.calculate_mapping(
                            s, r0+self.r0, r1+self.r0, ravel=True, cpu=True)
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

        # save chunk
        fnam = f'class_model_chunk_{self.class_id}_{self.r0}_{self.r1}.h5'
        with h5py.File(fnam, 'w') as f:
            f['N_n'] = N_n
            f['D_n'] = D_n

    def finish(self):
        # load chunks
        fnams = Path('./').glob(f'class_model_chunk_{self.class_id}_*.h5')

        N_n = np.zeros(self.model.size, dtype=float)
        D_n = np.zeros(self.model.size, dtype=float)

        for fnam in fnams:
            with h5py.File(str(fnam)) as f:
                N_n += f['N_n'][()]
                D_n += f['D_n'][()]

            # delete
            fnam.unlink()

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

        m = D_n == 0
        D_n[m] = 1.
        N_n /= D_n

        if self.filter is not None:
            if self.model.data is not None:
                if self.model.data.shape == N_n.shape:
                    N_n[m] = self.model.data[m]

            N_n = self.apply_filter(N_n, self.filter)

        return N_n

    def apply_filter(self, I_n, size):
        """
        apply soft Fourier low-pass filter
        with 2xsize width
        """
        N = I_n.shape[0]
        vox_size = 1 / (N * self.model.dq)

        inds = np.indices(I_n.shape)
        inds -= I_n.shape[0]//2
        axes = tuple([i+1 for i in range(len(I_n.shape))])
        inds = np.fft.ifftshift(inds, axes=axes)

        r = np.sum(inds.astype(float)**2, axis=0)**0.5
        rmax = min(size / vox_size, N//2-1)
        sig = (N//2 - rmax) / 4.

        filter = np.ones(r.shape, dtype=float)
        m = r > rmax
        filter[m] = np.exp(-(r[m]-rmax)**2 / (2 * sig**2))

        Ih_n = np.fft.ifftn(np.fft.ifftshift(I_n)) * filter
        out = np.fft.fftshift(np.fft.fftn(Ih_n))
        return np.clip(out.real, 0, None)

    def _calc_CP(self, r0=0, r1=None):
        if r1 is None:
            r1 = self.R

        D_ri = self.C_i[None, :] * \
                np.sum(self.P_dr[:, r0:r1], axis=0)[:, None]
        return D_ri


    def _calc_CwP(self, r0=0, r1=None):
        if r1 is None:
            r1 = self.R

        D_ri = self.C_i[None, :] * np.dot(self.w_d, self.P_dr[:, r0:r1])[:, None]
        return D_ri


    def _calc_CkP(self, r0=0, r1=None):
        if r1 is None:
            r1 = self.R

        self.N_ri *= self.wsums_r[r0:r1, None]
        D_ri = self.C_i[None, :] * \
            np.dot(self.K_d, self.P_dr[:, r0:r1])[:, None]
        return D_ri



def update_model_subprocess(config_file, config, p_per_device=2):
    """
    calculate tomogram sums and fluence in this process (might par. later)
    then farm off model update to subprocesses
    """
    import subprocess, sys
    from . import utils_cl, utils

    cl = utils_cl.opencl_init()

    # calculate tomogram sums if needed
    # ---------------------------------
    t0 = time()
    for c in config['classes']:
        # load model
        if c['model'].data is None:
            with h5py.File(c['model_file']) as f:
                c['model'].data = f['data'][()]

        c['mapper'].load_coords(c['data'].mask)
        c['wsums_r'] = calculate_wsums_cl(**c, cl=cl)

        # write to file
        with h5py.File(c['wsums_file'], 'w') as f:
            f['wsums_r'] = c['wsums_r']

    print(f'tomo time:', time() - t0)

    # calculate fluence if needed
    # ---------------------------
    t0 = time()
    # load probability matrix
    for c in config['classes']:
        fnam = c['probability_matrix_file']
        with h5py.File(fnam) as f:
            c['P_dr'] = f['P_dr'][()]

    w_d = calculate_fluence(config)

    # write to file
    with h5py.File(config['fluence_file'], 'w') as f:
        f['w_d'] = w_d
    print(f'fluence time:', time() - t0)

    # delete P_dr to save space
    for c in config['classes']:
        del c['P_dr']
        c['P_dr'] = None

    if not c['update_model']:
        return None

    # class ids
    cids = [i for i in range(len(config['classes'])) if config['classes'][i]['update_model']]

    assert(len(cids) == 1)

    devices = utils_cl.get_devices(device_type='gpu')

    # number parallel processes
    nproc = p_per_device * len(devices)

    # split calculation by r-chunks
    # str = '0-10000 10000-20000 20000-20323'
    R = c['mapper'].shape[1]
    r0_r1_str = []
    for s in np.array_split(np.arange(0, R), nproc):
        r0_r1_str.append('-'.join([str(s[0]), str(s[-1]+1)]))
    r0_r1_str = ' '.join(r0_r1_str)

    cmd = f"time parallel --halt now,fail=1 python -m emc3.update_models_single {config_file} {{}} {{%}} ::: {r0_r1_str}"
    print(cmd)
    p = subprocess.Popen(
        cmd,
        text=True,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
        )
    p.wait()

    if p.returncode != 0:
        raise ValueError('something went wrong with call')

    # now finish the job
    c = config['classes'][0]

    # load data
    c['data'].load_from_file()

    # load model
    with h5py.File(c['model_file']) as f:
        c['model'].data = f['data'][()]

    # load prob
    with h5py.File(c['probability_matrix_file']) as f:
        c['P_dr'] = f['P_dr'][()]

    # load wsums
    with h5py.File(c['wsums_file']) as f:
        c['wsums_r'] = f['wsums_r'][()]

    c['mapper'].load_coords(c['data'].mask)

    t0 = time()
    mupdate = Update_model_class(w_d, c, cl)

    I = mupdate.finish()

    c['model'].data = I
    print(f'update time:', time() - t0)

    # save
    with h5py.File(c['model_file'], 'w') as f:
        f['data'] = c['model'].data
        f['dq'] = c['model'].dq

    return p


"""
load config
calculate logR for selected classes
    this helps to reduce io and initialisation overhead
write to file

use platform and device specified on command line for multi gpu
"""
if __name__ == '__main__':
    import sys, pickle, h5py

    config_fnam = sys.argv[1]
    r0, r1 = [int(c) for c in sys.argv[2].split('-')]
    device = int(sys.argv[3])

    config = pickle.load(open(config_fnam, 'rb'))
    wd = config['working_directory']

    # load fluence
    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]

    cl = utils_cl.opencl_init(device_no=device)

    for class_id in [0,]:
        c = config['classes'][class_id]

        if not c['update_model']:
            continue

        # load data
        c['data'].load_from_file()

        # load model
        with h5py.File(c['model_file']) as f:
            c['model'].data = f['data'][()]

        # load prob
        with h5py.File(c['probability_matrix_file']) as f:
            c['P_dr'] = f['P_dr'][:, r0:r1]

        # load wsums
        with h5py.File(c['wsums_file']) as f:
            c['wsums_r'] = f['wsums_r'][()]

        c['mapper'].load_coords(c['data'].mask)

        t0 = time()
        mupdate = Update_model_class(w_d, c, cl, r0=r0, r1=r1)

        # calculate chunk
        mupdate.calculate()

        print(f'update time:', time() - t0)
