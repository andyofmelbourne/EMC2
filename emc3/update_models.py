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

import pyopencl as cl
import pyopencl.array as cl_array
import pyclblast

import warnings, os
# warnings.filterwarnings("ignore", category=cl.CompilerWarning)
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def gpu_dot(A, B, queue, a_transp=False, b_transp=False):
    if a_transp:
        k, m = A.shape
        a_ld = m
    else:
        m, k = A.shape
        a_ld = k

    if b_transp:
        n = B.shape[0]
        b_ld = k
    else:
        n = B.shape[1]
        b_ld = n

    c_ld = n

    assert(A.flags.c_contiguous)
    assert(B.flags.c_contiguous)

    assert(np.issubdtype(A.dtype, np.float32))
    assert(np.issubdtype(B.dtype, np.float32))

    C = np.zeros((m, n), dtype=np.float32)
    A_dev = cl_array.to_device(queue, A)      # cl_array.Array; has .dtype
    B_dev = cl_array.to_device(queue, B)
    C_dev = cl_array.to_device(queue, C)      # or cl_array.empty(queue, (m,n), dtype=np.float32)

    # Call pyclblast.gemm using your version's signature:
    # gemm(queue, m, n, k, a, b, c, a_ld, b_ld, c_ld, ...)
    evt = pyclblast.gemm(
        queue,
        m, n, k, # (m, n) = (m, k) . (k, n)
        A_dev, B_dev, C_dev,
        a_ld, b_ld, c_ld, # a_ld = k, b_ld = n, c_ld = n (row-major)
        alpha=1.0,
        beta=0.0,
        a_transp=a_transp,
        b_transp=b_transp
    )

    # evt.wait()
    # C_res = C_dev.get()
    return C_dev, evt



class Update_model_class():
    """
    chunked:
    3. calculate D_ri
    4. calculate P . K
    5. calculate voxel mapping n_sri
    6. merge N and D (in tomos or model space)
    7. apply symmetry
    """

    def __init__(self, w_d, c, cl=None):
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
        # self.PdotK = AdotB(self.P_dr.T, self.K_di)
        # self.r_chunk_size = self.PdotK.M_chunksize
        self.R = self.P_dr.shape[1]
        self.r_chunk_size = self.R

        # prepare mapping operations
        if cl is None:
            cl = utils_cl.opencl_init()

        self.queue = cl['queue']

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
        # self.N_ri = self.PdotK()

        N_ri_dev, evt = gpu_dot(
            self.P_dr.astype(np.float32),
            self.K_di[:].astype(np.float32),
            self.queue, a_transp=True, b_transp=False)
        evt.wait()
        self.N_ri = N_ri_dev.get()
        # assert(np.allclose(N_ri, self.N_ri))
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
    cpu all in memory update (in place operation on config)

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
        cl=None,
        **kwargs
        ):

    if cl is None:
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

    # class ids
    cids = [i for i in range(len(config['classes'])) if config['classes'][i]['update_model']]

    devices = utils_cl.get_devices(device_type='gpu')

    # number parallel processes
    nproc = p_per_device * len(devices)

    # make cids_str = '0,1,2 3,4,5 6,7,8 9'
    cids_str = []
    for s in np.array_split(cids, nproc):
        cids_str.append(','.join(s.astype(str)))
    cids_str = ' '.join(cids_str)

    cmd = f"time parallel --halt now,fail=1 python -m emc3.update_models {config_file} {{}} {{%}} ::: {cids_str}"
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
    class_ids = [int(c) for c in sys.argv[2].split(',')]
    device = int(sys.argv[3])

    config = pickle.load(open(config_fnam, 'rb'))
    wd = config['working_directory']

    # load fluence
    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]

    cl = utils_cl.opencl_init(device_no=device)

    for class_id in class_ids:
        c = config['classes'][class_id]

        if not c['update_logR']:
            sys.exit()

        # load data
        c['data'].load_from_file()

        # load prob
        with h5py.File(c['probability_matrix_file']) as f:
            c['P_dr'] = f['P_dr'][()]

        # load wsums
        with h5py.File(c['wsums_file']) as f:
            c['wsums_r'] = f['wsums_r'][()]

        c['mapper'].load_coords(c['data'].mask)

        t0 = time()
        mupdate = Update_model_class(w_d, c, cl)

        I = mupdate.calculate()

        c['model'].data = I
        print(f'update time:', time() - t0)

        # save
        with h5py.File(c['model_file'], 'w') as f:
            f['data'] = c['model'].data
            f['dq'] = c['model'].dq

