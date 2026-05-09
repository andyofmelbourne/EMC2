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
from . import profiling
from .tomograms import Tomograms, Tomograms_cl
from .mapper import Mapper_cl
from .dot import AdotB

import pyopencl as cl
import pyopencl.array as cl_array
import pyclblast

import os



def models_no_nan(I_n):
    """
    ensure non-negativity and finite values
    """
    I_n = np.nan_to_num(I_n, copy=False, nan=1e-8)
    I_n = np.clip(I_n, 0, None)
    return I_n

def models_apply_sym(N_in_n, D_in_n, model, is_asymmetric_unit=False):
    """
    apply symmetry to numerator (N_n) and denominator (D_n)
    then divide the two unless the symmeterised D_n == 0
    """
    sym = symmetry.Symmetry(
        model.i0,
        model.shape,
        model.symmetry
    )

    if is_asymmetric_unit:
        n_asy = sym.get_asymmetric_unit()
        N_n = np.zeros(np.prod(model.shape), dtype=float)
        N_n[n_asy] = N_in_n

        D_n = np.zeros_like(N_n)
        D_n[n_asy] = D_in_n
    else:
        N_n = N_in_n
        D_n = D_in_n

    N_n = sym.apply_symmetry(
        N_n.reshape(model.shape),
    )

    D_n = sym.apply_symmetry(
        D_n.reshape(model.shape),
    )

    # I = N / D

    # in case background subtraction
    # gives -ve's
    N_n = np.clip(N_n, 0, None)

    N_sym = N_n.copy()
    D_sym = D_n.copy()

    m = D_n == 0
    D_n[m] = 1.
    N_n /= D_n
    return N_n, N_sym, D_sym


def apply_filter(I_n, I0_n, dq, filter_size):
    m = I_n == 0

    if filter_size:
        t = I_n.copy()
        if I0_n is not None:
            if I0_n.shape == N_n.shape:
                t[m] = I0_n[m]

        for i in range(3):
            t[~m] = I_n[~m]
            t = _apply_filter(dq, t, filter_size)

        I_n = t
    return I_n

def model_rms(I_n, I0_n):
    if I0_n is not None:
        rms = np.mean((I0_n - I_n)**2)**0.5
    return rms

def limit_change(I_n, I0_n, max_change):
    """
    out = I0_n + max_change (I_n - I0_n)
    """
    if I0_n is None or not max_change:
        return I_n

    return I0_n + max_change * (I_n - I0_n)


def finish_model(N_n, D_n, c, profiling, is_asymmetric_unit=False):
    """
    1. apply voxel perfect symmetry
    2. filter with autoc
    3. ensure non-negative finite
    4. get rms
    5. restrict change
    """
    filter_model = c.get('filter_model', None)
    model_max_change = c.get('model_max_change', None)
    model = c['model']

    # get old model if shape and dq match
    # -----------------------------------
    with h5py.File(c['model_file']) as f:
        data = f['data']
        dq = f['dq'][()]
        I0_n = None
        if data.shape == model.shape:
            if dq == model.dq:
                I0_n = data[()]

    N_n = models_no_nan(N_n)
    D_n = models_no_nan(D_n)

    I_n, N_sym, D_sym = models_apply_sym(N_n, D_n, model, is_asymmetric_unit)
    I_n = apply_filter(I_n, I0_n, model.dq, filter_model)
    I_n = limit_change(I_n, I0_n, model_max_change)
    rms = model_rms(I_n, I0_n)

    cid = c['class_id']
    print(f'rms difference for model {cid}: {rms}')
    print(f'{cid}: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    # save model to disk
    with profiling.cl_timed('io:save_model'):
        with h5py.File(c['model_file'], 'w') as f:
            f['data'] = I_n
            f['dq'] = c['model'].dq
            f['N_n'] = N_sym
            f['D_n'] = D_sym

    return I_n



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

    with profiling.cl_timed('h2d', nbytes=A.nbytes + B.nbytes):
        A_dev = cl_array.to_device(queue, A)
        B_dev = cl_array.to_device(queue, B)
        C_dev = cl_array.to_device(queue, C)
        queue.finish()

    with profiling.cl_timed('gemm', m=m, n=n, k=k):
        evt = pyclblast.gemm(
            queue,
            m, n, k,
            A_dev, B_dev, C_dev,
            a_ld, b_ld, c_ld,
            alpha=1.0,
            beta=0.0,
            a_transp=a_transp,
            b_transp=b_transp
        )
        evt.wait()

    return C_dev, evt


def _apply_filter(dq, I_n, size):
    """
    apply soft Fourier low-pass filter
    with 2xsize width
    """
    N = I_n.shape[0]
    vox_size = 1 / (N * dq)

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


class Update_model_class():
    """
    chunked:
    3. calculate D_ri
    4. calculate P . K
    5. calculate voxel mapping n_sri
    6. merge N and D (in tomos or model space)
    7. apply symmetry
    """

    def __init__(self, w_d, c, cl=None, cl_cpu=None, r_chunk_size=1024, d_chunk_size=1024):
        self.wsums_r = c['wsums_r']
        self.w_d = w_d
        self.C_i = c['data'].C_i
        self.P_dr = c['P_dr']
        self.K_d = c['data'].data_sum
        self.K_di = c['data']
        self.maximise = c['maximise']
        self.model = c['model']
        self.filter = c.get('filter_model', None)
        self.c = c

        # determine offset calculation
        # ----------------------------
        op = (c['likelihood'], c['frame_model'])

        self.background = False
        self.background_fluence_free = False
        if op == ('Poisson', 'basic'):
            self._calc_D = self._calc_CP

        elif op == ('Poisson', 'fluence'):
            self._calc_D = self._calc_CwP

        elif op[1] == ('background') and (op[0] in ['Poisson', 'fluence_free']):
            self.b_d = self.K_di.B_di.b_d
            self.B_i = self.K_di.B_di.B_ji[0]
            self.background = True

            if op[0] == 'Poisson':
                self._calc_D = self._calc_CwP

            elif op[0] == 'fluence_free':
                self._calc_D = self._calc_KBPC
                self.KB_d = self.K_d - self.K_di.B_di.data_sum
                self.background_fluence_free = True

        elif op == ('fluence_free', 'basic'):
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
        self.D = self.K_di.shape[0]
        self.I = self.K_di.shape[1]
        self.r_chunk_size = min(r_chunk_size, self.R)
        self.d_chunk_size = min(d_chunk_size, self.D)

        # prepare mapping operations
        if cl is None:
            cl = utils_cl.opencl_init()

        if cl_cpu is None:
            cl_cpu = utils_cl.opencl_init_cpu()

        self.queue = cl['queue']

        self.mapper_cl = Mapper_cl(
                c['mapper'],
                cl['context'],
                cl['queue'])

        self.mapper_cl.load_buffers(
                r_chunk_size=self.r_chunk_size,
                ravel=True)

        self.scatter_add = utils_cl.ScatterAdd_cl(
                self.model.size,
                context=cl_cpu['context'])

        self.c = c

    @profiling.timed
    def calculate(self):
        N_n = np.zeros(self.model.size, dtype=float)
        D_n = np.zeros(self.model.size, dtype=float)
        mtime = 0.
        btime = 0.
        for r0, r1, dr in utils.chunker(self.r_chunk_size, self.R):
            # ------------------
            # 3. calculate P . K
            # ------------------
            self.N_ri = np.zeros((dr, self.I), dtype=np.float32)
            for d0, d1, _ in utils.chunker(self.d_chunk_size, self.D):
                # t0 = time()
                # self.N_ri = self.PdotK()
                P_dr = self.P_dr[d0:d1, r0:r1].astype(np.float32)
                K_di = self.K_di[d0:d1, :].astype(np.float32)

                """
                N_ri_dev, evt = gpu_dot(
                    P_dr,
                    K_di,
                    self.queue, a_transp=True, b_transp=False)

                evt.wait()
                self.N_ri += N_ri_dev.get()
                """
                # test np
                t0 = time()
                N_ri_dev = np.dot(P_dr.T, K_di)
                self.N_ri += N_ri_dev
                print(f'np.dot(P_dr.T, K_di) {r0=} {r1=} {d1-d0=} dot time:', time() - t0)

                # assert(np.allclose(N_ri, self.N_ri))
                # print(f'P dot K time:', time() - t0)
                # return self.N_ri

            # ----------------------
            # 3.5 subtract backgroud
            # N_ri = \sum_d P_dr (K_di - b_d B_i)
            #      = PK - B_i sum_d b_b P_dr
            # ----------------------
            if self.background:
                self.N_ri -= self.B_i[None, :] * np.dot(self.b_d, self.P_dr[:, r0:r1])[:, None]
                print(f'applied background offset to N_Ri')

            # -----------------
            # 4. calculate D_ri
            # -----------------
            # t0 = time()
            D_ri = self._calc_D(r0, r1)
            print(f'applied offset to D_ri')
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
                        s, r0, r1, ravel=True, cpu=True)
                mtime += time() - t0
                print(f'mapping time:', time() - t0)

                t0 = time()
                self.scatter_add.add(n_sri, self.N_ri, N_n)
                self.scatter_add.add(n_sri, D_ri,      D_n)
                print(f'bincount time:', time() - t0)
                btime += time() - t0

            print(f'mapping time:', mtime)
            print(f'bincount time:', btime)

        return N_n, D_n

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


    def _calc_KBPC(self, r0=0, r1=None):
        """
        W_ri <- 1/C_i sum_d P_dr (K_di - B_di) / sum_d w_dr P_dr
        w_rd = (K_d - B_d) / sum_i C_i W_ri

        D_ri = C_i / wsums_r [sum_d (K_d - B_d) P_dr]
        """
        if r1 is None:
            r1 = self.R

        self.N_ri *= self.wsums_r[r0:r1, None]

        D_ri = self.C_i[None, :] * \
                np.dot(self.KB_d, self.P_dr[:, r0:r1])[:, None]
        return D_ri

    def _calc_CkP(self, r0=0, r1=None):
        if r1 is None:
            r1 = self.R

        # test
        self.N_ri *= self.wsums_r[r0:r1, None] / self.C_i[None, :]
        D_ri = np.zeros(self.N_ri.shape, dtype=float)
        D_ri.T[:] = np.dot(self.K_d, self.P_dr[:, r0:r1])
        #self.N_ri *= self.wsums_r[r0:r1, None]
        #D_ri = self.C_i[None, :] * \
        #    np.dot(self.K_d, self.P_dr[:, r0:r1])[:, None]
        return D_ri


@profiling.timed
def calculate_fluence(config):
    D = config['classes'][0]['data'].shape[0]
    K_d = config['classes'][0]['data'].data_sum

    # only works globally!
    if config['classes'][0]['frame_model'] == 'background':
        B_d = config['classes'][0]['data'].B_di.data_sum
        K_d = np.clip(K_d.copy() - B_d, B_d+1, None)
        assert (np.all(K_d > 0))

    w_d = np.zeros((D,), dtype=float)

    for c in config['classes']:
        with h5py.File(c['wsums_file'], 'r') as f:
            w_d += f['P_dot_wsums'][()]

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


@profiling.timed
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

    # do it anyway
    #if not (likelihood == 'Poisson' and frame_model == 'basic'):
    if True:
        tomos = Tomograms(
                mapper,
                model,
                data.C_i)

        tomos_cl = Tomograms_cl(tomos, cl['context'], cl['queue'])

        wsums_r = tomos_cl.calculate_wsums()

    else:
        wsums_r = None

    return wsums_r

def update_model_subprocess(config_file, config, p_per_device=2, cids=None, update_w=True):
    """
    calculate tomogram sums and fluence in this process (might par. later)
    then farm off model update to subprocesses
    """
    import subprocess, sys
    from . import utils_cl, utils

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()

    cl = utils_cl.opencl_init()

    # get classes to process
    if cids is None:
        cids = list(range(len(config['classes'])))

    # calculate tomogram sums if needed
    # ---------------------------------
    t0 = time()
    for ci in cids:
        c = config['classes'][ci]

        # load model
        if c['model'].data is None:
            with h5py.File(c['model_file']) as f:
                c['model'].data = f['data'][()]

        c['mapper'].load_coords(c['data'].mask)
        c['wsums_r'] = calculate_wsums_cl(**c, cl=cl)

        fnam = c['probability_matrix_file']
        with h5py.File(fnam) as f:
            P_dr = f['P_dr'][()]

        Pw = np.dot(P_dr, c['wsums_r'])
        del P_dr

        # write to file
        with h5py.File(c['wsums_file'], 'w') as f:
            f['wsums_r'] = c['wsums_r']
            f['P_dot_wsums'] = Pw


    print(f'tomo time:', time() - t0)

    comm.Barrier()

    # calculate fluence if needed
    # ---------------------------
    if rank == 0:
        t0 = time()
        w_d = calculate_fluence(config)

        # write to file
        if config['update_fluence']:
            with h5py.File(config['fluence_file'], 'w') as f:
                f['w_d'] = w_d

        print(f'fluence time:', time() - t0)

    comm.Barrier()

    for ci in cids:
        if not config['classes'][ci]['update_model']:
            cids.remove(ci)

    devices = utils_cl.get_devices(device_type='gpu')

    # number parallel processes
    nproc = p_per_device

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
    from pathlib import Path

    config_fnam = sys.argv[1]
    class_ids = [int(c) for c in sys.argv[2].split(',')]
    device = int(sys.argv[3])

    config = pickle.load(open(config_fnam, 'rb'))
    wd = config['working_directory']

    profiling.setup(Path(wd) / 'profile')

    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]

    print(f'initialising opencl')
    cl = utils_cl.opencl_init(device_no=device)
    cl_cpu = utils_cl.opencl_init_cpu()

    for class_id in class_ids:
        c = config['classes'][class_id]

        if not c['update_model']:
            continue

        with profiling.cl_timed('io:load_data'):
            print(f'initialising mapper {class_id=}')
            c['data'].load_from_file()
            with h5py.File(c['model_file']) as f:
                c['model'].data = f['data'][()]
            with h5py.File(c['probability_matrix_file']) as f:
                c['P_dr'] = f['P_dr'][()]
            with h5py.File(c['wsums_file']) as f:
                c['wsums_r'] = f['wsums_r'][()]
            c['mapper'].load_coords(c['data'].mask)

        t0 = time()
        print(f'setting up model update {class_id=}')
        mupdate = Update_model_class(w_d, c, cl, cl_cpu, r_chunk_size=4*1024, d_chunk_size=4*1024)
        print(f'calling model update {class_id=}')
        N_n, D_n = mupdate.calculate()
        print(f'update time:', time() - t0)

        # finish and save model
        I_n = finish_model(N_n, D_n, c, profiling)
