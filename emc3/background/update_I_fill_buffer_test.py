"""
script to solve for subset of model voxels without caching

The photon data can raveled in dsri space:

dens:
         |d=0 s=0 r=0|d=0 s=0 r=1|....|
i_dsri = 012345......012345...... (repeats over d, s, r)
K_dsri = 0000100200030004000001001000101000304
n_dsri = 125463... (repeats over d)

photon sparse:
         |d=0 s=0 r=0|d=0 s=0 r=1|....|
i_dsri = 4589            356      089
K_dsri = 1234            111      134
n_dsri = 3928            184      710

P-matrix sparse:
         |d=0 s=0 r=0|d=0 s=0 r=2|....|
i_dsri = 4589            089
K_dsri = 1234            134
n_dsri = 3928            710

The above could still be a very large dataset. Now let us restrict
this to a sub-set of n's.

P and K sparse:
d_dsri = 438349098273409387
i_dsri = 458908901785613612
K_dsri = 123413429384021384
n_dsri = 392871029384704702

P and K sparse for n:[0, 4):
d_dsri = 444666677
i_dsri = 488907112
K_dsri = 133423284
n_dsri = 321023002

generating this table involves loading every frame
and calculating n for every non-zero pixel for
every s, r-value in the sparse P-matrix.

It should be possible to reduce the number of n-calculations
by noting the pixel radius and other things but this would be
complicated.

Anyway let's say we do this. Actually I think we need:
d_j, r_j, i_j, n_j
depending on how we order the index j, we could compress
the d_j and r_j datasets because of repitition.

then we have to sort, or do may passes through n_dsri, to
collect all values associated with a given n.

then we have to do a lot of out-of-order memory access to
get K, B, C (w, P to a lesser extent).

But actually we just need a_dsri and b_dsri.

So if I could find a way to argsort n_dsri then we can solve
that problem


- n_min, n_max

- generate sparse P-matrix

- calculate number of pixels contributing to each voxel: N_n

- calculate number of pixels contributing to each voxel
  in range for each frame and rotation in sparse matrix: N_dr
  This will be as large as the probability matrix in the worst case

- make buffers a, b, n for n:[n_min, n_max)

do this on cpu? with an a, b, n buffer for each cpu
calculate number of pixels contributing

- loop over d
    - get sparse r's for this d: rs
    - load Kd_i, pixels
    - load Bd_i = B_d[pixels]
    - load C_i = C[pixels]

    - calculate n_sri = M_sri[:, rs, pixels]

    - calculate number of pixels contributing to n in range
      for each s, r: Nd_sr (bincount)

    - calculate cumulative number of pixels contributing to n in range
      for each s, r: Ncd_sr

    j = offset + s * R * I
    a_dsri
- loop over d
"""





import argparse
import sys
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import subprocess

from .. import utils_cl
from .update_I_buffer_size import get_sparse_P_matrix
from ..utils import solve_axbc
from .. import symmetry
from ..mapper import Mapper_cl_cpu_sparse, Mapper_cl_gpu_sparse
from ..update_models import apply_filter

import pyopencl as cl

from time import time


class Fill_buffers():

    def __init__(self, counts_m, m_min, m_max, model_shape, model_symmetry, queue, context):
        sym = symmetry.Symmetry(
            model_shape[0]//2,
            model_shape,
            symmetry=model_symmetry
        )

        # n: raveled model voxel location
        # m: raveled asymmetric unit voxel location
        # n = n_asy[m]
        # m = M_n[n]
        n_asy = sym.get_asymmetric_unit()
        self.M_n = sym.get_asymmetric_unit_mapping()

        if m_min is None:
            m_min = 0

        if m_max is None:
            m_max = n_asy.size

        buffer_size = np.sum(counts_m[m_min: m_max]).astype(int)
        index_m = np.concatenate(
                ([0], np.cumsum(counts_m[m_min: m_max])[:-1]), axis=0
        )
        self.index_m = np.ascontiguousarray(index_m.astype(np.int32))
        self.index0_m = self.index_m.copy()
        self.counts_m = counts_m[m_min: m_max]

        self.m_min = np.int32(m_min)
        self.m_max = np.int32(m_max)

        # allocate buffers
        self.a = np.zeros((buffer_size,), dtype=np.float32)
        self.b = np.zeros((buffer_size,), dtype=np.float32)

        self.cl_code = cl.Program(context, self.code()).build()
        self.queue = queue
        self.checked = False
        self.events = None
        self.n_asy = n_asy

        # self.counts = 0
        # self.counts_test_m = np.zeros((m_max-m_min,), dtype=int)

    def fill_frame(self, P_r, K_i, B_i, w, C_i, N_sri):
        # a_j = P_dr K_di         for i in M_rn and for non-zero elements
        # b_j = B_di / (w_d C_i)  for above elements

        M_sri = np.ascontiguousarray(self.M_n[N_sri].astype(np.int32))

        if not self.checked:
            assert(P_r.dtype == np.float32)
            assert(K_i.dtype == np.uint8)
            assert(B_i.dtype == np.float32)
            assert(w.dtype == np.float32)
            assert(C_i.dtype == np.float32)
            assert(M_sri.dtype == np.int32)

            for ar in (P_r, K_i, B_i, w, C_i, M_sri):
                assert(ar.flags.c_contiguous)

        S, R, I = M_sri.shape
        S, R, I = np.int32(S), np.int32(R), np.int32(I)

        # self.counts += S * R * I
        # self.counts_test_m += np.bincount(M_sri.ravel(),
        #                                   minlength=self.m_max-self.m_min)

        assert(np.max(self.index_m) <= self.a.size)

        event = cl.Kernel(self.cl_code, 'fill_frame')(
                self.queue, (1,), (1,),
                cl.SVM(self.a),
                cl.SVM(self.b),
                cl.SVM(M_sri),
                cl.SVM(P_r),
                cl.SVM(K_i),
                cl.SVM(B_i),
                cl.SVM(C_i),
                w,
                cl.SVM(self.index_m),
                self.m_min,
                self.m_max,
                S,
                R,
                I,
                wait_for=self.events
        )

        # have to wait unless I have buffers
        event.wait()
        return

    def code(self):
        return """
        __kernel void fill_frame (
            global float *a_dsri,
            global float *b_dsri,
            global int   *m_dsri,
            global float *P_r,
            global uchar *K_i,
            global float *B_i,
            global float *C_i,
            const  float w,
            global int   *index_m,
            const  int   m_min,
            const  int   m_max,
            const  int   S,
            const  int   R,
            const  int   I
        ) {
        int m=0;
        int j=0;
        for (int s=0; s<S; s++){
        for (int r=0; r<R; r++){
        for (int i=0; i<I; i++){
            m = m_dsri[s * R * I + r * I + i];
            if ((m >= m_min) && (m < m_max)) {
                j = index_m[m-m_min];
                a_dsri[j] = P_r[r] * (float)K_i[i];
                b_dsri[j] = B_i[i] / (C_i[i] * w);
                index_m[m-m_min] += 1;
            }
        }}}
        }
        """

def merge_I(c):
    cid = c['class_id']

    # merge out_m
    fnams = list(Path('./').glob(f'class_{cid}_asymmetric_unit_*.h5'))

    desc = f'loading and mergeing asymmetric unit for class {cid}'
    I_m = []
    m_mins = []
    m_maxs = []
    for fnam in tqdm(fnams, desc=desc):
        with h5py.File(fnam, 'r') as f:
            I_m.append(f['I_m'][()])
            m_mins.append(f['m_min'][()])
            m_maxs.append(f['m_max'][()])

    M = np.max(m_maxs)
    out_m = np.zeros((M,), dtype=float)
    for m_min, m_max, I in zip(m_mins, m_maxs, I_m):
        out_m[m_min: m_max] = I

    sym = symmetry.Symmetry(
        c['model'].shape[0]//2,
        c['model'].shape,
        symmetry=c['model'].symmetry
    )

    n_asy = sym.get_asymmetric_unit()

    # apply symmetry to model
    shape = c['model'].shape
    out_n = np.zeros(np.prod(shape), dtype=float)
    out_n[n_asy] = out_m

    out_n = sym.apply_symmetry(out_n.reshape(shape))

    O_n = np.zeros(out_n.size, dtype=int)
    O_n[n_asy] = 1
    O_n = sym.apply_symmetry(O_n.reshape(shape))
    O_n[O_n == 0] = 1
    out_n /= O_n

    I_n = out_n

    with h5py.File(c['model_file']) as f:
        data = f['data']
        dq = f['dq'][()]
        I0_n = None
        if data.shape == I_n.shape:
            if dq == c['model'].dq:
                m = I_n == 0.
                I0_n = data[()]
                I_n[m] = I0_n[m]

    I_n = apply_filter(c['model'].dq, I_n, c['filter_model'])

    if I0_n is not None:
        rms = np.mean((I0_n - I_n)**2)**0.5
        print(f'rms difference for model {cid}: {rms}')
        print(f'{cid}: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    # save
    with h5py.File(c['model_file'], 'w') as f:
        f['data'] = I_n
        f['dq'] = c['model'].dq

def fill_buffer(config, config_file):
    max_mem_gb = 4
    max_buf_size = max_mem_gb * 1024**3 / 4 / 2

    cmds = []

    fnam_jobs = Path(f'class_fill_buffer_jobs.txt')
    if fnam_jobs.is_file():
        fnam_jobs.unlink()

    for ci, c in enumerate(config['classes']):
        if not c['update_model']:
            continue

        # load counts
        class_id = c['class_id']
        fnam = f'class_{class_id}_buffersize.h5'
        with h5py.File(fnam) as f:
            counts_m = f['counts_m'][()]
            n_asym = f['n_asym'][()]

        # check for null solution:
        if np.sum(counts_m) > 0:
            # find the set of m_min and m_max values that fit in memory
            cc = np.cumsum(counts_m)
            m_min = 0
            m_max = 0
            m_mins = []
            m_maxs = []
            while m_max < len(n_asym):
                m_max = np.searchsorted(cc, max_buf_size+cc[m_max])
                m_mins.append(m_min)
                m_maxs.append(m_max)
                m_min = m_max

            # write jobs file
            with open(fnam_jobs, 'a') as file:
                device = 0
                for m_min, m_max in zip(m_mins, m_maxs):
                    cmd = f'python -m emc3.background.update_I_fill_buffer '\
                          f'{m_min} {m_max} {ci} '\
                          f'{device} '\
                          f'{config_file}\n'

                    file.write(cmd)
                    device += 1

    cmd = f'parallel --halt now,fail=1 --verbose --jobs 2 --delay 2 < class_fill_buffer_jobs.txt'
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

    # now merge output
    for ci, c in enumerate(config['classes']):
        if not c['update_model']:
            continue

        merge_I(c)

if __name__ == "__main__":
    m_min, m_max, ci, device, config_file = sys.argv[1:]
    m_min, m_max, ci, device = int(m_min), int(m_max), int(ci), int(device)

    # prepare data
    # ------------
    config = pickle.load(open(config_file, 'rb'))

    c = config['classes'][ci]

    with h5py.File(c['probability_matrix_file']) as f:
        P_dr = f['P_dr'][()]

    cl_cpu = utils_cl.opencl_init_cpu()

    with h5py.File(c['fluence_file']) as f:
        w_d = f['w_d'][()]

    C_i = np.ascontiguousarray(c['data'].C_i.astype(np.float32))

    N = c['model'].size

    c['mapper'].load_coords(c['data'].mask)

    mapper_cpu = Mapper_cl_cpu_sparse(c['mapper'], cl_cpu['context'], cl_cpu['queue'])
    S = mapper_cpu.shape[0]

    cid = c['class_id']

    c['data'].load_from_file()
    K_di = c['data']
    B_di = c['data'].B_di

    fnam = f'class_{cid}_buffersize.h5'
    with h5py.File(fnam, 'r') as f:
        counts_m = f['counts_m'][()]

    fnam = f'class_{cid}_c_n.h5'
    with h5py.File(fnam, 'r') as f:
        c_n = f['c_n'][()].ravel()

    Ps_dr, rs_d, Nr = get_sparse_P_matrix(P_dr)

    # Start main calculation
    # ----------------------
    # m_min = 32**3
    # m_max = 32 * 32**3
    m_min = m_min
    m_max = m_max
    rmax = 32
    #rmax = 1024

    # set buffer size for mapper
    cl_gpu = utils_cl.opencl_init(device)
    mapper_gpu = Mapper_cl_gpu_sparse(c['mapper'], cl_gpu['context'], cl_gpu['queue'])
    buffer_size = rmax * K_di.shape[1]
    mapper_gpu.load_buffers(buffer_size, ravel=True)


    fill_buffers = Fill_buffers(
            counts_m, m_min, m_max, c['model'].shape, c['model'].symmetry,
            cl_cpu['queue'],
            cl_cpu['context']
    )

    t_map = 0
    t_fill = 0
    # single pass algorithm
    counts = 0
    # counts_test_n = np.zeros(c_n.shape, dtype=int)
    D = c['data'].shape[0]
    for d in tqdm(range(13000, 14000), desc='filling buffers'):
    # for d in tqdm(range(D), desc='filling buffers'):
        if len(rs_d[d]) > 0:
            wd = np.float32(w_d[d])
            pixels, Kd_i = K_di.data.get_sparse(d)
            Bd_i = B_di[d][0][pixels]
            Cd_i = np.ascontiguousarray(C_i[pixels].astype(np.float32))
            Pd_r = np.ascontiguousarray(Ps_dr[d].astype(np.float32))
            # N_sri = mapper[:, rs_d[d], pixels]
            index = 0
            counts += len(rs_d[d]) * len(pixels)
            while index < len(rs_d[d]):
                i = index
                j = min(index+rmax, len(rs_d[d]))
                rs = rs_d[d][i: j]
                # N_sri = mapper[:, rs, pixels]
                ns = []
                t0 = time()
                for s in range(S):
                    n_ri = mapper_cpu.calculate_mapping_rlist_pixlist(s, rs, pixels, ravel=True)
                    # n_ri = mapper_gpu.calculate_mapping_rlist_pixlist(s, rs, pixels)
                    # i = np.where((n_ri != ng_ri).ravel())[0]
                    # print(f'{n_ri.ravel()[i]=}')
                    # print(f'{ng_ri.ravel()[i]=}')
                    # assert(np.allclose(n_ri, ng_ri))
                    ns.append(n_ri)
                N_sri = np.ascontiguousarray(np.array(ns).astype(np.int32))
                t_map += time()-t0
                # counts_test_n += np.bincount(N_sri.ravel(),
                #                              minlength=counts_test_n.size)
                t0 = time()
                fill_buffers.fill_frame(Pd_r, Kd_i, Bd_i, wd, Cd_i, N_sri)
                t_fill += time()-t0
                index = j

    print(f'mapping time: {t_map}')
    print(f'filling time: {t_fill}')

    with h5py.File('temp_gpu.h5', 'w') as f:
        f['a'] = fill_buffers.a
        f['b'] = fill_buffers.b
    # with h5py.File('temp_cpu.h5', 'w') as f:
    #     f['a'] = fill_buffers.a
    #     f['b'] = fill_buffers.b

    # testing
    sys.exit()

    # apply symmetry (add symmetry partners) to c_n
    shape = c['model'].shape
    model_symmetry = c['model'].symmetry
    sym = symmetry.Symmetry(
        shape[0]//2,
        shape,
        symmetry=model_symmetry
    )

    # solve
    M = fill_buffers.index0_m.size
    out_m = np.zeros((M,), dtype=float)
    a_n = fill_buffers.a
    b_n = fill_buffers.b

    for m in tqdm(range(M), desc='solving'):
        i = fill_buffers.index0_m[m]
        j = fill_buffers.index0_m[m] + fill_buffers.counts_m[m]
        n = fill_buffers.n_asy[m+m_min]
        if j > i:
            ns = list(sym.get_symmetry_partners(n))
            cc = np.sum(c_n[ns])
            out_m[m] = solve_axbc(
                a_n[i: j],
                b_n[i: j],
                cc,
                fill_value=0., ftol=1e-2, xtol=1e-3,
                maxiters=1000, algorithm='Halley', debug=False
            )

    # save
    fnam = f'class_{cid}_asymmetric_unit_{device}.h5'
    with h5py.File(fnam, 'w') as f:
        f['I_m'] = out_m
        f['m_min'] = fill_buffers.m_min
        f['m_max'] = fill_buffers.m_max

