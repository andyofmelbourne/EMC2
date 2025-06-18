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

from context import emc2
from emc2 import utils
from emc2 import tomograms
from emc2 import utils_cl
from emc2 import classes
from emc2 import data_getter
from emc2 import model_update_background_sparse
from emc2 import get_script_logger
from emc2 import symmetry

import pyopencl as cl


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""Update model intensities"""
    )

    parser.add_argument(
        'class_file',
        type=str,
        help='file name class file'
    )

    parser.add_argument(
        '--m_min',
        type=int,
        help='starting index in asymmetric unit to solve'
    )

    parser.add_argument(
        '--m_max',
        type=int,
        help='stopping index in asymmetric unit to solve'
    )

    parser.add_argument(
        '--device',
        type=int,
        required=True,
        help='opencl device no. (wraps)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help="h5 file to write 'model' dataset to. \
        Default is input class file."
    )

    args = parser.parse_args()
    return args


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

        event = self.cl_code.fill_frame(
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


if __name__ == "__main__":
    args = get_args()

    working_directory = Path(args.class_file).parent
    cachedir = working_directory.joinpath('cachdir')

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_I_fill_buffer_2 (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file, skip=['probability_matrix'])

    logger.info(f'mean wsums for class {class_c.class_id}: '
                f'{np.mean(class_c.wsums)}')

    if class_c.update_model:
        logger.info('update_model is True, updating model')
    else:
        logger.info('update_model is False, skipping model update')
        sys.exit()

    I0_n = class_c.model.copy()

    # load data chunk
    K_di_getter = data_getter.Data_getter(
        mask=class_c.mask,
        split_frames=class_c.split_frames,
        cxi_file=class_c.cxi_file,
        filter=class_c.frame_selection,
        working_directory=working_directory,
        frame_model=class_c.frame_model
    )
    I = K_di_getter.shape[1]
    D = K_di_getter.shape[0]

    B_di = data_getter.Data_getter_background(K_di_getter)

    opencl_stuff = utils_cl.opencl_init(args.device)

    with h5py.File(args.class_file) as f:
        P_dr = f['probability_matrix'][()]

    opencl_stuff_cpu = utils_cl.opencl_init_cpu()

    # Start main calculation
    # ----------------------
    w_d = class_c.relative_fluence
    C_i = class_c.C
    N = class_c.model.size

    Ps_dr, rs_d, Nr = model_update_background_sparse.get_sparse_P_matrix(P_dr)

    mapper = tomograms.Mapper(
        class_c.model.ndim,
        class_c.model.shape[0],
        class_c.xyz,
        class_c.mapping_matrix,
        opencl_stuff['context'],
        opencl_stuff['queue'],
        interpolation=class_c.interpolation_forward
    )

    # m_min = 32**3
    # m_max = 32 * 32**3
    m_min = args.m_min
    m_max = args.m_max
    rmax = 32

    fnam = f'class_{class_c.class_id}_buffersize.h5'
    fnam = cachedir.joinpath(fnam)
    with h5py.File(fnam, 'r') as f:
        counts_m = f['counts_m'][()]

    fnam = f'class_{class_c.class_id}_c_n.h5'
    fnam = cachedir.joinpath(fnam)
    with h5py.File(fnam, 'r') as f:
        c_n = f['c_n'][()]

    fill_buffers = Fill_buffers(
            counts_m, m_min, m_max, class_c.model.shape, class_c.symmetry,
            opencl_stuff_cpu['queue'],
            opencl_stuff_cpu['context']
    )

    # single pass algorithm
    mapper.cpu = True
    counts = 0
    # counts_test_n = np.zeros(c_n.shape, dtype=int)
    for d in tqdm(range(D), desc='filling buffers'):
        if len(rs_d[d]) > 0:
            wd = np.float32(w_d[d])
            Kd_i, pixels = K_di_getter.sparse(d)
            Bd_i = B_di.sparse(d, pixels)
            Cd_i = np.ascontiguousarray(C_i[pixels].astype(np.float32))
            Pd_r = np.ascontiguousarray(Ps_dr[d].astype(np.float32))
            # N_sri = mapper[:, rs_d[d], pixels]
            index = 0
            counts += len(rs_d[d]) * len(pixels)
            while index < len(rs_d[d]):
                i = index
                j = min(index+rmax, len(rs_d[d]))
                rs = rs_d[d][i: j]
                N_sri = mapper[:, rs, pixels]
                # counts_test_n += np.bincount(N_sri.ravel(),
                #                              minlength=counts_test_n.size)
                fill_buffers.fill_frame(Pd_r, Kd_i, Bd_i, wd, C_i, N_sri)
                index = j

    # testing
    # sys.exit()

    # apply symmetry (add symmetry partners) to c_n
    shape = class_c.model.shape
    model_symmetry = class_c.symmetry
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
            ns = sym.get_symmetry_partners(n)
            c = np.sum(c_n[ns])
            out_m[m] = utils.solve_axbc(
                a_n[i: j],
                b_n[i: j],
                c,
                fill_value=0., ftol=1e-2, xtol=1e-3,
                maxiters=1000, algorithm='Halley', debug=False
            )

    # save
    fnam = f'class_{class_c.class_id}_asymmetric_unit_{args.device}.h5'
    fnam = cachedir.joinpath(fnam)
    with h5py.File(fnam, 'w') as f:
        f['I_m'] = out_m
        f['m_min'] = fill_buffers.m_min
        f['m_max'] = fill_buffers.m_max

    logger.info('update_I_fill_buffer_2 (stop)')

