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
        '--scale',
        type=float,
        help='Only merge r-indices with a given scale factor (background only)'
    )

    parser.add_argument(
        '--data_chunk',
        type=int,
        default=0,
        help='calculate for a subset of frames'
    )

    parser.add_argument(
        '--data_chunks',
        type=int,
        default=1,
        help='number of blocks to split frames over'
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

    def __init__(self, counts_n, m_min, m_max, model_shape, model_symmetry, queue, context):
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

        # buffer size = symmetry operations sum_n>=n_min^n_max counts_n
        counts_m = np.bincount(self.M_n, weights=counts_n,
                               minlength=n_asy.size).astype(int)

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

        print(f'{np.sum(counts_n)=} {buffer_size=}')

        # allocate buffers
        self.a = np.zeros((buffer_size,), dtype=np.float32)
        self.b = np.zeros((buffer_size,), dtype=np.float32)

        self.cl_code = cl.Program(context, self.code()).build()
        self.queue = queue
        self.checked = False
        self.events = None
        self.n_asy = n_asy

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
                j = index_m[m];
                a_dsri[j] = P_r[r] * (float)K_i[i];
                b_dsri[j] = B_i[i] / (C_i[i] * w);
                index_m[m] += 1;
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
    logger.info('update_I (start)')

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

    # merge out_m
    search_string = f'class_{class_c.class_id}_asymmetric_unit_*.h5'
    fnams = list(cachedir.glob(search_string))
    if len(fnams) == 0:
        raise ValueError(
            f'Error no files matching the pattern {search_string} '
            f'where found in the directory: {cachedir}'
        )

    desc = f'loading and mergeing asymmetric unit for class {class_c.class_id}'
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
        class_c.model.shape[0]//2,
        class_c.model.shape,
        symmetry=class_c.symmetry
    )

    n_asy = sym.get_asymmetric_unit()

    # apply symmetry to model
    shape = class_c.model.shape
    out_n = np.zeros(np.prod(shape), dtype=float)
    out_n[n_asy] = out_m

    out_n = symmetry.apply_symmetry(
        out_n.reshape(shape),
        class_c.symmetry,
        shape[0]//2
    )

    O_n = np.zeros(out_n.size, dtype=int)
    O_n[n_asy] = 1
    O_n = symmetry.apply_symmetry(
        O_n.reshape(shape),
        class_c.symmetry,
        class_c.model.shape[0]//2
    )

    O_n[O_n == 0] = 1
    out_n /= O_n

    I_n = out_n

    rms = np.mean((I0_n - I_n)**2)**0.5
    logger.info(f'rms difference for model {class_c.class_id}: {rms}')
    logger.info(f'class_c.class_id: {np.mean(I0_n)=} --> {np.mean(I_n)=}')
    print(f'rms difference for model {class_c.class_id}: {rms}')
    print(f'class_c.class_id: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    if args.output is None:
        args.output = args.class_file

    # save
    with h5py.File(args.output, 'a') as f:
        if 'model' in f:
            f['model'][:] = I_n
        else:
            f['model'] = I_n

    logger.info('update_I (stop)')

