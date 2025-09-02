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
from emc2 import classes
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


if __name__ == "__main__":
    args = get_args()

    working_directory = Path(args.class_file).parent
    cachedir = working_directory.joinpath('cachdir')

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_I_fill_buffer (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file, skip=['probability_matrix'])

    logger.info(f'mean wsums for class {class_c.class_id}: '
                f'{np.mean(class_c.wsums)}')

    if class_c.update_model:
        logger.info('update_model is True, updating model')
    else:
        logger.info('update_model is False, skipping model update')

        fnam_jobs = cachedir.joinpath(f'class_fill_buffer_jobs.txt')
        with open(fnam_jobs, 'a') as file:
            cmd = f'echo nothing to do for {class_c.class_id}\n'
            file.write(cmd)

        sys.exit()

    # initialise mapper: r,i -> s,n
    # Start main calculation
    # ----------------------

    # counts_per_n = np.bincount(n_dsri, minlength=N)
    # n_indices = np.concatenate(([0], np.cumsum(counts_per_n)))

    # apply symmetry to counts_n
    # counts_n = symmetry.apply_symmetry(
    #     counts_n.reshape(class_c.model.shape),
    #     class_c.symmetry,
    #     class_c.model.shape[0]//2
    # )

    # get asymmetric unit

    # single pass on cpu to get a, b
    # keep track of counts per n-index
    # only works if all symmetry operations are applied by
    # mapper or are calculated during single pass

    # I need a function that takes n-values and returns
    # m'th index in the ravelled asymetric unit.
    # needs to be fast: so store an array N_asym[n] = m
    # this helps with filtering n-values for the single pass
    # algorithm:
    # - loop over d:
    #   - loop over r:
    #       - get n-values for non-zero pixels
    #       - get n_asym from n-values
    #       - for n_asym in range [n_min, n_max)
    #         calculate a and b and store them in buffers

    # load and merge counts
    counts_n = np.zeros(class_c.model.size, dtype=int)

    # merge a, b, n
    search_string = f'class_{class_c.class_id}_buffersize_*.h5'
    fnams_counts = list(cachedir.glob(search_string))
    if len(fnams_counts) == 0:
        raise ValueError(
            f'Error no files matching the pattern {search_string} '
            f'where found in the directory: {cachedir}'
        )

    desc = f'loading and mergeing buffer sizes for class {class_c.class_id}'
    for fnam in tqdm(fnams_counts, desc=desc):
        with h5py.File(fnam, 'r') as f:
            counts_n += f['counts_n'][()]

        # fnam.unlink()

    # merge c_n
    c_n = np.zeros(class_c.model.size, dtype=float)
    search_string = f'class_{class_c.class_id}_c_n_*.h5'
    fnams_cn = list(cachedir.glob(search_string))
    if len(fnams_cn) == 0:
        raise ValueError(
            f'Error no files matching the pattern {search_string} '
            f'where found in the directory: {cachedir}'
        )

    desc = f'loading and mergeing c_n for class {class_c.class_id}'
    for fnam in tqdm(fnams_cn, desc=desc):
        with h5py.File(fnam, 'r') as f:
            c_n += f['c_n'][()]

        # fnam.unlink()

    # save
    fnam = f'class_{class_c.class_id}_c_n.h5'
    fnam = cachedir.joinpath(fnam)
    with h5py.File(fnam, 'w') as f:
        f['c_n'] = c_n

    # m_min = 32**3
    # m_max = 32 * 32**3
    max_mem_gb = 4

    max_buf_size = max_mem_gb * 1024**3 / 4 / 2

    sym = symmetry.Symmetry(
        class_c.model.shape[0]//2,
        class_c.model.shape,
        symmetry=class_c.symmetry
    )

    # n: raveled model voxel location
    # m: raveled asymmetric unit voxel location
    # n = n_asy[m]
    # m = M_n[n]
    n_asy = sym.get_asymmetric_unit()
    M_n = sym.get_asymmetric_unit_mapping()

    # buffer size = symmetry operations sum_n>=n_min^n_max counts_n
    counts_m = np.bincount(M_n, weights=counts_n,
                           minlength=n_asy.size)

    counts_m = np.rint(counts_m).astype(int)

    # testing
    # sys.exit()

    # save
    fnam = f'class_{class_c.class_id}_buffersize.h5'
    fnam = cachedir.joinpath(fnam)
    with h5py.File(fnam, 'w') as f:
        f['counts_m'] = counts_m

    # check for null solution:
    if np.sum(counts_m) > 0:
        # find the set of m_min and m_max values that fit in memory
        cc = np.cumsum(counts_m)
        m_min = 0
        m_max = 0
        m_mins = []
        m_maxs = []
        while m_max < len(n_asy):
            m_max = np.searchsorted(cc, max_buf_size+cc[m_max])
            m_mins.append(m_min)
            m_maxs.append(m_max)
            m_min = m_max

        # write jobs file
        fnam_jobs = cachedir.joinpath(f'class_fill_buffer_jobs.txt')
        with open(fnam_jobs, 'a') as file:
            device = 0
            for m_min, m_max in zip(m_mins, m_maxs):
                cmd = f'python scripts/update_I_fill_buffer_2.py '\
                      f'--m_min {m_min} --m_max {m_max} '\
                      f'--device {device} '\
                      f'{args.class_file}\n'

                file.write(cmd)
                device += 1


    logger.info('update_I_fill_buffer (stop)')
