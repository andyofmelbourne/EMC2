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

"""
Wsum_r = sum_i C_i W_ri

Basic:
    N_ri  = sum_d P_dr K_di
    D_ri  = C_i sum_d P_dr

Fluence:
    N_ri  = sum_d P_dr K_di
    D_r   = C_i sum_d w_d P_dr

    a_d   = sum_i K_di
    b_d   = sum_r P_dr Wsum_r
    w'_d  = a_d / b_d

Fluence free:
    N_ri = Wsum_r sum_d P_dr K_di
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
    1. calculate P . K
    2. calculate voxel mapping n_sri
    3. merge N and D

If I had a faster way to share / pipe I would split
these into separate processes for optimal load balancing
"""


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
        '--r_chunk_size',
        type=int,
        help='reduce memory consumption by processing r-indices in chunks.\
        Default is all values.'
    )

    parser.add_argument(
        '--numpy',
        action='store_true',
        default=False,
        help='use numpy dot'
    )

    parser.add_argument(
        '--sparse',
        action='store_true',
        default=False,
        help='skip frames where P_dr < P_thresh'
    )

    parser.add_argument(
        '--P_thresh',
        type=float,
        default=1e-4,
        help='see --sparse'
    )

    parser.add_argument(
        '--frame_chunk_size',
        type=int,
        help='reduce memory consumption by loading K_di in <frame_chunk_size>\
        chunks. Default is all frames.'
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='Determines which opencl device to use (will wrap if device >\
        total).'
    )

    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('wb'),
        default=sys.stdout.buffer,
        help="Python pickle output file. \
            The result is written as a dictionary"
    )

    args = parser.parse_args()
    return args


def K_dot_P(
    K_di_getter,
    P_dr,
    numpy=True,
    sparse=False,
    frame_chunk_size=None,
    P_thresh=0.,
    N_ri=None
):
    D, R = P_dr.shape
    I = K_di_getter.shape[1]

    if N_ri is None:
        N_ri = np.empty((R, I), dtype=P_dr.dtype)

    if args.numpy and not sparse:
        d_iter = tqdm(
            utils.chunker(frame_chunk_size, D),
            desc='calculating K . P using numpy',
            leave=False
        )

        for d0, d1, dd in d_iter:
            K_di = K_di_getter[d0:d1]
            np.dot(P_dr[d0:d1].T, K_di, out=N_ri)

    elif numpy and sparse:
        Ds = 0
        d_iter = tqdm(
            range(R),
            desc='calculating K . P (sparse)',
            leave=False
        )
        for r in d_iter:
            ds = np.where(P_dr[:, r] > P_thresh)[0]
            if len(ds) > 0:
                K_di = K_di_getter[ds]
                Ds += len(ds)
                np.dot(P_dr[ds, 0], K_di, out=N_ri[r])
        logger.info(f'processed {100 * Ds / (D * R):.2f}% of frames')
    else:
        raise ValueError(f'could not parse {numpy=} {sparse=}')
    return N_ri


def main(
    class_file,
    mapper,
    model,
    K_di_getter,
    wsums,
    w_d,
    C_i,
    r_chunk_size,
    frame_chunk_size,
    ksums,
    numpy=True,
    sparse=False,
    P_thresh=0.,
    likelihood='Poisson',
    frame_model='basic',
    symmetry_name='P1'
):
    with h5py.File(class_file) as f:
        D, R = f['probability_matrix'].shape

    if r_chunk_size:
        r_chunk_size = min(r_chunk_size, R)
    else:
        r_chunk_size = R

    r_iter = tqdm(
        utils.chunker(r_chunk_size, R),
        desc='updating model over r-chunks'
    )

    N = model.shape[0]
    N_n = np.zeros(model.size, dtype=float)
    D_n = np.zeros(model.size, dtype=float)

    assert (D == K_di_getter.shape[0])

    if frame_chunk_size:
        frame_chunk_size = min(frame_chunk_size, D)
    else:
        frame_chunk_size = D

    for r0, r1, dr in r_iter:
        with h5py.File(class_file) as f:
            P_dr = f['probability_matrix'][:, r0:r1]

        wsums_r = wsums[r0:r1]
        D, R = P_dr.shape

        # calculate sum_d K_di P_dr
        # which is common to all non-background models
        N_ri = K_dot_P(
            K_di_getter,
            P_dr,
            numpy=numpy,
            sparse=sparse,
            frame_chunk_size=frame_chunk_size,
            P_thresh=P_thresh,
            N_ri=None
        )

        # now calculate N_ri and D_ri
        # according to model parameters
        # for merging into I-space
        if (
            likelihood == 'Poisson' and
            frame_model == 'basic'
        ):
            D_ri = C_i[None, :] * \
                    np.sum(P_dr, axis=0)[:, None]

        elif (
            likelihood == 'Poisson'
            and frame_model == 'fluence'
        ):
            D_ri = C_i[None, :] * np.dot(w_d, P_dr)[:, None]

        elif (
            likelihood == 'Poisson_fluence_free'
            and frame_model == 'basic'
        ):
            N_ri *= wsums_r[:, None]
            D_ri = C_i[None, :] * \
                np.dot(ksums, P_dr)[:, None]

        else:
            raise ValueError(f'could not parse likelihood {class_c.likelihood}'
                             f'and frame_model {class_c.frame_model}')

        n_sri = mapper[:, r0:r1, :]

        # now merge N_ri and D_ri to I-space
        if class_c.maximise == 'W':
            D_ri[D_ri == 0] = 1.
            N_ri /= D_ri
            D_ri[:] = 1.

        for s in tqdm(range(n_sri.shape[0]), leave=False):
            for r in range(n_sri.shape[1]):

                N_n += np.bincount(
                    n_sri[s, r],
                    N_ri[r],
                    minlength=model.size
                )

                D_n += np.bincount(
                    n_sri[s, r],
                    D_ri[r],
                    minlength=model.size
                )

    # apply symmetry
    i0 = N // 2

    N_n = symmetry.apply_symmetry(
        N_n.reshape(model.shape),
        symmetry_name,
        i0
    )

    D_n = symmetry.apply_symmetry(
        D_n.reshape(model.shape),
        symmetry_name,
        i0
    )

    # I = N / D
    D_n[D_n == 0] = 1.
    N_n /= D_n

    return N_n


if __name__ == "__main__":
    args = get_args()

    working_directory = Path(args.class_file).parent

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_I (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file, skip=['probability_matrix'])

    logger.info(f'mean wsums for class {class_c.class_id}: '
                f'{np.mean(class_c.wsums)}')

    I0_n = class_c.model.copy()

    if class_c.update_model is False:
        logger.info('update_model is False, skipping model update')

    # calculate model voxel indices for each r,i pair
    # -----------------------------------------------
    opencl_stuff = utils_cl.opencl_init(args.device)

    # initialise mapper: r,i -> s,n
    mapper = tomograms.Mapper(
        class_c.model.ndim,
        class_c.model.shape[0],
        class_c.xyz,
        class_c.mapping_matrix,
        opencl_stuff['context'],
        opencl_stuff['queue'],
        interpolation=class_c.interpolation_forward
    )

    # load data
    working_directory = Path(args.class_file).parent
    K_di_getter = data_getter.Data_getter(
        mask=class_c.mask,
        split_frames=class_c.split_frames,
        cxi_file=class_c.cxi_file,
        filter=class_c.frame_selection,
        working_directory=working_directory,
        frame_model=class_c.frame_model
    )
    I = K_di_getter.shape[1]

    if (
        class_c.frame_model == 'basic'
        or class_c.frame_model == 'fluence'
    ):
        I_n = main(
            args.class_file,
            mapper,
            class_c.model,
            K_di_getter,
            class_c.wsums,
            class_c.relative_fluence,
            class_c.C,
            args.r_chunk_size,
            args.frame_chunk_size,
            class_c.ksums,
            args.numpy,
            args.sparse,
            args.P_thresh,
            class_c.likelihood,
            class_c.frame_model,
            class_c.symmetry
        )

    elif class_c.frame_model == 'background':
        B_di = data_getter.Data_getter_background(K_di_getter)

        with h5py.File(args.class_file) as f:
            P_dr = f['probability_matrix'][()]

        I_n = model_update_background_sparse.I_update(
            class_c.relative_fluence,
            class_c.model,
            P_dr,
            K_di_getter,
            B_di,
            mapper,
            class_c.C,
            opencl_stuff['queue'],
            opencl_stuff['context'],
            opencl_stuff['device'],
            class_c.symmetry
        )
    else:
        raise ValueError(f'{class_c.frame_model=} not supported')

    rms = np.mean((I0_n - I_n)**2)**0.5
    logger.info(f'rms difference for model {class_c.class_id}: {rms}')

    logger.info(f'class_c.class_id: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    # test
    # I_n = np.clip(I_n, 1e-8, None)

    # save
    with h5py.File(args.class_file, 'r+') as f:
        f['model'][:] = I_n

    logger.info('update_I (stop)')
