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
    logger.info('update_I_buffer_size (start)')

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

    # load data chunk
    K_di_getter = data_getter.Data_getter(
        mask=class_c.mask,
        split_frames=class_c.split_frames,
        cxi_file=class_c.cxi_file,
        filter=class_c.frame_selection,
        working_directory=working_directory,
        mpi_split_frames=(args.data_chunk, args.data_chunks),
        frame_model=class_c.frame_model
    )
    I = K_di_getter.shape[1]

    d0 = K_di_getter.d_start_mpi[args.data_chunk]
    d1 = K_di_getter.d_stop_mpi[args.data_chunk]
    dd = d1 - d0

    opencl_stuff = utils_cl.opencl_init(args.data_chunk)

    with h5py.File(args.class_file) as f:
        P_dr = f['probability_matrix'][d0: d1]

    opencl_stuff_cpu = utils_cl.opencl_init_cpu()

    # Start main calculation
    # ----------------------
    w_d = class_c.relative_fluence[d0: d1]
    C_i = class_c.C
    N = class_c.model.size

    Ps_dr, rs_d, Nr = model_update_background_sparse.get_sparse_P_matrix(P_dr)

    # re-initialising this makes it much faster
    # probably large buffers slow it down
    # should look into it
    # could be cpu vs gpu
    mapper = tomograms.Mapper(
        class_c.model.ndim,
        class_c.model.shape[0],
        class_c.xyz,
        class_c.mapping_matrix,
        opencl_stuff['context'],
        opencl_stuff['queue'],
        interpolation=class_c.interpolation_forward
    )

    # get buffer size per-n
    # number of pixels contributing to each model voxel
    # N_n = sum_dsr{i: for K_di>0} delta(n - M_sri)

    rmax = np.max([len(rs) for rs in rs_d])
    rmax = min(rmax, 32)
    max_buf_shape = (mapper.shape[0],) + (rmax,) + (K_di_getter.litpix.max(),)
    print(f'{max_buf_shape=}')
    print(f'{np.prod(max_buf_shape)=}')
    bincount = utils_cl.Bincount_cl(
            N,
            np.prod(max_buf_shape),
            opencl_stuff_cpu['queue'],
            opencl_stuff_cpu['context']
    )

    # N_n = np.zeros((N,), dtype=int)
    mapper.cpu = True
    for d in tqdm(range(dd), desc='calculating buffer sizes'):
        if len(rs_d[d]) > 0:
            Kd_i, pixels = K_di_getter.sparse(d)
            # N_sri = mapper[:, rs_d[d], pixels]
            index = 0
            while index < len(rs_d[d]):
                i = index
                j = min(index+rmax, len(rs_d[d]))
                rs = rs_d[d][i: j]
                N_sri = mapper[:, rs, pixels]
                bincount.add(N_sri)
                index = j
                # N_n += np.bincount(N_sri.ravel(), minlength=N_n.size)

    bincount.queue.finish()
    counts_n = bincount.out

    # assert(np.allclose(counts_n, N_n))

    print(f'{np.sum(counts_n)=}')

    # save
    fnam = f'class_{class_c.class_id}_buffersize_{args.data_chunk}.h5'
    fnam = cachedir.joinpath(fnam)
    with h5py.File(fnam, 'w') as f:
        f['counts_n'] = counts_n
        f['d0'] = d0
        f['d1'] = d1

    logger.info('update_I_buffer_size (stop)')
