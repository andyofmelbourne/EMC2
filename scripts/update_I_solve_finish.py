import argparse
import h5py
from pathlib import Path
import numpy as np
import sys

from context import emc2
from emc2 import utils
from emc2 import classes
from emc2 import model_update_background_sparse
from emc2 import get_script_logger
from emc2 import symmetry


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
        '--n_chunks',
        type=int,
        default=1,
        help='number of blocks to split model voxels over'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help="h5 file to write 'model' dataset to. \
        Default is <working_director>/cachdir/class_<c>_model_<chunkno>.h5"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    working_directory = Path(args.class_file).parent
    cachedir = working_directory.joinpath('cachdir')

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_I_sort (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(
        args.class_file,
        skip=['mapping_matrix', 'probability_matrix']
    )

    if class_c.update_model:
        logger.info('update_model is True, updating model')
    else:
        logger.info('update_model is False, skipping model update')
        sys.exit()

    I0_n = class_c.model.copy()
    shape = I0_n.shape
    out_n = np.zeros(I0_n.size, dtype=float)
    model_symmetry = class_c.symmetry

    fnams = []
    for n in range(args.n_chunks):
        fnam = f'class_{class_c.class_id}_model_{n}.h5'
        fnam = cachedir.joinpath(fnam)
        fnams.append(fnam)

        with h5py.File(fnam) as f:
            inds = f['inds'][()]
            out_n[inds] += f['model'][()]

    sym = symmetry.Symmetry(
        shape[0]//2,
        shape,
        model_symmetry
    )

    out_n = sym.apply_symmetry(
        out_n.reshape(shape)
    )

    n_asy = sym.get_asymmetric_unit()

    O_n = np.zeros(out_n.size, dtype=int)
    O_n[n_asy] = 1
    O_n = sym.apply_symmetry(
        O_n.reshape(shape),
    )

    O_n[O_n == 0] = 1
    out_n /= O_n

    I_n = out_n

    rms = np.mean((I0_n - I_n)**2)**0.5
    logger.info(f'rms difference for model {class_c.class_id}: {rms}')
    logger.info(f'class_c.class_id: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    if args.output is None:
        args.output = args.class_file

    # save
    with h5py.File(args.output, 'a') as f:
        if 'model' in f:
            f['model'][:] = I_n
        else:
            f['model'] = I_n

    logger.info('update_I (stop)')
