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
        '--n_chunk',
        type=int,
        default=0,
        help='calculate for a subset of model voxels'
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
    logger.info('update_I_solve_part (start)')

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

    fnam = f'class_{class_c.class_id}_abn.h5'
    fnam = cachedir.joinpath(fnam)

    """
    s = model_update_background_sparse.Solve_I_mp(
        class_c.model.shape,
        class_c.symmetry,
        fnam
    )
    I_n = s.solve()
    """

    I_n = model_update_background_sparse.solve_I_part(
        class_c.model.shape,
        class_c.symmetry,
        fnam,
        args.n_chunk,
        args.n_chunks,
    )

    if args.output is None:
        output = f'class_{class_c.class_id}_model_{args.n_chunk}.h5'
        output = cachedir.joinpath(output)
    else:
        output = args.output

    mask = I_n > 0
    inds = np.where(mask)[0]

    # save
    with h5py.File(output, 'w') as f:
        f['model'] = I_n[mask]
        f['inds'] = inds

    logger.info('update_I_solve_poart (stop)')
