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
        help='calculate for a subset of frames'
    )

    parser.add_argument(
        '--n_chunks',
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


if __name__ == '__main__':
    args = get_args()

    working_directory = Path(args.class_file).parent
    cachedir = working_directory.joinpath('cachdir')

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_I_merge (start)')

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

    I_n = model_update_background_sparse.solve_I(
        class_c.model.shape,
        class_c.symmetry,
        fnam
    )

    I0_n = class_c.model.copy()

    rms = np.mean((I0_n - I_n)**2)**0.5
    logger.info(f'rms difference for model {class_c.class_id}: {rms}')
    logger.info(f'class_c.class_id: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    # test
    # I_n = np.clip(I_n, 1e-8, None)

    if args.output is None:
        args.output = args.class_file

    # save
    """
    with h5py.File(args.output, 'a') as f:
        if 'model' in f:
            f['model'][:] = I_n
        else:
            f['model'] = I_n
    """

    logger.info('update_I (stop)')
