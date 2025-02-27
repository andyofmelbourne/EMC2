import argparse
from pathlib import Path
import numpy as np
import h5py

import context
from emc2 import utils
from emc2 import get_script_logger


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""save model slices to iteration info"""
    )
    parser.add_argument(
        'class_files',
        type=str,
        nargs='+',
        help='file name class file(s)'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    working_directory = Path(args.class_files[0]).parent
    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('save_model_slices (start)')

    models = []
    class_ids = []

    # load models and sort by class_id
    for class_file in args.class_files:
        with h5py.File(class_file, 'r') as f:
            models.append(f['model'][()])
            class_ids.append(f['class_id'][()])
            dq = f['dq'][()]

    i = np.argsort(class_ids)
    models_sorted = [models[j] for j in i]

    # save slices in iteration info
    working_directory = Path(args.class_files[0]).parent

    utils.save_model_slices(
        models_sorted,
        dq,
        working_directory
    )

    logger.info('save_model_slices (stop)')
