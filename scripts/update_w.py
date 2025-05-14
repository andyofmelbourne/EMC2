import argparse
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

from context import emc2
from emc2 import utils
from emc2 import classes
from emc2 import get_script_logger

"""
Wsum_r = sum_i C_i W_ri

w'_d  = sum_i K_di / sum_r P_dr Wsum_r
"""


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""Update model fluence factors (w_d)

        output is written to each class file"""
    )

    parser.add_argument(
        'class_files',
        nargs='+',
        type=str,
        help='class file names'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    working_directory = Path(args.class_files[0]).parent

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_w (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(
        args.class_files[0],
        skip=['probability_matrix', 'model', 'mapping_matrix']
    )

    if class_c.update_fluence is False:
        logger.info('update_fluence is False, skipping w_d update')
        sys.exit()

    K_d = class_c.ksums
    w_d = np.zeros((K_d.shape[0],), dtype=float)

    # calculate PW_d = sum_r P_dr Wsum_r
    # ----------------------------------
    for class_file in tqdm(args.class_files, desc='calculating fluence terms'):
        with h5py.File(class_file, 'r') as f:
            P_dr = f['probability_matrix'][()]
            wsums_r = f['wsums'][()]
            w_d += np.dot(P_dr, wsums_r)

    w_d = K_d / w_d

    rms = np.mean((w_d - class_c.relative_fluence)**2)**0.5
    logger.info(f'rms difference for w_d : {rms}')
    logger.info(f'{np.mean(w_d)=}')

    # write result
    c_iter = tqdm(args.class_files, desc='writing to:')
    for class_file in c_iter:
        c_iter.set_description(f'writing to: {class_file}')
        with h5py.File(class_file, 'r+') as f:
            f['relative_fluence'][:] = w_d

    logger.info('update_w (stop)')
