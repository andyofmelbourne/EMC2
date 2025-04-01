"""
read config file & initialise / check input files for further processing

class_0.h5
    model
    relative_fluence
    dq
    rotation_order
    maximise

    update_fluence
    update_model

    probability_matrix
    beta
    P_thresh
    likelihood
    frame_model

    cxi_file
    frame_list
    polarisation
    split_frames

    mapping (S, R)
    symmetry
    interpolation_forward
    xyz_offset
"""

import argparse
import numpy as np
import h5py

from context import emc2
from emc2 import input_output
from emc2 import utils
from emc2 import get_script_logger
from emc2 import init


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read config file & initialise / check input"
                    "files for further processing"
    )

    parser.add_argument(
        'config',
        type=str,
        help='configuration file name'
    )

    parser.add_argument(
        '--class_no',
        type=int,
        help='only process class "class_no"'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # load config
    args = get_args()

    config = {}

    # set working directory as the directory in which config.py resides
    config.update(
        input_output.set_working_directory(args.config)
    )

    logger = get_script_logger.get_logger(
        working_directory=config['working_directory']
    )
    logger.info('init (start)')

    # load config file
    logger.debug('loading configuration file (start)')
    config.update(
        input_output.load_config(args.config)
    )
    logger.debug('loading configuration file (stop)')

    # update each class dict with global parameters
    for c in config['classes']:
        c['cxi_file'] = config['cxi_file']
        c['filter'] = config['filter']
        c['split_frames'] = config['split_frames']
        c['working_directory'] = config['working_directory']

    # generate a unique class id for each class
    for ci, c in enumerate(config['classes']):
        c['class_id'] = ci

    # add frame filter to spped up data checking
    with h5py.File(config['cxi_file']) as f:
        config['filter'] = config['filter'](f)

    config['frames'] = np.sum(config['filter'])

    config['iteration'] = 0

    beta = utils.get_beta(**config)

    # load opencl get context and queue
    # have to import pyopencl after fork
    from emc2 import utils_cl
    logger.info('loading opencl context and devices')
    config.update(utils_cl.opencl_init())

    rotation_matrices = {}

    for class_c in config['classes']:
        class_c['queue'] = config['queue']
        class_c['context'] = config['context']
        class_c['frames'] = config['frames']
        class_c['beta'] = beta

        init.add_geometry(class_c)

        init.init_model(class_c)

        init.init_fluence(class_c)

        init.init_mapping(class_c, rotation_matrices)

        T_sr = class_c['mapping_matrix']

        # initialise probability matrix
        class_c['probability_matrix'] = np.empty(
            (class_c['frames'], T_sr.shape[1]),
            dtype=np.float64
        )

        class_c['wsums'] = np.zeros((T_sr.shape[1]), dtype=float)
        class_c['P_wsums'] = np.zeros((T_sr.shape[1]), dtype=float)

        init.check_data(class_c)

        init.save_classes(class_c)

    logger.info('init (stop)')
