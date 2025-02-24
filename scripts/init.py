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
from tqdm import tqdm
from pathlib import Path
import h5py

from context import emc2
from emc2 import input_output
from emc2 import geometry
from emc2 import data_getter
from emc2 import orientations
from emc2 import utils_cl
from emc2 import symmetry
from emc2 import classes
from emc2 import utils
from emc2 import get_script_logger


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

    # load config file
    config.update(
        input_output.load_config(args.config)
    )

    # set working directory as the directory in which config.py resides
    config.update(
        input_output.set_working_directory(args.config)
    )

    logger = get_script_logger.get_logger(
        working_directory=config['working_directory']
    )

    rotation_matrices = {}
    mapping_matrices = {}

    # load opencl get context and queue
    logger.info('loading opencl context and devices')
    config.update(utils_cl.opencl_init())

    if args.class_no is not None:
        config['classes'] = [config['classes'][args.class_no]]
        class_numbers = [args.class_no]
    else:
        class_numbers = range(len(config['classes']))

    for ci, c in zip(class_numbers, config['classes']):
        c['cxi_file'] = config['cxi_file']
        c['filter'] = config['filter']
        c['split_frames'] = config['split_frames']
        c['working_directory'] = config['working_directory']

        c_prob = dict(c)

        # get pixel mask etc.
        logger.info(f'calculating mask and pixel geometry for class {ci}')
        c.update(geometry.geometry(**c))

        with h5py.File(c['cxi_file'], 'r') as f:
            c['frame_selection'] = c['filter'](f)

        # get pixel mask, xyz, C for probability matrix.
        key = 'P_mask_padding'
        if key in c:
            logger.info(f'calculating probability pixel mask '
                        f'and pixel geometry for class {ci}')
            c_prob['voxel_cut'] = c[key]
            c_prob.update(geometry.geometry(**c_prob))

            c['P_mask'] = c_prob['mask']
            c['P_C'] = c_prob['C']
            c['P_xyz'] = c_prob['xyz']
        else:
            c['P_mask'] = c['mask']
            c['P_C'] = c['C']
            c['P_xyz'] = c['xyz']

        # this is just to initialise sparse data if needed
        logger.info('loading sparse frames (probability matrix)')
        K_di = data_getter.Data_getter(**c_prob)
        c['ksums'] = K_di.photon_sums

        # this is just to initialise sparse data if needed
        logger.info('start:loading sparse frames')
        K_di = data_getter.Data_getter(**c)
        config['frames'] = K_di.shape[0]
        config['pixels'] = K_di.shape[1]

        # initialise models
        N = c['model_length']
        shape = c['dimensions'] * (N,)
        c['model'] = np.random.random(shape)
        c['relative_fluence'] = np.ones(K_di.shape[0], dtype=float)

        # calculate rotation matrices
        d, r = c['dimensions'], c['rotation_order']
        if (d, r) not in rotation_matrices:
            desc = f'calculating rotation matrices for class {ci}'
            for _ in tqdm([1], desc=desc):
                rotation_matrices[(d, r)] = orientations.get_rotation_matrices(
                    queue=config['queue'],
                    context=config['context'],
                    rotation_order=r,
                    dimensions=d
                ).get()
            logger.info(
                f'generating rotation matrices for dimension {d} and rotation '
                f'order {r}: shape = {rotation_matrices[(d, r)].shape}'
            )
        else:
            logger.info(
                f'already have rotation matrices for '
                f'dimension {d} and rotation_order {r}'
            )

        if c['xyz_offset']:
            t = c['xyz_offset']
            for dr in t:
                assert (len(dr) == 3)

            r_offsets = t
        else:
            r_offsets = [[0, 0, 0]]

        # get symmetry opperators
        S0 = symmetry.get_non_voxel_operators(c['dimensions'], c['symmetry'])

        # make transformation parameters
        # n_si = A_sr . (r-dr) / |r-dr| + b_sr
        # A_sr = S_s . R_r / wav dq
        # b_sr = - ( A_sr . (0, 0, 1) + i0)
        # T = {dr, b, A}
        R0 = rotation_matrices[(d, r)]

        # add third dimension for 2D
        if d == 2:
            R = np.zeros(R0.shape[:1] + (3, 3), dtype=float)
            R[:, :2, :2] = R0
            R[:, 2, 2] = 1

            S = np.zeros(S0.shape[:1] + (3, 3), dtype=float)
            S[:, :2, :2] = S0
            S[:, 2, 2] = 1
        else:
            R = R0
            S = S0

        T_sr = np.zeros(
            (S.shape[0], len(r_offsets) * R.shape[0], 5, 3),
            dtype=np.float32
        )
        orientation_index_r = np.empty(T_sr.shape[1], dtype=np.int32)
        x_offset_r = np.empty(T_sr.shape[1], dtype=np.float32)
        y_offset_r = np.empty(T_sr.shape[1], dtype=np.float32)
        z_offset_r = np.empty(T_sr.shape[1], dtype=np.float32)
        for si, s in enumerate(S):
            index = 0
            for dri, dr in enumerate(r_offsets):
                # R.shape = (M, 3, 3)
                # s.shape = (3, 3)
                # A.shape = (M, 3, 3)
                A = s.dot(R).transpose(1, 0, 2) / (c['wavelength'] * c['dq'])
                b = c['i0'] - A[:, :, 2]
                M = A.shape[0]
                T_sr[si, index:index+M, 0] = dr
                T_sr[si, index:index+M, 1] = b
                T_sr[si, index:index+M, 2:5] = A
                if si == 0:
                    orientation_index_r[index:index+M] = np.arange(M)
                    x_offset_r[index:index+M] = dr[0]
                    y_offset_r[index:index+M] = dr[1]
                    z_offset_r[index:index+M] = dr[2]
                index += M

        c['mapping_matrix'] = T_sr
        c['orientation_index_r'] = orientation_index_r
        c['x_offset_r'] = x_offset_r
        c['y_offset_r'] = y_offset_r
        c['z_offset_r'] = z_offset_r

        # initialise probability matrix
        c['probability_matrix'] = np.empty(
            (K_di.shape[0], T_sr.shape[1]),
            dtype=np.float64
        )

        c['wsums'] = np.zeros((T_sr.shape[1]), dtype=float)

        config['iteration'] = 0

        c['beta'] = utils.get_beta(**config)

        # get beta plan

        c['class_id'] = ci

        class_c = classes.Class(**c)

        # write class file
        fnam = Path.joinpath(Path(c['working_directory']), f'class_{ci}.h5')
        class_c.save(fnam, overwrite=True)
