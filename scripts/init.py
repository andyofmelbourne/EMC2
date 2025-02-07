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
    pointing_fluctuations
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

def get_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description="""
    Read config file & initialise / check input files for further processing
    """)
    parser.add_argument('config', type=str, help='configuration file name')
    parser.add_argument('--class_no', type=int, help='only process class "class_no"')
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

    rotation_matrices = {}
    mapping_matrices = {}

    # load opencl get context and queue
    print('\nloading opencl context and devices:')
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

        # get pixel mask etc.
        print(f'\ncalculating mask and pixel geometry for class {ci}')
        c.update(geometry.geometry(**c))

        with h5py.File(c['cxi_file'], 'r') as f:
            c['frame_selection'] = c['filter'](f)

        # this is just to initialise sparse data if needed
        print('\nloading sparse frames:')
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
        if (d, r) not in rotation_matrices :
            for _ in tqdm(range(1), desc = f'calculating rotation matrices for class {ci}'):
                rotation_matrices[(d, r)] = orientations.get_rotation_matrices(
                    queue   = config['queue'], 
                    context = config['context'], 
                    rotation_order = r,
                    dimensions     = d
                ).get()
        else :
            print(f'already have rotation matrices for dimension {d} and rotation_order {r}')
        
        r_offsets = []
        if c['pointing_fluctuations'] :
            N, step = c['pointing_fluctuations'] 
            for n in (np.arange(N) - (N//2)):
                for m in (np.arange(N) - (N//2)):
                    r_offsets.append([n * step, m * step, 0])
        else :
            r_offsets.append([0, 0, 0])
        
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
                index += M

        c['mapping_matrix'] = T_sr

        # initialise probability matrix
        c['probability_matrix'] = np.empty(
            (K_di.shape[0], T_sr.shape[1]),
            dtype=np.float32
        )

        c['wsums'] = np.zeros((T_sr.shape[1]), dtype=float)

        config['iteration'] = 0

        c['beta'] = utils.get_beta(**config)

        c['class_id'] = ci

        class_c = classes.Class(**c)

        # write class file
        fnam = Path.joinpath(Path(c['working_directory']), f'class_{ci}.h5')
        class_c.save(fnam, overwrite=True)
