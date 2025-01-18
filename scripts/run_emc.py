from context import emc2

from emc2 import utils 
from emc2 import utils_cl

from emc2 import data_getter
from emc2 import geometry
from emc2 import tomograms
from emc2 import model
from emc2 import probability
from emc2 import model_update
from emc2 import input_output
from emc2 import orientations

from concurrent.futures import ThreadPoolExecutor
import pyopencl as cl

import time
import argparse
import os
import sys
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description="""
    Calculate probability matrix
    and update models
    """)
    parser.add_argument('config', type=str, help='configuration file name')
    args = parser.parse_args()
    return args


#def main():
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
    
    # set absolute path to iteration_info.h5
    config.update(
        input_output.set_iteration_info_fnam(config['working_directory'])
    )
    
    # get current iteration number (if continuing from previous calculation)
    config['iteration'] = input_output.get_iteration_number(config)
    
    # get pixel mask etc.
    print(f'\ncalculating mask and pixel geometry')
    config.update(geometry.geometry(**config))
    
    print(f'\nloading sparse frames:')
    K_di = data_getter.Data_getter(**config)
    config['frames'] = K_di.shape[0]
    config['pixels'] = K_di.shape[1]
    
    # initialise models
    print(f'\nloading models:')
    models_I = model.Models(**config)
    
    # load opencl get context and queue
    print(f'\nloading opencl context and devices:')
    config.update(utils_cl.opencl_init())
    
    # initialise tomogram objects 
    # one for each class
    dimensions_c      = utils.int_to_list(config['models'], config['dimensions'],     'dimensions')
    rotation_orders_c = utils.int_to_list(config['models'], config['rotation_order'], 'rotation_order')
    symmetry_c        = utils.int_to_list(config['models'], config['symmetry']      , 'symmetry')
    symmetry_c        = utils.int_to_list(config['models'], config['symmetry']      , 'symmetry')
    
    # make rotation matrices for each
    # unique dimension rotation order pair
    dr = set(zip(dimensions_c, rotation_orders_c))
    
    rotation_matrices = {}
    for (d, r) in dr:
        rotation_matrices[(d, r)] = orientations.get_rotation_matrices(
            queue   = config['queue'], 
            context = config['context'], 
            rotation_order = r,
            dimensions     = d
        )
    
    # now make unique transformation matrices
    # assume pointing fluctuations are the same for 
    # all classes (for now)
    r_offsets = []
    if config['pointing_fluctuations'] :
        N, step = config['pointing_fluctuations'] 
        for n in (np.arange(N) - (N//2)):
            for m in (np.arange(N) - (N//2)):
                r_offsets.append([n * step, m * step, 0])
    else :
        r_offsets.append([0, 0, 0])

    # make transformation parameters
    # n_si = A_sr . (r-dr) / |r-dr| + b_sr
    # A_sr = S_s . R_r / wav dq
    # b_sr = - ( A_sr . (0, 0, 1) + i0)
    dr = set(zip(dimensions_c, rotation_orders_c))
    
    
    """
    # initialise tomograms
    print(f'\nInitialising tomograms:')
    W_ri = tomograms.Tomograms(models_I, cpu = False, **config)
    config['rotations'] = W_ri.shape[0]
    
    # get cpu context and queue
    cl_cpu_stuff = {}
    cl_cpu_stuff.update(utils_cl.opencl_init_cpu(0))
    
    print(f'\nCompiling cpu code for offset and normalisation of P_dr')
    cl_cpu_code = cl.Program(cl_cpu_stuff['context'], probability.code.format(rotations = config['rotations'])).build()
    cl_cpu_stuff['cl_code'] = cl_cpu_code
    
    D, R, I = config['frames'], config['rotations'], config['pixels']
    
    P_dr = None
    for iteration in range(config['iterations']):
        print(f'\n\n*************** iteration {iteration} ***************\n')
        
        # probability matrix
        # ------------------
        config['beta'] = utils.get_beta(**config)
        print(f'{config["beta"]=}')
        
        P_dr, wsums_r, rmax_d, Pmax_d, occupancy_dc, Q_d, P_thresh = probability.calc_P(K_di, W_ri, cl_cpu_stuff, **config)
        #P_dr = np.ones((D, R))
        #wsums_r = np.ones((R,))
        
        assert(np.all(np.isfinite(P_dr)))
        
        executor = ThreadPoolExecutor()
        
        save_it_event = executor.submit(
            utils.save_iteration_info, P_dr, Pmax_d, Q_d, 
            rmax_d, W_ri.class_r, W_ri.orientation_r, occupancy_dc, **config
        )
        
        # update models
        # -------------
        Is, w_d = model_update.calc_I(K_di, P_dr, wsums_r, models_I, **config)
        
        models_I.I   = Is
        models_I.w_d = w_d
        
        save_it_event.result()
        executor.submit(utils.save_model_slices, models_I, **config)
        
        W_ri.load_models(models_I)
                
        executor.shutdown()

        config['iteration'] += 1
    
    if P_dr is not None : 
        utils.save_prob(P_dr, wsums_r, W_ri.class_r, **config)
        utils.save_models(models_I, **config)
        
    """
