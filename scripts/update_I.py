from context import emc2

from emc2 import utils 
from emc2 import utils_cl

from emc2 import data_getter
from emc2 import geometry
from emc2 import model
from emc2 import model_update

from concurrent.futures import ThreadPoolExecutor
import pyopencl as cl

import time
import argparse
import os
import sys
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm
import h5py


def get_P(working_directory):
    fnam = os.path.join(working_directory, 'probability_matrix.h5')
    with h5py.File(fnam, 'r') as f:
        P_dr = f['P_dr'][()]
        Wsums_r = f['wsums_r'][()]
    return P_dr, Wsums_r

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
    # setup
    # -----
    
    # load config
    args = get_args()

    print(f'\nloading configuration file')
    config = utils.load_config(args.config)
    config['working_directory'] = os.path.abspath(os.path.dirname(config['__file__']))
    print(f'working directory: {config["working_directory"]}')

    if config['restart'] :
        config['iteration'] = 0
    else :
        config['iteration'] = utils.get_iterations(**config) 

    print(f'iteration: {config["iteration"]}')
    
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
    
    # get cpu context and queue
    cl_cpu_stuff = {}
    cl_cpu_stuff.update(utils_cl.opencl_init_cpu(0))

    # probability matrix
    # ------------------
    P_dr, wsums_r = get_P(config['working_directory'])
    config['rotations'] = P_dr.shape[1]
    assert(np.all(np.isfinite(P_dr)))
    
    D, R, I = config['frames'], config['rotations'], config['pixels']
    
    print(f'\n\n*************** iteration {config["iteration"]} ***************\n')

    # update models
    # -------------
    Is, w_d = model_update.calc_I(K_di, P_dr, wsums_r, models_I, **config)
    
    models_I.I   = Is
    models_I.w_d = w_d
    
    utils.save_model_slices(models_I, **config)
    utils.save_models(models_I, **config)
        
