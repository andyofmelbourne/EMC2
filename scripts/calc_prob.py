"""
parallelise over frames
trust that numpy already does multithreaded np.dot
mpi or multiprocessing?
I can't get opencl to behave with multiprocessing
so mpi it is
"""
from context import emc2

from emc2 import utils 
from emc2 import utils_cl
from emc2 import data_getter
from emc2 import probability
from emc2 import geometry
from emc2 import tomograms
from emc2 import model
from emc2 import frames

import argparse
import os
import sys
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description="""
    Calculate probability matrix
    """)
    parser.add_argument('config', type=str, help='configuration file name')
    args = parser.parse_args()
    return args

def main():
    # load config
    args = get_args()
    
    config = utils.load_config(args.config)
    config['working_directory'] = os.path.abspath(os.path.dirname(config['__file__']))
    config['iteration']         = utils.get_iterations(**config)
    if rank == 0 : print('iteration:', config['iteration'])
    
    # get pixel mask etc.
    config.update(geometry.geometry(**config))

    if rank == 0 : print(config['C'].min(), config['C'].max())
    
    # load opencl get context and queue
    config.update(utils_cl.opencl_init(rank))
    
    # initialise data getter
    # ----------------------
    # split frames over mpi ranks
    config['mpi_split_frames'] = True
    
    K_di = data_getter.Data_getter(**config)
    config['frames'] = K_di.shape[0]
    config['pixels'] = K_di.shape[1]

    # for mpi
    config['d_start_mpi']   = K_di.d_start_mpi
    config['d_stop_mpi']    = K_di.d_stop_mpi
    config['total_frames']  = K_di.total_frames
    
    # initialise models
    models_I = model.Models(**config)
    
    # initialise probability calc
    if config['frame_model'] == 'background':
        # initialise tomograms
        W_ri = tomograms.Tomograms(models_I, cpu = False, **config)
        config['rotations'] = W_ri.shape[0]
         
        B_di  = data_getter.Data_getter_background(K_di)
        F_dri = frames.Frames(B_di, models_I.w, W_ri, **config)
    
        prob = probability.Probability_background(K_di, F_dri, **config)
    else :
        # initialise tomograms
        W_ri = tomograms.Tomograms(models_I, cpu = True, **config)
        config['rotations'] = W_ri.shape[0]
         
        prob = probability.Probability(K_di, W_ri, models_I, **config)
    
    # run main code
    prob.calc()
    prob.normalise()
    
    # save
    if config['iteration'] == 0 : 
        w = comm.gather(models_I.w, root = 0)
        
        if rank == 0 :
            print([i.shape for i in w])
            models_I.w = np.concatenate(w, axis=0)
            utils.save_models(models_I, **config)
        
    utils.save_prob(prob, **config)
    utils.save_iteration_info(prob, W_ri, **config)

if __name__ == '__main__':
    main()

