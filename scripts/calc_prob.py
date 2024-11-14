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
    """)
    parser.add_argument('config', type=str, help='configuration file name')
    parser.add_argument('--nproc', type=int, default = 1, help='number of proccessors to use')
    args = parser.parse_args()
    return args

def main(rank = 0, lock = None, barrier = None):
    # load config
    args = get_args()
    config = utils.load_config(args.config)
    config['iteration']         = utils.get_iterations(**config)
    config['working_directory'] = os.path.abspath(os.path.dirname(config['__file__']))
    
    # get pixel mask etc.
    config.update(geometry.geometry(**config))
    
    # load opencl get context and queue
    config.update(utils_cl.opencl_init(rank))
    
    # initialise models + qmask
    models_I = model.Models(**config)
    
    # initialise data getter
    # will split over frames if config['split']=='frames'
    K_di = data_getter.Data_getter(**config)
    config['frames'] = K_di.shape[0]
    config['pixels'] = K_di.shape[1]
    config['d_start_mpi'] = K_di.d_start_mpi
    config['d_stop_mpi']  = K_di.d_stop_mpi
    config['total_frames']  = K_di.total_frames
    
    # initialise tomograms
    W_ri = tomograms.Tomograms(models_I, **config)
    config['rotations'] = W_ri.shape[0]
    
    # initialise probability calc
    prob = probability.Probability(K_di, W_ri, **config)
    
    # run main code
    prob.calc()
    
    # save
    utils.save_prob(prob, **config)
    utils.save_iteration_info(models_I, prob, **config)

if __name__ == '__main__':
    # load config
    args = get_args()
    
    if args.nproc > 1 :
        lock    = mp.Lock()
        barrier = mp.Barrier(args.nproc)
        jobs = []
        for n in range(args.nproc):
            job = mp.Process(
                target = main, args=(n, lock, barrier)
            )
            job.start()
            jobs.append(job)
        
        for j in jobs:
            j.join()
    else :
        main()
        


