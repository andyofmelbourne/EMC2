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
from emc2 import mapping
from emc2 import model

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
    parser.add_argument('--nproc', type=int, default = 1, help='number of proccessors to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #main()
    
    # load config
    args = get_args()
    config = utils.load_config(args.config)
    config['working_directory'] = os.path.abspath(os.path.dirname(config['__file__']))
    config['iteration']         = utils.get_iterations(**config)
    #config['iteration']         = 0
    
    # get pixel mask etc.
    config.update(geometry.geometry(**config))
    
    # load opencl get context and queue
    config.update(utils_cl.opencl_init())
    
    # initialise data getter
    # will split over frames if config['split']=='frames'
    K_di = data_getter.Data_getter(**config)
    config['frames'] = K_di.shape[0]
    config['pixels'] = K_di.shape[1]

    # initialise models + qmask
    models_I = model.Models(**config)
    
    # initialise tomograms
    W_ri = tomograms.Tomograms(models_I, cpu = True, **config)
    M_ri = mapping.Mapping(models_I, cpu = True, **config)
    config['rotations'] = W_ri.shape[0]
        
    # view tomograms

    mask = np.zeros(config['pixels'], dtype = bool)
    mask[np.random.randint(0, config['pixels'], config['pixels']//4)] = True

    mask_g         = config['mask'].copy()
    mask_g[mask_g] = mask
    geom = utils.Geom_corr_masked(mask_g)
    
    rs = np.sort(np.random.randint(0, 7000, 20))
    
    import pyqtgraph as pg
    N_sri = M_ri[rs, np.where(mask)[0]]
    N_sri = N_sri.reshape(-1, N_sri.shape[-1])
    ims   = geom.apply(N_sri)
    pg.show(ims)
    #pg.exec()


