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
from emc2 import model_update
from emc2 import mapping

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description="""
    Update models
    """)
    parser.add_argument('config', type=str, help='configuration file name')
    args = parser.parse_args()
    return args

def get_P(working_directory):
    fnam = os.path.join(working_directory, 'probability_matrix.h5')
    with h5py.File(fnam, 'r') as f:
        P_dr = f['P_dr'][()]
        Wsums_r = f['wsums_r'][()]
    return P_dr, Wsums_r

def main():
    # load config
    args = get_args()
    config = utils.load_config(args.config)
    config['working_directory'] = os.path.abspath(os.path.dirname(config['__file__']))
    config['iteration']         = utils.get_iterations(**config)
    print('********* update_I iteration:', config['iteration'])
    
    # get pixel mask etc.
    config.update(geometry.geometry(**config))
    
    # load opencl get context and queue
    config.update(utils_cl.opencl_init(rank))
    
    # initialise data getter
    # do not split frames over processors
    K_di = data_getter.Data_getter(**config)
    config['frames'] = K_di.shape[0]
    config['pixels'] = K_di.shape[1]
    
    # initialise tomograms
    M_ri = mapping.Mapping(**config)
    #M_ri = tomograms.Mapping(**config)
    config['rotations'] = M_ri.shape[0]
    
    # initialise models
    # skip gpu (waste of memory)
    models_I = model.Models(no_gpu = True, **config)
    
    # initialise update_I calc
    # perhaps we should save P_rd and only load rows of P for each rank...
    P_dr, Wsums_r = get_P(config['working_directory'])
    mu = model_update.Model_update(K_di, M_ri, P_dr, Wsums_r, models_I, **config)
    
    # run main code
    mu.calc()
    
    if rank == 0 :
        utils.save_models(models_I, **config)
        utils.save_model_slices(models_I, **config)
    return models_I.I

if __name__ == '__main__':
    Is = main()
    
    """
    if rank == 0 :
        import pickle
        Is = np.stack(Is, axis=0)
        #pickle.dump(Is, open('temp.pickle', 'wb'))
        Is0 = pickle.load(open('temp.pickle', 'rb'))
        print(np.sum((Is0 - Is)**2)**0.5, np.sum(Is0), np.sum(Is))
        sys.stdout.flush()
        
        import pyqtgraph as pg
        pg.show(Is)
        pg.exec()
    """
