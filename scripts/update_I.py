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
from emc2 import model_update_background
#from emc2 import model_update_background_test as model_update_background
from emc2 import mapping
from emc2 import frames

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
    config['rotations'] = M_ri.shape[0]
    
    # perhaps we should save P_rd and only load rows of P for each rank...
    P_dr, Wsums_r = get_P(config['working_directory'])

    # special case for background classes
    # always update with:
    #   likelihood = 'Poisson' and frame_model = 'basic'
    k = 'background_classes'
    if k not in config or not config[k] :
        config['background_classes'] = []
        
    if config['frame_model'] == 'background':
        B_di     = data_getter.Data_getter_background(K_di)
        models_I = model.Models(no_gpu = False, **config)
        W_ri     = tomograms.Tomograms(models_I, cpu = True, **config)
        
        for i in range(3):
            #w_d = model_update_background.w_update(P_dr, K_di, B_di, W_ri, Wsums_r, **config)
            w_d = models_I.w
            
            # load transposed data
            K_id, B_id = utils.get_transpose(K_di, B_di, working_directory = config['working_directory'])
             
            # maybe we don't need this 
            #del K_di
            #del B_di
            
            # initialise update_I calc
            I = model_update_background.model_update(K_id, B_id, w_d, P_dr, M_ri, models_I, **config)
            
            if rank == 0 : 
                print(f'rms change in w: {np.mean( (w_d - models_I.w)**2 )**0.5}')
                for c in range(len(I)):
                    print(f'rms change in I[{c}]: {np.mean( (I[c] - models_I.I[c])**2 )**0.5}')
            
            models_I.w = np.clip(w_d, 1e-4, None)
            models_I.I = I
            
            models_I.load_models_cl()
            W_ri.models = models_I
        
    else :
        # initialise models
        models_I = model.Models(no_gpu = True, **config)
        
        # initialise update_I calc
        mu = model_update.Model_update(K_di, M_ri, P_dr, Wsums_r, models_I, **config)
        
        # run main code
        I, w_d = mu.calc()
    
    if rank == 0 :
        models_I.w = np.clip(w_d, 1e-4, None)
        for c in range(len(models_I.I)):
            if c not in config['background_classes'] :
                models_I.I[c] = I[c]
        
        for c in tqdm(config['background_classes'], desc='updating background models'):
            rs   = np.where(M_ri.class_r == c)[0]
            if len(rs) == 0 :
                break
            
            ds   = np.where(np.sum(P_dr[:, rs], axis=1) > 0)[0]
            if len(ds) == 0 :
                break
            
            Pb_dr = np.ascontiguousarray(P_dr[np.ix_(ds, rs)])
            
            I = np.zeros(models_I.I[c].shape, dtype = float)
            O = np.zeros(models_I.I[c].shape, dtype = float)
            
            # split d over local chunks
            d_iter = utils.chunker(1024, len(ds))
            Nd = len(list(d_iter))
            
            PK_ri  = np.zeros((len(rs), config['pixels']), dtype = float)
            for d0, d1, dd in tqdm(d_iter, desc = 'looping over d', total = Nd, leave = False):
                PK_ri += np.dot(Pb_dr[d0:d1].T, K_di[ds[d0: d1], :])
                D_ri   = np.outer(np.sum(Pb_dr[d0:d1], axis=0), config['C'])
            
            # calculate pixel mappings
            # enfoce continuity
            assert(np.allclose(np.diff(rs), 1))
            n  = M_ri[rs[0]: 1+rs[-1], :]
            for m in n :
                I += np.bincount(m.ravel(), PK_ri.ravel(), minlength = I.size).reshape(I.shape)
                O += np.bincount(m.ravel(), D_ri.ravel(),  minlength = O.size).reshape(O.shape)
            
            O[O==0] = 1
            I /= O
            models_I.I[c] = I.copy()
    
    if rank == 0 :
        utils.save_models(models_I, **config)
        utils.save_model_slices(models_I, **config)
    
    return models_I

if __name__ == '__main__':
    m = main()
    #w_d = main()
    
    #if rank == 0 :
    #    I   = m.I
    #    w_d = m.w
    #    import pyqtgraph as pg
    #    pg.plot(w_d)
    #    for c in range(len(I)):
    #        pg.show(I[c])
    #    pg.exec()
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
