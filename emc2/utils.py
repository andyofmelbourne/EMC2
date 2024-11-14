import pathlib
import runpy
import numpy as np
import math
import h5py
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_config(path):
    p = pathlib.Path(path)
    
    # returns a dict
    config = runpy.run_path(str(p.absolute()))
    
    return config

class Geom_corr():
    def __init__(self, geom_fnam = '/home/andyofmelbourne/Documents/git_repos/xfel7927/geom/r0600.geom'):
        # can we do this inside __init__?
        import extra_geom
        self.geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fnam)
    
    def apply(self, ar):
        out = self.geom.position_modules(ar)[0]
        return out

class Geom_corr_masked():
    def __init__(self, mask, geom_fnam = '/home/andyofmelbourne/Documents/git_repos/xfel7927/geom/r0600.geom'):
        # can we do this inside __init__?
        import extra_geom
        self.geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fnam)
         
        self.mask  = mask
        self.frame = np.zeros(self.geom.expected_data_shape, dtype = float) 
        self.frame[:] = np.nan
    
    def apply(self, ar):
        if ar.ndim == 2:
            frames = []
            for a in ar :
                self.frame[self.mask] = a
                frames.append(self.frame.copy())
        else :
            self.frame[self.mask] = ar
            frames = self.frame
        out = self.geom.position_modules(np.array(frames))[0]
        return out

def get_beta(
        iteration  = None, 
        iterations = None, 
        beta = None, 
        beta_start = None, 
        beta_stop = None, 
        beta_strategy = None,
        **kwargs):
    
    if beta :
        out = beta
    
    elif beta_strategy == 'exponential':
        beta = (beta_stop / beta_start)**(iteration/(iterations-1)) * beta_start
    
    elif beta_strategy == 'linear':
        beta = np.linspace(beta_start, beta_stop, iterations, endpoint = True)[iteration]
    
    else :
        raise ValueError('could not resolve beta strategy from config')
    
    return beta
    
# thanks to Gaëtan de Menten
# https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def chunker(chunksize, size):
    D      = math.ceil(size/chunksize)
    dstart = np.arange(D) * chunksize
    dstop  = np.clip(dstart + chunksize, 0, size)
    dd     = dstop - dstart
    return zip(np.int32(dstart), np.int32(dstop), np.int32(dd))

def chunker_mpi(size, N):
    ds = np.linspace(0, N, size+1, endpoint=True).astype(int)
    dstart = ds[:-1]
    dstop  = ds[1:]
    dd     = dstop - dstart
    return dstart, dstop, dd

def save_prob(prob, **config):
    """
    Should we save any metadata?
    Should we compress? 
    P is not really sparse unless we threshold
    which we do, so yes
    """
    fnam = os.path.join(config['working_directory'], 'probability_matrix.h5')
    
    P = prob.P
    d_start = config['d_start_mpi'][rank]
    d_stop  = config['d_stop_mpi'][rank]
    D  = config['d_stop_mpi'][-1]
    R  = P.shape[1]
    
    if rank == 0 :
        with h5py.File(fnam, 'w') as f:
            f.create_dataset(
                'P_dr', 
                shape = (D, R),
                dtype = P.dtype,
                chunks = (1, R),
                compression = 'gzip',
                compression_opts = 1
            )
            
            # shared by all processes
            f['wsums_r'] = prob.wsums
    
    comm.barrier()
        
    for r in range(size):
        if r == rank :
            with h5py.File(fnam, 'r+') as f:
                f['P_dr'][d_start : d_stop] = P
        comm.barrier()

def save_models(I, **config):
    """
    assume that every process has all models for now
    """
    fnam = os.path.join(config['working_directory'], 'models.h5')
    if rank == 0 :
        with h5py.File(fnam, 'w') as f:
            f['I'] = I
        

def save_iteration_info(models_I, prob, **config):
    fnam = os.path.join(config['working_directory'], 'iteration_info.h5')

    P = prob.P
    d_start = config['d_start_mpi'][rank]
    d_stop  = config['d_stop_mpi'][rank]
    D       = config['total_frames']
    R       = P.shape[1]
    N       = config['iteration']+1
    
    # initialise or resize datasets
    if rank == 0: 
        if N == 1 :
            with h5py.File(fnam, 'w') as f:
                f['iterations'] = 0
                f.create_dataset('beta',                shape = (1,),   maxshape = (None,),   dtype = np.float32)
                f.create_dataset('occupancy_r',         shape = (1, R), maxshape = (None, R), dtype = prob.occ.dtype, fillvalue = 0)
                f.create_dataset('P_gini_d',            shape = (1, D), maxshape = (None, D), dtype = prob.gini.dtype)
                f.create_dataset('Q_d',                 shape = (1, D), maxshape = (None, D), dtype = prob.Q.dtype)
                f.create_dataset('most_likely_state_d', shape = (1, D), maxshape = (None, D), dtype = prob.rmax.dtype)
        else :
            with h5py.File(fnam, 'r+') as f:
                for key in f.keys():
                    if key != 'iterations':
                        f[key].resize(N, axis=0)
    
    if rank == 0 :
        occupancy = np.empty((R,), dtype = prob.occ.dtype)
    else :
        occupancy = None
    
    comm.Reduce(prob.occ, occupancy, op = MPI.SUM, root = 0) 
    comm.barrier()
    
    for r in range(size):
        if r == rank :
            with h5py.File(fnam, 'r+') as f:
                f['P_gini_d'][N-1, d_start : d_stop]            = prob.gini
                f['Q_d'][N-1, d_start : d_stop]                 = prob.Q
                f['most_likely_state_d'][N-1, d_start : d_stop] = prob.rmax
                
                if rank == 0 :
                    f['iterations'][...]  = N
                    f['beta'][N-1]        = prob.beta
                    f['occupancy_r'][N-1] = occupancy
        comm.barrier()

def get_iterations(**config):
    fnam = os.path.join(config['working_directory'], 'iteration_info.h5')
    if os.exists(fnam):
        with h5py.File(fnam) as f:
            iterations = f['iterations'][()]
    else :
        iterations = 0
    return iterations
