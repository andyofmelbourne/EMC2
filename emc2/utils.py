import pathlib
import runpy
import numpy as np
import math
import h5py
import os
import shutil

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
        beta = (beta_stop / beta_start)**(iteration/iterations) * beta_start
    
    elif beta_strategy == 'linear':
        beta = np.linspace(beta_start, beta_stop, iterations, endpoint = True)[iteration]
    
    else :
        raise ValueError('could not resolve beta strategy from config')
    
    return beta

def calc_q(wav, xyz):
    r = np.sum(xyz**2, axis=0)**0.5
    q = xyz / r
    q[2] -= 1
    q /= wav
    return q

def get_chunks(i0, i1, changes):
    """
    say we have an index 'i' where things change at location 'changes'

    and we have a range of values starting at i0 
    and ending at i1-1 that we wish to iterate over
    in chunks such that nothing changes 

    e.g. 
    changes: [2, 7]
    0+diff : [0 0 1 0 0 0 0 1 0]
    thing  : [0 0 1 1 1 1 1 2 2]
                  -------------
                .............
                    +++++++
    -- i0 = 2, i1 = 9
    output = [(2, 7), (7, 9)]

    .. i0 = 1, i1 = 8
    output = [(1, 2), (2, 7), (7, 9)]
    
    ++ i0 = 3, i1 = 7
    output = [(3, 7)]
    """
    #j0s = [i0,] + [ i0 > changes < i1]
    #j1s = [i0 > changes < i1] + [i1,]
    
    m = (changes > i0) * (changes < i1)
    changes_in_range = tuple(changes[m])
    return list(zip((i0,) + changes_in_range, changes_in_range + (i1,)))
        

def int_to_list(N, v, name = 'parameter'):
    """
    if v is a list of length N, do nothing
    if v is an int then broadcast to list
    """
    if isinstance(v, int):
        out = N * [v]
    elif isinstance(v, float):
        out = N * [v]
    elif isinstance(v, str):
        out = N * [v]
    elif hasattr(v, '__len__') and len(v) == N :
        out = v
    else :
        raise ValueError(f'could not parse {name} in configuration file: {v}')

    return out
    
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


def chunker(chunksize, size, offset = 0):
    D      = math.ceil(size/chunksize)
    dstart = np.arange(D) * chunksize + offset
    dstop  = np.clip(dstart + chunksize, 0, size + offset)
    dd     = dstop - dstart
    return list(zip(np.int32(dstart), np.int32(dstop), np.int32(dd)))

def chunker_mpi(size, N):
    ds = np.linspace(0, N, size+1, endpoint=True).astype(int)
    dstart = ds[:-1]
    dstop  = ds[1:]
    dd     = dstop - dstart
    return dstart, dstop, dd

def chunker_mpi_local(chunksize, size, N):
    """
    split a list of length N into 'size' chunks of roughly equal size
    then split the chunks into 'p' sub-chunks with all sub-chunks   
    =< chunksize in size
        
    size = number of mpi processors
    p    = number of local chunks
    N    = total number of elememts to iterate over
    chunksize = the size of the chunks for all but the last chunk (which is <= chunksize)
    
    e.g. chunksize = 3, size = 2, N = 9
    l = 0 1 2 3 4 5 6 7 8
        ------- +++++++++ <- 2 mpi chunks
        +++++ - ##### ... <- 2 mpi chunks x 2 local chunks
    """
    # return an iterator for each rank
    out = []
    dstart_mpi, dstop_mpi, dd_mpi = chunker_mpi(size, N)
    for d0_mpi, d1_mpi, dd_mpi in zip(dstart_mpi, dstop_mpi, dd_mpi):
        out.append(chunker(chunksize, d1_mpi - d0_mpi, d0_mpi))
    return out
        

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

def get_model_slices(Is):
    N = Is[0].shape[0]
    
    slices = []
    for i in Is :
        if i.ndim == 2 :
            slices.append(i)
        elif i.ndim == 3 :
            slices.append(i[N//2])
            slices.append(i[:, N//2])
            slices.append(i[:, :, N//2])
    return np.array(slices)
    

def make_models_2D_image(Is):
    # 3D --> 3 x 2D slices
    
    # calculate grid 
    n = math.ceil(len(Is)**0.5)

    N = Is[0].shape[0]
    
    slices_im = np.zeros((n * N, n * N), dtype = np.float32)
    
    for i in range(n):
        for j in range(n):
            c = n*i + j
            if c < len(Is) :
                slices_im[i*N: (i+1)*N, j*N: (j+1)*N] = Is[c]

    return slices_im

def save_models(I, **config):
    fnam = os.path.join(config['working_directory'], 'models.h5')
    with h5py.File(fnam, 'w') as f:
        for c in range(len(I.I)):
            f[f'model_{c}'] = I.I[c]
        f['relative_fluence'] = I.w
        f['dq']    = I.dq
        f['q_max'] = I.q_max
        f['i0']    = I.i0

def save_model_slices(models_I, **config):
    fnam = os.path.join(config['working_directory'], 'iteration_info.h5')
    
    slices = get_model_slices(models_I.I)
    
    N = config['iteration']
    with h5py.File(fnam, 'r+') as f:
        g = f[f'iteration_{N}']
        k = 'model_slices'
        
        if k in g and g[k].shape == slices.shape :
            g[k][:] = slices
        elif k in g and g[k].shape != slices.shape :
            del g[k] 
        
        if k not in g:
            g.create_dataset(k, data = slices, chunks = slices.shape, compression = 'gzip')
        
        k = 'model_dq'
        if k in g :
            del g[k]
        g[k] = models_I.dq

def save_iteration_info(prob, W_ri, **config):
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
                f['iterations'] = N
                f.create_dataset('beta',   shape = (1,),   maxshape = (None,),   dtype = np.float32)
                f.create_dataset('Q',      shape = (1,),   maxshape = (None,),   dtype = np.float32)
                f.create_dataset('P_gini', shape = (1,),   maxshape = (None,),   dtype = np.float32)
        else :
            keys = ['beta', 'Q', 'P_gini']
            with h5py.File(fnam, 'r+') as f:
                for key in keys:
                    f[key].resize(N, axis=0)
        
        with h5py.File(fnam, 'r+') as f:
            g = f.create_group(f'iteration_{N}')
            g.create_dataset('occupancy_r',         shape = (R,), dtype = prob.occ.dtype, fillvalue = 0)
            g.create_dataset('P_gini_d',            shape = (D,), dtype = prob.gini.dtype)
            g.create_dataset('Q_d',                 shape = (D,), dtype = prob.Q.dtype)
            g.create_dataset('most_likely_state_d', shape = (D,), dtype = prob.rmax.dtype)
            g.create_dataset('most_likely_model_d', shape = (D,), dtype = prob.rmax.dtype)
            g.create_dataset('most_likely_orientation_d', shape = (D,), dtype = prob.rmax.dtype)
            g.create_dataset('occupancy_dc',        shape = (D, config['models']), dtype = prob.occ_dc.dtype)
    
    if rank == 0 :
        occupancy = np.empty((R,), dtype = prob.occ.dtype)
    else :
        occupancy = None
    
    comm.Reduce(prob.occ, occupancy, op = MPI.SUM, root = 0) 

    Q = np.sum(prob.Q)
    Q = comm.reduce(Q, op = MPI.SUM, root=0)

    P_gini = np.sum(prob.gini)
    P_gini = comm.reduce(P_gini, op = MPI.SUM, root=0)
    comm.barrier()
    
    for r in range(size):
        if r == rank :
            with h5py.File(fnam, 'r+') as f:
                g = f[f'iteration_{N}']
                g['P_gini_d'][d_start : d_stop]            = prob.gini
                g['Q_d'][d_start : d_stop]                 = prob.Q
                g['most_likely_state_d'][d_start : d_stop] = prob.rmax
                g['most_likely_model_d'][d_start : d_stop] = W_ri.class_r[prob.rmax]
                g['occupancy_dc'][d_start : d_stop]        = prob.occ_dc
                g['most_likely_orientation_d'][d_start : d_stop] = W_ri.orientation_r[prob.rmax]
                
                if rank == 0 :
                    f['iterations'][...]  = N
                    f['beta'][N-1]        = prob.beta
                    f['Q'][N-1]           = Q / D
                    f['P_gini'][N-1]      = P_gini / D
                    g['occupancy_r'][:]   = occupancy
        comm.barrier()

def get_iterations(**config):
    fnam = os.path.join(config['working_directory'], 'iteration_info.h5')
    if os.path.exists(fnam):
        with h5py.File(fnam) as f:
            iterations = f['iterations'][()]
    else :
        iterations = 0
    return iterations
