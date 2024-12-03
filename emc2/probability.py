import numpy as np
import pyopencl as cl
import pyopencl.array 
import math
from tqdm import tqdm
import sys

from . import utils
from . import utils_cl
from .utils import chunker, chunker_mpi
from .tomograms import Tomograms


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    quiet = False
else :
    quiet = True


class Probability():
    """
    over fluence:
        logR_dr = sum_i K_id log(W_ri) 
                - K_d log(sum_i C_i W_ri) 
    
    Benchmarking pyclblast shows that cpu is faster than 
    gpu at 2D @ 2D matrix matmul. I guess the cpu is just
    much faster with that kind of memory access pattern.
    """
    def __init__(self, K_di, W_ri, models_I, **config):
        self.D         = np.int32(K_di.shape[0])
        self.I         = np.int32(W_ri.shape[1])
        self.R         = np.int32(W_ri.shape[0])
        self.W_ri      = W_ri 
        self.K_di      = K_di 
        self.I         = models_I

        # mapping from r --> class index
        self.class_r = W_ri.class_r
        
        self.C      = config['C']
        self.wsums  = np.empty((self.R,), dtype = np.float32)
        self.P      = np.zeros((self.D, self.R), dtype = np.float32)
        self.beta   = utils.get_beta(**config)
        self.rmax   = np.empty((self.D,), dtype = np.uint32)
        self.occ    = np.zeros((self.R,), dtype = float)
        self.gini   = np.zeros((self.D,), dtype = float)
        self.Q      = np.zeros((self.D,), dtype = float)
        self.occ_dc = np.zeros((self.D, config['models']), dtype = float)
        
        self.P_thresh = config['P_thresh']
        
        self.calc_tomo_sums(d_chunk_size = 2048, r_chunk_size = 256)
        
        # logR_dr = \sum_i K_di logW_ri - K_d log(sum_i C_i W_ri)
        # -------------------------------------------------------
        if config['likelihood'] == 'Poisson_fluence_free' and \
           config['frame_model'] == 'basic':
            self.w = self.K_di.photon_sums
            self.wsums2 = np.log(self.wsums)
        
        # logR_dr = \sum_i K_di logW_ri - sum_i C_i W_ri
        # -------------------------------------------------------
        elif config['likelihood'] == 'Poisson' and \
           config['frame_model'] == 'basic':
            self.w = np.ones((self.D,), dtype = float)
            self.wsums2 = self.wsums
        
        # logR_dr = \sum_i K_di logW_ri - w_d sum_i C_i W_ri
        # -------------------------------------------------------
        elif config['likelihood'] == 'Poisson' and \
           config['frame_model'] == 'fluence':
            self.w = self.I.w
            self.wsums2 = self.wsums
        
        else :
            err = f'could not reconcile likelihood {likelihood} with frame_model {frame_model}'
            raise ValueError(err)
    
    def calc_tomo_sums(self, d_chunk_size = 2048, r_chunk_size = 256):
        d_chunk_size = min(d_chunk_size, self.D)
        r_chunk_size = min(r_chunk_size, self.R)
        r_iters = math.ceil(self.R/r_chunk_size)
        d_iters = math.ceil(self.D/d_chunk_size)
        
        # wsums_r = sum_i C_i W_ri
        # ------------------------
        for r0, r1, dr in tqdm(chunker(r_chunk_size, self.R), desc = 'calculating tomogram sums', total = r_iters, disable = quiet):
            W = self.W_ri[r0:r1, :]
            
            self.wsums[r0:r1] = np.sum(self.C * W[:dr], axis=1)
    
    def calc(self, d_chunk_size = 2048, r_chunk_size = 1024):
        d_chunk_size = min(d_chunk_size, self.D)
        r_chunk_size = min(r_chunk_size, self.R)
        r_iters = math.ceil(self.R/r_chunk_size)
        d_iters = math.ceil(self.D/d_chunk_size)
        
        # make a buffer for increased precision
        P = np.zeros((d_chunk_size, self.R), dtype = float)
        
        # logR_dr = \sum_i K_di logW_ri - K_d log(sum_i C_i W_ri)
        # -------------------------------------------------------
        for d0, d1, dd in tqdm(chunker(d_chunk_size, self.D), desc = 'calculating probability matrix', total = d_iters, leave = True, disable = quiet):
            K_di = self.K_di[d0:d1, :]
            #
            P[:] = 0
            for r0, r1, dr in tqdm(chunker(r_chunk_size, self.R), total = r_iters, leave = False, disable = quiet):
                W_ri = self.W_ri.log[r0:r1, :]
                #
                P[:dd, r0:r1] += np.dot(K_di[:dd], W_ri[:dr].T)
                P[:dd, r0:r1] -= self.w[d0:d1, None] * self.wsums2[None, r0:r1]
                    
            self.P[d0:d1, :] = P[:dd, :]
                        
    def normalise(self):
        P = np.zeros((self.R,), dtype = float)
        for d in tqdm(range(self.D), desc = 'normalising probabilities', disable = quiet):
            P[:]    = self.P[d]
            
            rmax    = np.argmax(P)
            logRmax = P[rmax]
             
            P[:]    = np.exp( self.beta * (P - logRmax)) 
            
            # apply threshold before normalisation
            if self.P_thresh :
                threshold = P[rmax] * self.P_thresh
                P[P < threshold] = 0
            
            P      /= np.sum(P)
             
            self.rmax[d]    = rmax
            self.occ       += P
            self.gini[d]   += utils.gini(P)
            self.Q[d]      += np.sum(P * self.P[d])
            self.P[d]       = P
            self.occ_dc[d] += np.bincount(self.class_r, weights = P)


class Probability_background():
    """
    R_dr = \sum_i K_di log F_dri - F_dr
    """
    def __init__(self, K_di, F_dri, **config):
        self.K_di  = K_di
        self.F_dri = F_dri
        
        self.D         = np.int32(F_dri.shape[0])
        self.R         = np.int32(F_dri.shape[1])
        self.I         = np.int32(F_dri.shape[2])
        self.K_di      = K_di 
        
        # for tomogram sums
        self.C     = config['C']
        self.W_ri  = F_dri.W_ri
        self.wsums = np.empty((self.R,), dtype = np.float32)
        
        # mapping from r --> class index
        # should probably put this in config
        self.class_r = F_dri.W_ri.class_r
        
        self.P      = np.zeros((self.D, self.R), dtype = np.float32)
        self.beta   = utils.get_beta(**config)
        self.rmax   = np.empty((self.D,), dtype = np.uint32)
        self.occ    = np.zeros((self.R,), dtype = float)
        self.gini   = np.zeros((self.D,), dtype = float)
        self.Q      = np.zeros((self.D,), dtype = float)
        self.occ_dc = np.zeros((self.D, config['models']), dtype = float)
        
        self.P_thresh = config['P_thresh']
        
    def calc(self, d_chunk_size = 128, r_chunk_size = 32):
        self.W_ri.cpu = True
        Probability.calc_tomo_sums(self)
        self.W_ri.cpu = False
        
        d_chunk_size = min(d_chunk_size, self.D)
        r_chunk_size = min(r_chunk_size, self.R)
        r_iters = math.ceil(self.R/r_chunk_size)
        d_iters = math.ceil(self.D/d_chunk_size)
        
        # make a buffer for increased precision
        P = np.zeros((d_chunk_size, self.R), dtype = float)
        
        # logR_dr = \sum_i K_di log F_dri - F_dri
        # -------------------------------------------------------
        for d0, d1, dd in tqdm(chunker(d_chunk_size, self.D), desc = 'calculating probability matrix', total = d_iters, leave = True, disable = quiet):
            K_di = self.K_di[d0:d1, :]
            #
            P[:] = 0
            for r0, r1, dr in tqdm(chunker(r_chunk_size, self.R), total = r_iters, leave = False, disable = quiet):
                key = (slice(d0, d1, None), slice(r0, r1, None), slice(None))
                P[:dd, r0:r1] += np.sum(self.F_dri.KlogF_F(key, K_di), axis=-1)
                
            self.P[d0:d1, :] = P[:dd, :]

    def normalise(self):
        Probability.normalise(self)
