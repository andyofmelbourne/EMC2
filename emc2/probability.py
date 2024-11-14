import numpy as np
import pyopencl as cl
import pyopencl.array 
import math
from tqdm import tqdm
import sys

from . import utils
from . import utils_cl
from .utils import chunker, chunker_mpi



class Probability():
    """
    over fluence:
        logR_dr = sum_i K_id log(W_ri) 
                - K_d log(sum_i C_i W_ri) 
    
    Benchmarking pyclblast shows that cpu is faster than 
    gpu at 2D @ 2D matrix matmul. I guess the cpu is just
    much faster with that kind of memory access pattern.
    """
    def __init__(self, K_di, W_ri, **config):
        self.queue   = config['queue']
        self.context = config['context']
        
        #self.compile()
         
        self.D         = np.int32(K_di.shape[0])
        self.I         = np.int32(W_ri.shape[1])
        self.R         = np.int32(W_ri.shape[0])
        self.W_ri      = W_ri 
        self.K_di      = K_di 
        
        self.C     = config['C']
        self.wsums = np.empty((self.R,), dtype = np.float32)
        self.P     = np.zeros((self.D, self.R), dtype = np.float32)
        self.beta  = utils.get_beta(**config)
        self.rmax  = np.empty((self.D,), dtype = np.uint32)
        self.occ   = np.zeros((self.R,), dtype = float)
        self.gini  = np.zeros((self.D,), dtype = float)
        self.Q     = np.zeros((self.D,), dtype = float)
        
        self.P_thresh = config['P_thresh']
    
    def calc(self, d_chunk_size = 2048, r_chunk_size = 256):
        d_chunk_size = min(d_chunk_size, self.D)
        r_chunk_size = min(r_chunk_size, self.R)
        
        # wsums_r = sum_i C_i W_ri
        # ------------------------
        W       = np.empty((r_chunk_size, self.I), dtype = self.W_ri.dtype)
        r_iters = math.ceil(self.R/r_chunk_size)
        d_iters = math.ceil(self.D/d_chunk_size)
        for r0, r1, dr in tqdm(chunker(r_chunk_size, self.R), total = r_iters):
            W_cl = self.W_ri[r0:r1, :]
            
            cl.enqueue_copy(self.queue, W[:dr], W_cl.data)
            
            self.wsums[r0:r1] = np.sum(self.C * W[:dr], axis=1)
         
        # logR_dr = \sum_i K_di logW_ri - K_d log(sum_i C_i W_ri)
        # -------------------------------------------------------
        # I don't like doing this here
        self.W_ri.set_log(True)
        
        for d0, d1, dd in tqdm(chunker(d_chunk_size, self.D), total = d_iters, leave = True):
            K_di = self.K_di[d0:d1, :]
            #
            for r0, r1, dr in tqdm(chunker(r_chunk_size, self.R), total = r_iters, leave = False):
                W_ri = self.W_ri[r0:r1, :]
                cl.enqueue_copy(self.queue, W[:dr], W_ri.data)
                #
                self.P[d0:d1, r0:r1] += np.dot(K_di[:dd], W[:dr].T)
                self.P[d0:d1, r0:r1] -= self.K_di.photon_sums[d0:d1, None] * np.log(self.wsums[None, r0:r1])
                        
        self.W_ri.set_log(False)
        
        self.normalise()
    
    def normalise(self):
        P = np.zeros((self.R,), dtype = float)
        for d in tqdm(range(self.D), desc = 'normalising probabilities'):
            P[:]    = self.P[d]
            
            rmax    = np.argmax(P)
            logRmax = P[rmax]
             
            P[:]    = np.exp( self.beta * (P - logRmax)) 
            
            # apply threshold before normalisation
            if self.P_thresh :
                threshold = P[rmax] * self.P_thresh
                P[P < threshold] = 0
            
            P      /= np.sum(P)
             
            self.rmax[d]  = rmax
            self.occ     += P
            self.gini[d] += utils.gini(P)
            self.Q[d]    += np.sum(P * self.P[d])
            self.P[d]     = P
