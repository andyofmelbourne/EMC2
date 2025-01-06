import pyopencl as cl
import pyopencl.array 
import pyclblast

from emc2 import utils 
from emc2 import utils_cl
from emc2 import probability

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

"""
F_dri = C_i w_d W_ri + B_di
F_dr  = sum_i (C_i w_d W_ri + B_di)
F_dr  = w_d sum_i (C_i W_ri) + sum_i (B_di)
F_dr  = w_d Wsums_r + Bsums_d

general:
logR_dr = \sum_i K_di logF_dri - F_dr

over fluence:
logR_dr = \sum_i K_di logF_dri - K_d logF_dr
"""


def calc_P(K_di, F_dri, **config):
    D, I = K_di.shape
    R    = F_dri.shape[1]
    P_dr = np.zeros((D, R), dtype = float)
    
    d_chunk_size = 64
    r_chunk_size = 64
    
    d_iter = tqdm(utils.chunker(d_chunk_size, D), desc = 'calculating dot product K . logF')
    r_iter = tqdm(utils.chunker(r_chunk_size, R), desc = 'looping over r', leave = False)
    
    for _ in tqdm(range(1), desc = 'calculating sum_i F_dri = w_d Wsums_r + Bsums_d'):
        wsums_r = probability.calculate_wsums_r(F_dri.W_ri)
        F_dr    = np.outer(F_dri.w_d, wsums_r) + F_dri.B_di.background_sums[:, None]
        
        if config['likelihood'] == 'Poisson_fluence_free' :
            for d in tqdm(range(D), desc = 'calculating K_d logF_dr', leave = False):
                F_dr[d] = K_di.photon_sums[d] * np.log(F_dr[d])
    
    for d0, d1, dd in d_iter:
        for r0, r1, dr in r_iter:
            s = (slice(d0, d1), slice(r0, r1), slice(None))
            F = F_dri.KlogF_F(s, K_di)
            #P_dr[d0:d1, r0:r1] += np.sum(F_dri[:dd, :dr, :], axis=-1)
    
    return P_dr, wsums_r
            
