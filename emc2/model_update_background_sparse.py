"""
    dQ / dw_d = sum_ri P_dr C_i W_ri (K_di / F_dri - 1)
              = sum_ri P_dr C_i W_ri K_di / (w_d C_i W_ri + B_di) - sum_ri P_dr C_i W_ri
                
              = sum_ri P_dr K_di / (w_d + B_di / (C_i W_ri)) - sum_ri P_dr C_i W_ri
              = sum_j a_j / (x + b_j) - c = 0
    
    a_j = P_dr K_di for P_dr > 0 
    
    wmax = sum_ri P_dr K_di / sum_ri P_dr C_i W_ri
         = sum_i K_di / sum_ri P_dr Wsums_r

assume sparse P_dr
we need:
    a_j = {P_dr K_di}         for non-zero elements
    b_j = {B_di / (C_i W_ri}  for above elements
    c_d = sum_r P_dr wsums_r  
"""

import numpy as np
from tqdm import tqdm


def w_update(P_dr, K_di, B_di, W_ri, Wsums_r, **config):
    """
    assume sparse P_dr and K_di
    we need:
        a_j = {Pd_r Kd_i}         for non-zero elements
        b_j = {Bd_i / (C_i W_ri}  for above elements
        c_d = sum_r Pd_r Wsums_r  
    
    calculate W_ri on cpu? give it a go
    
    or calculate W_ri on gpu with r and i list
    """
    P_thresh = 0.01
        
    D, I = K_di.shape
    R    = P_dr.shape[1]
    
    C   = config['C']
    w_d = np.zeros(D, dtype = float)
    
    # np accelerates this over cpus
    for i in tqdm(range(1), desc = 'calculating c_d = sum_r P_dr Wsums_r'):
        c_d = np.dot(P_dr, Wsums_r)

    # make sparse P_dr
    rs_d  = []
    Ps_dr = []
    Nr    = 0
    for d in tqdm(range(D), desc = 'generating sparse P-matrix'):
        rs = np.where(P_dr[d] > (P_thresh * P_dr[d].max()))[0]
        Ps_dr.append(P_dr[d][rs])
        rs_d.append(rs)
        Nr += len(rs)

    print(f'average number of tomograms per frame: {Nr/D:.3f}')
    
    # cpu 
    for d in tqdm(range(D), desc = 'updating w_d'):
        #K = K_di[d:d+1, :]
        Kd_i, pixels = K_di.sparse(d)
        Bd_i         = B_di.sparse(d, pixels)
        C_i          = C[pixels]
        a            = np.outer(Ps_dr[d], Kd_i)
        W            = W_ri[rs_d[d], pixels]
        #b = np.outer(1/C_i * W_ri, Bd_i)
    
    

    

def w_update_old(P_dr, K_di, B_di, W_ri, Wsums_r, **config):
    """
    split over frames
    
    dQ / dw_d = sum_ri P_dr C_i W_ri (K_di / F_dri - 1)
              = sum_ri P_dr C_i W_ri K_di / (w_d C_i W_ri + B_di) - sum_ri P_dr C_i W_ri
                
              = sum_ri P_dr K_di / (w_d + B_di / (C_i W_ri)) - sum_ri P_dr C_i W_ri
              = sum_j a_j / (x + b_j) - c = 0
    
    a_j = P_dr K_di for P_dr > 0 
    
    wmax = sum_ri P_dr K_di / sum_ri P_dr C_i W_ri
         = sum_i K_di / sum_ri P_dr Wsums_r
    """
    d0, d1, dd = utils.chunker_mpi(size, K_di.shape[0])
    d0, d1, dd = d0[rank], d1[rank], dd[rank]
    
    mask = np.zeros(K_di.shape[1], dtype = bool)
    
    w_d = np.zeros(dd, dtype = float)

    # maximum buffer size 
    PWC   = 0.
    C = config['C']

    # precalculate PCW
    PWC = np.dot(P_dr[d0:d1, :], Wsums_r)

    for d in tqdm(range(d0, d1), desc='solving for w_d', disable = quiet):
        index = 0
        
        rs = np.where(P_dr[d] > 0)[0]
        K  = K_di[d:d+1, :][0]
        B  = B_di[d:d+1, :][0]
        
        pixels = np.where(K>0)[0]
        
        # sparse W_ri
        mask.fill(0)
        mask[pixels] = True
        W_ri.update_mask(mask, **config)
         
        # we are not set up for non-continuous rs
        W = W_ri[:, :][rs]
        
        #r, i = np.ix_(rs, pixels)
        r, i   = np.where(W>0.)
        r0, i0 = rs[r], pixels[i]
        
        #print(f'{pixels.shape=} {rs.shape=} {r.shape=} {i.shape=}')
        
        a  = (P_dr[d][r0] * K[i0]).ravel()
        b  = (B[i0] / (C[i0] * W[(r, i)])).ravel()
        
        w_d[d-d0] = utils.solve_axbc(a, b, PWC[d-d0])
    
    w_d = np.concatenate(comm.allgather(w_d))
    assert(not np.any(np.isnan(w_d)))
    assert(np.any(w_d>0))
    return w_d
