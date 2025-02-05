from . import utils

from tqdm import tqdm
import numpy as np
import sys
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    quiet = False
else :
    quiet = True

def flatten(xss):
    return [x for xs in xss for x in xs]

def w_update(P_dr, K_di, B_di, W_ri, Wsums_r, **config):
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
        
def model_update(K_id, B_id, w_d, P_dr, M_ri, models_I, background_classes = [], **config):
    # mpi over n
    c0, c1, dc = utils.chunker_mpi(size, len(models_I.I))
    c0, c1, dc = c0[rank], c1[rank], dc[rank]
    
    P_rd = np.ascontiguousarray(P_dr.T)
    
    # loop over n-chunks to reduce the number of M_ri calls
    nchunks = 64
    M     = 10*1024**2
    PK    = np.zeros((nchunks, M,), dtype = float)
    BwC   = np.zeros((nchunks, M,), dtype = float)
    PwC   = np.zeros(nchunks, dtype = float)
    index = np.zeros(nchunks, dtype = int)
    
    wP_r = np.dot(P_rd, w_d)
    assert(not np.any(np.isnan(wP_r)))
    
    skipped_rs     = 0
    skipped_frames = 0
    skipped_no_photons = 0
    
    Is = []
    my_classes = []
    for c in range(c0, c1):
        N  = models_I.I[c].size
        Ic = np.zeros(N, dtype = float)
        Is.append(Ic.copy())
        my_classes.append(c)
    
    for c in tqdm(range(c0, c1), desc = 'updating models', disable = quiet):
        if c in background_classes :
            continue    
        
        # find r's for this class
        rs = np.where((M_ri.class_r == c) * (wP_r > 0))[0]
        
        skipped_rs += np.sum(wP_r == 0)
        
        N  = models_I.I[c].size
        
        for n0, n1, dn in tqdm(utils.chunker(nchunks, N), disable = quiet, leave = False):
            index.fill(0)
            PwC.fill(0)
            for r in tqdm(rs, disable = quiet, leave = False):
                # calculate mapping
                # (no. of symmetries, dr, i's)
                m = M_ri[r:r+1, :][:, 0, :]
                
                # determine frames to skip
                wP     = w_d * P_rd[r, :]
                frames = np.where(wP>0)[0]
                
                skipped_frames += np.sum(wP == 0)
                
                if len(frames) == 0:
                    continue
                
                for n in range(n0, n1):
                    ni = n-n0
                    pixels = np.concatenate([np.where(mm == n)[0] for mm in m])
                     
                    C        = config['C'][pixels]
                    PwC[ni] += wP_r[r] * np.sum(C)
                    
                    K = K_id[pixels, frames]
                    B = B_id[pixels][:, frames]
                     
                    #print(frames.shape, pixels.shape, K.shape, file = sys.stdout)
                    i, d   = np.where(K)
                    d0, i0 = frames[d], pixels[i]
                    
                    if len(i0) == 0 :
                        skipped_no_photons += 1
                        continue
                    
                    I = len(i)
                    j = index[ni]
                    
                    assert(np.all((index+I) < M))
                    PK[ni,  j: j + I] = P_rd[r, d0] * K[(i, d)]
                    BwC[ni, j: j + I] = B[(i, d)] / (w_d[d0] * C[i])
                    index[ni] += I
                 
            for n in range(n0, n1):
                ni = n-n0
                j  = index[ni]
                a  = PK[ni, :j]
                b  = BwC[ni, :j]
                cc = PwC[ni]
                out = utils.solve_axbc(a, b, cc, fill_value = 0., ftol = 1e-3, xtol = 1e-3, maxiters = 1000, algorithm = 'Newton')
                Is[c-c0][n] = out

    print(f'{rank=} {skipped_no_photons=} {skipped_frames=} {skipped_rs=}')
        
    # unflatten
    for c in range(c0, c1):
        Is[c-c0] = Is[c-c0].reshape(models_I.I[c].shape)
    
    print(f'rank {rank} done, gathering models...')
    sys.stdout.flush()
    
    comm.barrier()
    
    I       = comm.allgather(Is)
    classes = comm.allgather(my_classes)
    
    print(f'{rank=} gathering finished...')
    sys.stdout.flush()
    
    I       = flatten(I)
    classes = flatten(classes)
    return I

        
        
