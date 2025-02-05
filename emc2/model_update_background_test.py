from . import utils
from . import utils_cl

from tqdm import tqdm
import numpy as np
import sys
import time
import math

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



import pyopencl as cl
import pyopencl.array 

code_cl = """
    __kernel void overlap_PK (
        global int *overlap_n,  
        global int *PK_ri, 
        global int *n_sri,
        const int R,
        const int I,
        const int S
    )
    {
        int r, i, j, n, s, pk;
        
        for (r=0; r<R; r++) {
        for (i=0; i<I; i++) {
        
        pk = PK_ri[r*I + i]; 
        
        if (pk > 0) {
            for (s=0; s<S; s++) {
                j = s * R * I + r * I + i;
                n = n_sri[j] ;
                overlap_n[n] += pk;
        }}}}
    }

    //code.add_to_buffer(queue, (dd,dr,di), (1,1,1), cl.SVM(a_n), cl.SVM(b_n), cl.SVM(a), cl.SVM(b), cl.SVM(m), cl.SVM(index_n), np.int32(dr), np.int32(di), np.int32(m.shape[0])) 
    __kernel void add_to_buffer (
        global float *a_n,  
        global float *b_n, 
        global float *a_dri,
        global float *b_di,
        global int   *n_sri,
        global int   *i_n,
        const int D,
        const int R,
        const int I,
        const int S
    )
    {
        //int d = get_global_id(0);
        //int r = get_global_id(1);
        //int i = get_global_id(2);
        
        int d, r, i, j, n, s, pk;
        float al, bl;
        
        for (d=0; d<D; d++) {
        for (r=0; r<R; r++) {
        for (i=0; i<I; i++) {
        
        al = a_dri[d * R * I + r * I + i]; 
        bl = b_di[d * I + i]; 
        
        printf(" %d ", r * I + i);
        if (al > 0) {
            for (s=0; s<S; s++) {
                //n = n_sri[s * R * I + r * I + i];
                //j = i_n[n] ;
                printf(" %d ", s * R * I + r * I + i);
                //a_n[j] = al;
                //b_n[j] = bl;
                //i_n[n] += 1;
            }
        }}}}
    }
"""




        
def model_update(K_id, B_id, w_d, P_dr, M_ri, models_I, background_classes = [], **config):
    """
    calculate for all d r i:
        P_dr K_di
        B_di / (w_d C_i)
        C_i (sum_d w_d P_dr)
        n_ri
        
    over d r i chunks
    
    then (out of order and atomic memory access)
    for d r i in chunk :
        a[n_ri, index] = PK
        b[n_ri, index] = BwC
        C[n_ri]       += CwP
        index += 1

    is there any benefit to a particular chunking strategy? 
        number of n_ri calculations = d_chunks x R x I
        number of PK   calculations = D x R x I
        number of BwC  calculations = r_chunks x D x I

    So large chunks are good
    no penalty for small i chunks
    n_ri is pretty cheap (gpu)
    I'm guessing large r_chunks are the way to go
    """
    d_chunk_size = 1024
    r_chunk_size = 1024
    i_chunk_size = 128
    
    res = utils_cl.opencl_init_cpu(0)
    queue   = res['queue']
    context = res['context']
    
    code = cl.Program(context, code_cl).build()
    
    D, R, I = K_id.shape[1], P_dr.shape[1], K_id.shape[0]
    print(f'{D=} {R=} {I=} {D*R*I=}')
    
    PK      = np.zeros((d_chunk_size * r_chunk_size * i_chunk_size), dtype = float)
    P_dot_K = np.zeros((r_chunk_size, i_chunk_size), dtype = int)
    BwC     = np.zeros((d_chunk_size * i_chunk_size), dtype = float)
    C       = config['C']
    
    wP_r = np.dot(w_d, P_dr)
    assert(not np.any(np.isnan(wP_r)))
    
    # calculate buffer lengths for each class
    overlaps = []
    for c in tqdm(range(config['models']), desc = 'calculating buffer depth for each class voxel'):
        N = models_I.I[c].size
        overlap = np.zeros((N,), dtype = np.int32)
        for i0, i1, di in tqdm(utils.chunker(i_chunk_size, I), total = math.ceil(I/i_chunk_size), desc = 'update I (i-loop)', leave = False):
            for r0, r1, dr in tqdm(utils.chunker(r_chunk_size, R), total = math.ceil(R/r_chunk_size), desc = 'update I (r-loop)', leave = False):
                
                # check if any r's relate to c
                if not np.any(M_ri.class_r == c) :
                    continue
                
                P_dot_K.fill(0)
                for d0, d1, dd in tqdm(utils.chunker(d_chunk_size, D), total = math.ceil(D/d_chunk_size), desc = 'update I (d-loop)', leave = False):
                    K = K_id[i0:i1, d0:d1].copy()
                    K[K>1] = 1.
                     
                    P = P_dr[d0:d1, r0:r1].copy()
                    P[P>0] = 1. 
                    
                    P_dot_K[:dr, :di] += np.dot(P.T, K.T).astype(int)
                    
                # calculate mapping
                # (no. of symmetries, dr, i's)
                m  = np.ascontiguousarray(M_ri[r0:r1, i0:i1].astype(np.int32))
                pk = np.ascontiguousarray(P_dot_K[:dr, :di].astype(np.int32))
                
                # this seems to work without race conditions
                code.overlap_PK(queue, (1,), (1,), cl.SVM(overlap), cl.SVM(pk), cl.SVM(m), np.int32(dr), np.int32(di), np.int32(m.shape[0])) 
                queue.finish()
            
        overlaps.append(overlap)
        print(f'{np.max(overlap)=} {np.median(overlap)=} {np.sum(overlap)=}')
    
    for c in tqdm(range(config['models']), desc = 'accumulating buffers for each class voxel'):
        N = models_I.I[c].size
        overlap = overlaps[c]
        
        # accumulate buffers a and b
        M = np.sum(overlap)
        assert(2 * M * 4 / 1024**3 < config['max_mem'])
        assert(np.iinfo(np.int32).max > M)
         
        a_n = -np.ones(M, dtype = np.float32)
        b_n = -np.ones(M, dtype = np.float32)
        index_n     = np.zeros(N, dtype = np.int32)
        index_n[1:] = np.cumsum(overlap)[:-1]

        
        for i0, i1, di in tqdm(utils.chunker(i_chunk_size, I), total = math.ceil(I/i_chunk_size), desc = 'update I (i-loop)', leave = False):
            for d0, d1, dd in tqdm(utils.chunker(d_chunk_size, D), total = math.ceil(D/d_chunk_size), desc = 'update I (d-loop)', leave = False):
                K = K_id[i0:i1, d0:d1]
                for r0, r1, dr in tqdm(utils.chunker(r_chunk_size, R), total = math.ceil(R/r_chunk_size), desc = 'update I (r-loop)', leave = False):
                    # check if any r's relate to c
                    if not np.any(M_ri.class_r == c) :
                        continue
                    
                    PK[:dd * dr * di] = (P_dr[d0:d1, r0:r1, None] * K[:, None, :].T).ravel()
                    
                    BwC[:dd * di] = (B_id[i0:i1, d0:d1].T / (w_d[d0:d1, None] * C[None, i0:i1])).ravel()
                    
                    a = PK[:dd * dr * di]
                    b = BwC[:dd * di]
                     
                    m  = np.ascontiguousarray(M_ri[r0:r1, i0:i1].ravel().astype(np.int32))
                    
                    code.add_to_buffer(queue, (1,), (1,), cl.SVM(a_n), cl.SVM(b_n), cl.SVM(a), cl.SVM(b), cl.SVM(m), cl.SVM(index_n), np.int32(dd), np.int32(dr), np.int32(di), np.int32(m.shape[0])) 
                    queue.finish()
                    import sys
                    sys.stdout.flush()
                    sys.exit()
