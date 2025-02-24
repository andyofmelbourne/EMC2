"""
Wsum_r = sum_i C_i W_ri

Basic:
    N_ri  = sum_d P_dr K_di
    D_ri  = C_i sum_d P_dr

Fluence:
    N_ri  = sum_d P_dr K_di
    D_r   = C_i sum_d w_d P_dr

    a_d   = sum_i K_di
    b_d   = sum_r P_dr Wsum_r
    w'_d  = a_d / b_d

Fluence free:
    N_ri = Wsum_r sum_d P_dr K_di
    D_ri = C_i sum_d P_dr K_d

------------------------------------

merge tomograms (then I):
    W'_ri = N_ri / D_ri

    A_n = sum_ri M^-1(W'_ri, r, i)_n
    B_n = sum_ri M^-1(1,     r, i)_n
    I_n = A_n / B_n

merge I:
    A_n = sum_ri M^-1(N_ri, r, i)_n
    B_n = sum_ri M^-1(D_ri, r, i)_n
    I_n = A_n / B_n

Loop over model class (this reduces out-of-order memory operations)
Would be nice to skip frames with low P_dr values but this complicates
dot product which is about 40 times faster

Perform coordinate mappings on gpu
nearest: i --> n
    n0 = round(i0 + (R_r . q_i)_0 / dq)
    n1 = round(i0 + (R_r . q_i)_1 / dq)
    n2 = round(i0 + (R_r . q_i)_2 / dq)

    n = N^2 n0 + N n1 + n2

    M^-1(x, r, i)_m = x delta(n - m)

Perform sum in I on cpu (out-of-order memory operations)

w'_d  = sum_i K_di / sum_r P_dr Wsum_r
"""

import numpy as np
from tqdm import tqdm
from emc2 import utils 
from emc2 import utils_cl
from emc2 import mapping
from emc2 import symmetry
from concurrent.futures import ThreadPoolExecutor
import time

import pyopencl as cl
import pyopencl.array 
import pyclblast


class dummy_event():
    def wait():
        pass

    def result():
        pass

def calculate_K_dot_P_gpu(P_dr, K_di, queues, out = None, r_chunk_size = 256, d_chunk_size = 256, quiet = True):
    """
    assume P_dr in cpu memory but very large
    assume K_di is very large but kept in sparse format on cpu 
    so K_di takes time to call

    but P_dr and K_di can just be normal numpy arrays

    can be optimised by reversing loop and using beta parameter in gemm
    """
    D, I  = K_di.shape
    R     = P_dr.shape[1]
    if out is None :
        PK_ri = np.zeros((R, I), dtype = float)
    else :
        PK_ri = out

    r_chunk_size = min(r_chunk_size, R)
    d_chunk_size = min(d_chunk_size, D)
    
    PK_ri_cl = [cl.array.empty(queue, (r_chunk_size, I), dtype = np.float32) for queue in queues]
    
    def get(PK_ri, PK_ri_cl, r0, r1, queue):
        # causes memory leak!
        #PK = PK_ri_cl.get()
        
        PK = utils_cl.to_cpu(PK_ri_cl, queue = queue)
        PK_ri[r0:r1, :] += PK[:r1-r0, :]
    
    q = 0 
    max_depth  = 2 * len(queues)
    executor   = ThreadPoolExecutor()
    dot_events = []
    get_events = []
    d_iter = tqdm(utils.chunker(d_chunk_size, D), desc = 'calculating dot product sum_d P_dr . K_di', disable = quiet)
    
    for d0, d1, dd in d_iter:
        K    = K_di[d0:d1, :]
        K_cl = [utils_cl.to_gpu(K, queue = queue, dtype = np.float32) for queue in queues]
        
        r_iter = tqdm(utils.chunker(r_chunk_size, R), desc = 'looping over r', leave = False, disable = quiet)
        for r0, r1, dr in r_iter:
            queue = queues[q]
            
            P_cl = utils_cl.to_gpu(P_dr[d0:d1, r0:r1], queue = queue) 
            
            # non-blocking
            dot_events.append(pyclblast.gemm(queue, dr, I, dd, P_cl, K_cl[q], PK_ri_cl[q], dr, I, I, a_transp = True, b_transp = False))
            
            # non-blocking
            get_events.append(executor.submit(get, PK_ri, PK_ri_cl[q], r0, r1, queue))
            
            # blocking
            if len(dot_events) == max_depth:
                dot_events[0].wait()
                get_events[0].result()
                dot_events.pop(0)
                get_events.pop(0)
                
            if r1 == R:
                executor.shutdown()
                
                for queue in queues:    
                    queue.finish()
                
                # re-init
                executor   = ThreadPoolExecutor()
                dot_events = []
                get_events = []
                
            q = (q+1) % len(queues)

    executor.shutdown()
    
    return PK_ri


def calc_I(K_di, P_dr, wsums_r, models_I, **config):
    D, I = K_di.shape
    R    = P_dr.shape[1]
    
    # update w_d
    # w'_d  = sum_i K_di / sum_r P_dr Wsum_r
    # --------------------------------------
    w_d = np.zeros(D, dtype = float)
    def dot(w_d, K_d, P_dr, wsums_r):
        np.dot(P_dr, wsums_r, out = w_d)
        w_d[:] = K_d / w_d
    
    print('\nUpdating fluence estimates')
    executor = ThreadPoolExecutor()
    w_event    = executor.submit(dot, w_d, K_di.photon_sums, P_dr, wsums_r)
    
    # calculate one of:
    # D_ri  = C_i sum_d P_dr 
    def dot_1(D_r, P_dr):
        D_r[:] = np.sum(P_dr, axis = 0)
        return True

    # D_ri = C_i sum_d P_dr K_d
    def dot_2(D_r, P_dr, K_d):
        np.dot(K_d, P_dr, out = D_r)
        return True
    
    # D_ri  = C_i sum_d w_d P_dr
    def dot_3(D_r, P_dr, w_d, event):
        event.wait()
        np.dot(w_d, P_dr, out = D_r)
        return True
    
    D_r = np.zeros(R, dtype = float)
    l = config['likelihood']
    f = config['frame_model']
    if l == 'Poisson' and f == 'basic':
        dot_event = executor.submit(dot_1, D_r, P_dr)
        t_r       = np.ones(R, dtype = float)
    
    elif l == 'Poisson_fluence_free' and f == 'basic':
        dot_event = executor.submit(dot_2, D_r, P_dr, K_di.photon_sums)
        t_r       = wsums_r

    elif l == 'Poisson' and f == 'fluence':
        dot_event = executor.submit(dot_3, D_r, P_dr, w_d, w_event)
        t_r       = np.ones_like(wsums_r)

    else :
        err = f'could not parse likelihood "{l}" and frame_model "{f}" combination'
        raise ValueError(err)

    def add_to_I_O(I, O, n_sri, N_ri, D_r, C, t_r, wait = []):
        for event in wait:
            if hasattr(event, 'result'):
                event.result()
            elif hasattr(event, 'wait'):
                event.wait()
        
        N2_ri   = N_ri * t_r[:, None]
        D_ri    = D_r[:, None] * C[None, :]
        for m in n_sri.reshape(-1, N_ri.shape[0], N_ri.shape[1]):
            I += np.bincount(m.ravel(), N2_ri.ravel(), minlength = I.size)
            O += np.bincount(m.ravel(), D_ri.ravel(),  minlength = I.size)

        return True

    executor.shutdown()
    
    M_ri = mapping.Mapping(models_I, **config)

    max_depth = 2 * len(config['queues'])
    
    I_event   = None
    C = config['C']
    
    I_out_c = []
    
    c_iter = tqdm(range(config['models']), desc = 'updating models')
    for c in c_iter :
        Ic = np.zeros(models_I.I[c].size, dtype = float)
        Oc = np.zeros(models_I.I[c].size, dtype = float)
        
        # make buffers for adding independently
        #N_buffers  = 32
        #Ic_buffers = np.zeros((N_buffers, Ic.size), dtype = float)
        #Oc_buffers = np.zeros((N_buffers, Ic.size), dtype = float)
        
        # now subdivide r's into chunks speparated by changes in geometry
        rs = np.where(M_ri.class_r == c)[0]
        rs = tqdm(utils.get_chunks(rs[0], rs[-1]+1, M_ri.changes), desc = 'looping over changes in geom', leave=False)
         
        for r00, r11 in rs:
            r_chunk_size = 256
            r_iter       = tqdm(utils.chunker(r_chunk_size, r11-r00, offset = r00), desc = 'looping over r-chunks', leave = False)
            
            executor = ThreadPoolExecutor()
            PK_events = []
            n_events  = []
            for r0, r1, dr in r_iter:
                # calculate N_ri = sum_d P_dr K_di on the gpu (mainly)
                N_ri     = np.zeros((dr, I), dtype = float)
                PK_events.append(executor.submit(calculate_K_dot_P_gpu, P_dr[:, r0:r1], K_di, M_ri.queues, out = N_ri))
                
                # calculate mapping while dot product is happening
                # n = (symmetry, r, pixel)
                n_cl    = M_ri[r0: r1, :]
                n_sri   = np.empty(n_cl.shape, dtype = n_cl.dtype)
                n_events.append(cl.enqueue_copy(M_ri.queue, n_sri, n_cl.data, is_blocking = False))
                
                if I_event : I_event.result()
                
                # I[n_ri] += t_r N_ri 
                # O[n_ri] += C_i D_r
                I_event = executor.submit(add_to_I_O, Ic, Oc, n_sri, N_ri[:dr], D_r[r0:r1], C, t_r[r0:r1], wait = [PK_events[-1], n_events[-1]])
                PK_events[-1].result()
                
            # we need to do this so that python releases memory
            I_event.result()
            executor.shutdown()
        
        # apply symmetry
        Ic = symmetry.apply_symmetry(Ic.reshape(models_I.I[c].shape), M_ri.symmetry[c], int(models_I.i0))
        Oc = symmetry.apply_symmetry(Oc.reshape(models_I.I[c].shape), M_ri.symmetry[c], int(models_I.i0))
        
        Oc[Oc==0] = 1
        Ic /= Oc
        I_out_c.append(Ic.copy())
    
    return I_out_c, w_d
