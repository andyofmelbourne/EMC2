import pyopencl as cl
import pyopencl.array 
import pyclblast

from emc2 import utils 
from emc2 import utils_cl

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


def calculate_wsums_r(W_ri, r_chunk_size = 1024):
    # calculate tomogram sums
    R = W_ri.shape[0]
    wsums_r = np.zeros(R, dtype = float)

    def sum(wsums_r, W_cl, r0, r1):
        wsums_r[r0:r1] = np.sum(W_cl.get()[:r1-r0], axis=-1)
        
    r_iter = tqdm(utils.chunker(r_chunk_size, R), desc = 'calculating tomogram sums')
    
    executor = ThreadPoolExecutor()

    sum_events = []
    for r0, r1, dr in r_iter:
        # non-blocking
        W_cl = W_ri[r0:r1, :]
         
        # non-blocking
        sum_events.append(executor.submit(sum, wsums_r, W_cl, r0, r1))
         
        if r1 == R :
            executor.shutdown()
    return wsums_r

def calculate_K_dot_W_gpu(W_ri, K_di, r_chunk_size = 1024, d_chunk_size = 1024):
    D, I = K_di.shape
    R, _ = W_ri.shape
    P_dr = np.zeros((D, R), dtype = float)
    W_ri.set_log(True)
    
    d_iter = tqdm(utils.chunker(d_chunk_size, D), desc = 'calculating dot product K . W')
    
    P_dr_cl = [cl.array.empty(queue, (d_chunk_size, r_chunk_size), dtype = np.float32) for queue in W_ri.queues]
    
    def get(P_dr, P_dr_cl, d0, d1, r0, r1):
        P = P_dr_cl.get()
        P_dr[d0:d1, r0:r1] = P[:d1-d0, :r1-r0]
    
    max_depth  = len(W_ri.queues)
    executor   = ThreadPoolExecutor()
    dot_events = []
    get_events = []
    for d0, d1, dd in d_iter:
        K = K_di[d0:d1, :]
        K_cl = [utils_cl.to_gpu(K, queue = queue, dtype = np.float32) for queue in W_ri.queues]

        r_iter = tqdm(utils.chunker(r_chunk_size, R), desc = 'looping over r', leave = False)
        for r0, r1, dr in r_iter:
            # non-blocking
            W_cl = W_ri[r0:r1, :]
            
            # non-blocking
            q = W_ri.active_queue
            dot_events.append(pyclblast.gemm(W_ri.queue, dd, dr, I, K_cl[q], W_cl, P_dr_cl[q], I, I, r_chunk_size, a_transp = False, b_transp = True))
            
            # non-blocking
            get_events.append(executor.submit(get, P_dr, P_dr_cl[q], d0, d1, r0, r1))
            
            # blocking
            if len(dot_events) == max_depth:
                dot_events[0].wait()
                get_events[0].result()
                dot_events.pop(0)
                get_events.pop(0)
                
            if r1 == R:
                executor.shutdown()
                # re-init
                executor   = ThreadPoolExecutor()
                dot_events = []
                get_events = []
        
    W_ri.set_log(False)
    return P_dr

def calculate_K_dot_W(W_ri, K_di, r_chunk_size = 1024, d_chunk_size = 256):
    D, I = K_di.shape
    R, _ = W_ri.shape
    P_dr = np.zeros((D, R), dtype = float)
    W_ri.set_log(True)
    
    def dot(P_dr, K, W_cl, d0, d1, r0, r1):
        W = W_cl.get()
        P_dr[d0:d1, r0:r1] = np.dot(K[:d1-d0], W[:r1-r0].T)
    
    d_iter = tqdm(utils.chunker(d_chunk_size, D), desc = 'calculating dot product K . W')
    r_iter = tqdm(utils.chunker(r_chunk_size, R), desc = 'looping over r', leave = False)
    max_depth = 4
    
    executor   = ThreadPoolExecutor()
    dot_events = []
    
    for d0, d1, dd in d_iter:
        K = K_di[d0:d1, :]
        for r0, r1, dr in r_iter:
            # non-blocking
            W_cl = W_ri[r0:r1, :]
            
            # non-blocking
            dot_events.append(executor.submit(dot, P_dr, K, W_cl, d0, d1, r0, r1))
            
            if len(dot_events) == max_depth or r1 == R:
                executor.shutdown()
                dot_events = []
                executor   = ThreadPoolExecutor()
    
    W_ri.set_log(False)
    return P_dr

def calc_P(K_di, W_ri, cl_stuff, **config):
    D, I = K_di.shape
    R    = W_ri.shape[0]

    cl_cpu_code = cl_stuff['cl_code']
    queue_cpu   = cl_stuff['queue']
    
    # calculate tomogram sums
    wsums_r = calculate_wsums_r(W_ri)
    
    # calculate KW_dr = sum_i K_di W_ri 
    #P_dr1 = calculate_K_dot_W(W_ri, K_di)
    P_dr = calculate_K_dot_W_gpu(W_ri, K_di)
        
    # offset logR (just use cpu)
    # logR_dr = KlogW_dr - K_d log(wsums_r)
    # logR_dr = KlogW_dr - w_d wsums_r
    # logR_dr = KlogW_dr - wsums_r
    
    for i in tqdm(range(1), desc ='applying offset to logR'):
        # logR_dr = \sum_i K_di logW_ri - K_d log(sum_i C_i W_ri)
        # -------------------------------------------------------
        if config['likelihood'] == 'Poisson_fluence_free' and config['frame_model'] == 'basic':
            event = cl_cpu_code.logR_Klog_wsums(
                queue_cpu, 
                (D, R),
                None,
                cl.SVM(P_dr),
                cl.SVM(K_di.photon_sums),
                cl.SVM(wsums_r)
            )

        # logR_dr = \sum_i K_di logW_ri - sum_i C_i W_ri
        # -------------------------------------------------------
        elif config['likelihood'] == 'Poisson' and config['frame_model'] == 'basic':
            event = cl_cpu_code.logR_wsums(
                queue_cpu,
                (D, R),
                None,
                cl.SVM(P_dr),
                cl.SVM(wsums_r)
            )

        # logR_dr = \sum_i K_di logW_ri - w_d sum_i C_i W_ri
        # -------------------------------------------------------
        elif config['likelihood'] == 'Poisson' and config['frame_model'] == 'fluence':
            event = cl_cpu_code.logR_w_wsums(
                queue_cpu,
                (D, R),
                None,
                cl.SVM(P_dr),
                cl.SVM(models_I.w),
                cl.SVM(wsums_r)
            )
    
    event.wait()
    queue_cpu.finish()
    
    # normalise
    rmax_d       = np.zeros(D, dtype = int)
    Pmax_d       = np.zeros(D, dtype = float)
    occupancy_dc = np.zeros((D, config['models']), dtype = float)
    Q_d          = np.zeros(D, dtype = float)
    beta         = np.float64(config['beta'])
    P_thresh     = np.float64(config['P_thresh'])
    class_r      = np.ascontiguousarray(W_ri.class_r.astype(np.int32))
    
    d_chunk_size = 8
    d_iter = tqdm(utils.chunker(d_chunk_size, D), desc = 'calculating P_dr from logR')

    logR_dr = P_dr.copy()

    assert(P_dr.shape == (D, R))
    assert(logR_dr.shape == (D, R))
    assert(class_r.shape == (R,))
    assert(rmax_d.shape == (D,))
    assert(Pmax_d.shape == (D,))
    assert(occupancy_dc.shape == (D, config['models']))
    assert(Q_d.shape == (D,))
    
    for d0, d1, dd in d_iter:
        cl_cpu_code.normalise_P_dr(
            queue_cpu, 
            (dd,),
            None,
            cl.SVM(logR_dr), 
            cl.SVM(P_dr), 
            cl.SVM(class_r),
            cl.SVM(rmax_d),
            cl.SVM(Pmax_d),
            cl.SVM(occupancy_dc),
            cl.SVM(Q_d),
            beta,
            P_thresh,
            np.int32(d0),
            np.int32(config['models']),
            np.int32(R),
            wait_for = [event]
        )
    
    queue_cpu.finish()

    return P_dr, wsums_r, rmax_d, Pmax_d, occupancy_dc, Q_d, P_thresh


code = """
    // optimised for cpu with one worker per d
    __kernel void normalise_P_dr (
        global double *logR_dr, 
        global double *P_dr, 
        global int    *class_r,
        global long   *rmax_d, 
        global double *Pmax_d, 
        global double *occupancy_dc, 
        global double *Q_d, 
        const double beta,
        const double P_thresh,
        const int d_offset,
        const int C,
        const int R
    ) {{
        int d = d_offset + get_global_id(0);
        
        double t, thresh;
        int r, rmax;
        
        double logR_max = -DBL_MAX;
        //double P_dr[{rotations}]; // define R at compile time
        
        // find argmax and max of logR_dr
        for (r=0; r<R; r++) {{
            t = logR_dr[d * R + r];
            //printf("       %e %e       ", t, logR_max);
            if (t > logR_max){{
                rmax = r;
                logR_max = t;
            }}
            P_dr[d * R + r] = t;
        }}
        
        rmax_d[d] = (long)rmax;
         
        // calculate 
        // P_dr = exp( beta * (logR - logRmax))
        for (r=0; r<R; r++) {{
            P_dr[d * R + r] = exp(beta * (P_dr[d * R + r] - logR_max));
            //t = exp(beta * (P_dr[d * R + r] - logR_max));
            //printf("  %d  ", r);
        }}
        
        // threshold 
        if (P_thresh > 0.) {{
            thresh = P_thresh * P_dr[d * R + rmax] ;
            for (r=0; r<R; r++) {{
                if (P_dr[d * R + r] < thresh) 
                    P_dr[d * R + r] = 0.;
            }}
        }}
        
        // normalise \sum_r P_dr to 1
        t = 0.;
        for (r=0; r<R; r++) {{
            t += P_dr[d * R + r];
        }}
        for (r=0; r<R; r++) {{
            P_dr[d * R + r] /= t;
        }}
        
        //printf("            %d %d %d        ", d, R, rmax);
        
        Pmax_d[d] = P_dr[d * R + rmax];
        
        // calculate occupancy_dc and Q
        // Q = \sum_r P_dr logR_dr
        for (r=0; r<R; r++) {{
            occupancy_dc[d * C + class_r[r]] += P_dr[d * R + r];
            Q_d[d] += P_dr[d * R + r] * logR_dr[d * R + r];
        }}
    }}
    
    // logR_dr = KlogW_dr - K_d log(wsums_r)
    // optimised for gpu with one worker per d and r
    __kernel void logR_Klog_wsums (
        global double *KlogW_dr, 
        global long   *K_d,
        global double *wsums_r
    ) {{
        int d = get_global_id(0);
        int r = get_global_id(1);
        int R = get_global_size(1);

        KlogW_dr[d * R + r] -= (double)K_d[d] * log(wsums_r[r]);
    }}

    // logR_dr = KlogW_dr - w_d wsums_r
    __kernel void logR_w_wsums (
        global double *KlogW_dr, 
        global double *w_d,
        global double *wsums_r
    ) {{
        int d = get_global_id(0);
        int r = get_global_id(1);
        int R = get_global_size(1);
        
        KlogW_dr[d * R + r] -= w_d[d] * wsums_r[r];
    }}

    // logR_dr = KlogW_dr - wsums_r
    __kernel void logR_wsums (
        global double *KlogW_dr, 
        global double *wsums_r
    ) {{
        int d = get_global_id(0);
        int r = get_global_id(1);
        int R = get_global_size(1);
        
        KlogW_dr[d * R + r] -= wsums_r[r];
    }}

"""
