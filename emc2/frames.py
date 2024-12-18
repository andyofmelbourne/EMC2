"""
F_dri = w_d C_i W_ri + B_di
"""
import numpy as np

import pyopencl as cl
import pyopencl.array 
import time

from .utils_cl import to_gpu, to_cpu

code = """
// F_dri = w_d C_i W_ri + B_di
// assume no offset for W_ri and B_di
__kernel void calculate_F_dri_v0 (
    global float *F_dri,  
    global float *w_d, 
    global float *C_i, 
    global float *W_ri, 
    global float *B_di, 
    const int rotations,
    const int frames,
    const int w_offset,
    const int C_offset)
{
    int i = get_global_id(0);
    int I = get_global_size(0);
    
    int j;

    float wl, Bl;
    int Cl = C_i[i + C_offset];
    
    for (int d = 0; d < frames; d++) {
        
        wl = w_d[d + w_offset]; 
        Bl = B_di[d * I + i];
        
        for (int r = 0; r < rotations; r++) {
            j = d * rotations * I + r * I + i;
            F_dri[j] = wl * Cl * W_ri[r * I + i] + Bl;
    }}
}


// a little faster than above on intelHD
__kernel void calculate_F_dri_v1 (
    global float *F_dri,  
    global float *w_d, 
    global float *C_i, 
    global float *W_ri, 
    global float *B_di, 
    const int w_offset,
    const int C_offset)
{
    int d = get_global_id(2);
    int r = get_global_id(1);
    int i = get_global_id(0);
    int I = get_global_size(0);
    int rotations = get_global_size(1);
    
    int j;

    j = d * rotations * I + r * I + i;
    F_dri[j] = w_d[d + w_offset] * C_i[i + C_offset] * W_ri[r * I + i] + B_di[d * I + i];
}

// out = K_di log(F_dri) - F_dri
__kernel void calculate_K_logF_dri_v0 (
    global float *out,  
    global uchar *K_di,  
    global float *w_d, 
    global float *C_i, 
    global float *W_ri, 
    global float *B_di, 
    const int w_offset,
    const int C_offset)
{
    int d = get_global_id(2);
    int r = get_global_id(1);
    int i = get_global_id(0);
    int I = get_global_size(0);
    int rotations = get_global_size(1);
    
    float F;
    
    F = w_d[d + w_offset] * C_i[i + C_offset] * W_ri[r * I + i] + B_di[d * I + i];

    int j = d * rotations * I + r * I + i;
        
    if (F>0.)
        out[j] = (float)K_di[d * I + i] * log(F) - F;
    else 
        out[j] = 0.;
}



// gw_d  = P_dr C_i W_ri (K_di / F_dri - 1)
// cw_d = -P_dr K_di (C_i W_ri / F_dri)^2 
// assume no offset for W_ri, B_di, P_dr
__kernel void PCWKF (
    global float *gw,  
    global float *cw,  
    global uchar *K_di,  
    global float *w_d, 
    global float *C_i, 
    global float *W_ri, 
    global float *B_di, 
    global float *P_dr, 
    const int w_offset,
    const int C_offset)
{
    int d = get_global_id(2);
    int r = get_global_id(1);
    int i = get_global_id(0);
    int I = get_global_size(0);
    int rotations = get_global_size(1);
    
    float F, C, W, K, P;

    W = W_ri[r * I + i];
    C = C_i[i + C_offset];
    K = (float)K_di[d * I + i];
    P = P_dr[d * rotations + r];
    
    F = w_d[d + w_offset] * C * W + B_di[d * I + i];

    int j = d * rotations * I + r * I + i;
        
    if (F>0.) {
        gw[j] =  P * C * W * (K / F - 1);
        cw[j] = -P * K * (C * W / F) * (C * W / F);
    } else {
        gw[j] = - P * C * W ;
        cw[j] = 0.;
    }
}

// gW_ri  = \sum_d P_dr w_d C_i (K_di / F_dri - 1)
// cW_d   = -\sum_d  P_dr K_di (C_i w_d / F_dri)^2 
// cwW_ri = \sum_d pw_d P_dr C_i (K_di B_di / F^2_dri - 1)
__kernel void PwCKF (
    global float *gW,  
    global float *cW,  
    global float *cwW,  
    global uchar *K_di,  
    global float *w_d, 
    global float *pw_d, 
    global float *C_i, 
    global float *W_ri, 
    global float *B_di, 
    global float *P_dr, 
    const int w_offset,
    const int C_offset)
{
    int d = get_global_id(2);
    int r = get_global_id(1);
    int i = get_global_id(0);
    int I = get_global_size(0);
    int rotations = get_global_size(1);
    
    float w, F, C, K, P, B;
    
    C = C_i[i + C_offset];
    K = (float)K_di[d * I + i];
    P = P_dr[d * rotations + r];
    w = w_d[d + w_offset];
    B = B_di[d * I + i];
    
    F = w * C * W_ri[r * I + i] + B;
    
    int j = d * rotations * I + r * I + i;
        
    if (F>0.) {
        gW[j]  =  P * w * C * (K / F - 1);
        cW[j]  = -P * K * (C * w / F) * (C * w / F);
        cwW[j] = pw_d[d + w_offset] * P * C * (K * B / (F*F) - 1.);
    } else {
        gW[j]  = - P * w * C ;
        cW[j]  = 0.;
        cwW[j] = -pw_d[d + w_offset] * P * C ;
    }
}
"""

class Frames():
    
    def __init__(self, B_di, w_d, W_ri, **config):
        self.queue   = config['queue']
        self.context = config['context']
        
        self.B_di = B_di
        self.W_ri = W_ri
        self.w_d  = w_d
        self.C    = config['C']
        
        self.frame_indices = B_di.data_getter.frame_indices
        self.pixel_indices = B_di.data_getter.pixel_indices
        self.r_indices     = W_ri.r_indices
        
        self.shape = (B_di.shape[0], W_ri.shape[0], B_di.shape[1])
        self.dtype = np.float32
        self.F_dri    = None
        self.F_cl     = None
        self.F2_dri   = None
        self.F2_cl    = None
        self.F3_dri   = None
        self.F3_cl    = None
        self.K_cl     = None
        self.P_cl     = None
        self.B_cl     = None
        self.W_cl     = None
        self.pw_cl    = None
        self.last_key = None

        self.w_cl     = to_gpu(self.w_d, queue = self.queue, dtype = np.float32)
        self.C_cl     = to_gpu(self.C,   queue = self.queue, dtype = np.float32)
        
        self.cl_code = cl.Program(self.context, code).build()
    
    def parse_key(self, key):
        # only supports slicing
        assert(isinstance(key, tuple))
        assert(len(key) == 3)
        
        d0 = np.int32(self.frame_indices[key[0]][0])
        d1 = np.int32(1+self.frame_indices[key[0]][-1])

        r0 = np.int32(self.r_indices[key[1]][0])
        r1 = np.int32(1+self.r_indices[key[1]][-1])
        
        i0 = np.int32(self.pixel_indices[key[2]][0])
        i1  = np.int32(1+self.pixel_indices[key[2]][-1])
        
        shape = (d1 - d0, r1 - r0, i1 - i0)
        
        # because I use int32 for indexing 
        return shape, d0, d1, r0, r1, i0, i1
    
    def make_buffer(self, shape, key, K_di = None, P_dr = None, F2 = None, F3 = None):
        size = np.prod(shape)
        if self.F_dri is None or self.F_dri.size < np.prod(shape):
            self.F_dri = np.empty(size, dtype = self.dtype)
            self.F_cl  = cl.array.empty(self.queue, size, dtype = self.dtype)
        
        if self.F2_dri is None or self.F2_dri.size < np.prod(shape):
            self.F2_dri = np.empty(size, dtype = self.dtype)
            self.F2_cl  = cl.array.empty(self.queue, size, dtype = self.dtype)
        
        if self.F3_dri is None or self.F3_dri.size < np.prod(shape):
            self.F3_dri = np.empty(size, dtype = self.dtype)
            self.F3_cl  = cl.array.empty(self.queue, size, dtype = self.dtype)

        # see if we need to update B
        if self.B_cl is None or not self.last_key or key[0] != self.last_key[0] or key[2] != self.last_key[2] :
            B_di      = self.B_di[key[0], key[2]]
            self.B_cl = to_gpu(B_di, self.B_cl, queue = self.queue)
        
        # see if we need to update W
        if self.W_cl is None or not self.last_key or key[1] != self.last_key[1] or key[2] != self.last_key[2] :
            self.W_cl     = self.W_ri[key[1], key[2]]
    
        # see if we need to update K
        if K_di is not None and (not self.last_key or key[0] != self.last_key[0] or key[2] != self.last_key[2]) :
            self.K_cl = to_gpu(K_di[:shape[0], :shape[2]], self.K_cl, queue = self.queue, dtype = np.uint8)
        
        # see if we need to update P
        if P_dr is not None and (not self.last_key or key[0] != self.last_key[0] or key[1] != self.last_key[1]) :
            self.P_cl = to_gpu(P_dr[:shape[0], :shape[2]], self.P_cl, queue = self.queue, dtype = np.float32)
        
        self.last_key = key
        
    
    def __getitem__(self, key):
        shape, d0, d1, r0, r1, i0, i1 = self.parse_key(key)
        
        size = np.prod(shape)
        
        self.make_buffer(shape, key)
        
        F_cl = self.calculate_F(d0, d1, r0, r1, i0, i1)
                
        cl.enqueue_copy(self.queue, self.F_dri[:size], self.F_cl.data)
        out = self.F_dri[:size].reshape(shape)
        
        return out
    
    def KlogF_F(self, key, K_di):
        shape, d0, d1, r0, r1, i0, i1 = self.parse_key(key)
        
        size = np.prod(shape)
        
        self.make_buffer(shape, key, K_di)
        
        self.cl_code.calculate_K_logF_dri_v0(self.queue, (i1-i0, r1-r0, d1-d0,), None,
            self.F_cl.data,
            self.K_cl.data,
            self.w_cl.data, 
            self.C_cl.data, 
            self.W_cl.data, 
            self.B_cl.data, 
            d0,
            i0
        )
                
        cl.enqueue_copy(self.queue, self.F_dri[:size], self.F_cl.data)
        self.queue.finish()
        out = self.F_dri[:size].reshape(shape)
        return out

    def PCWKF(self, key, P_dr, K_di):
        shape, d0, d1, r0, r1, i0, i1 = self.parse_key(key)
        
        self.make_buffer(shape, key, K_di, P_dr, F2 = True)
        
        #g, c = self.F_dri.PCWKF(P, K)
        self.cl_code.PCWKF(self.queue, (i1-i0, r1-r0, d1-d0,), None,
            self.F_cl.data,
            self.F2_cl.data,
            self.K_cl.data,
            self.w_cl.data, 
            self.C_cl.data, 
            self.W_cl.data, 
            self.B_cl.data, 
            self.P_cl.data, 
            d0,
            i0
        )
                
        size = np.prod(shape)
        cl.enqueue_copy(self.queue, self.F_dri[:size], self.F_cl.data)
        cl.enqueue_copy(self.queue, self.F2_dri[:size], self.F2_cl.data)
        self.queue.finish()
        gw = self.F_dri[:size].reshape(shape)
        cw = self.F2_dri[:size].reshape(shape)
        return gw, cw

    def PwCKF(self, key, pw, P_dr, K_di):
        shape, d0, d1, r0, r1, i0, i1 = self.parse_key(key)
        
        self.make_buffer(shape, key, K_di, P_dr, F2 = True, F3 = True)
        
        if self.pw_cl is None :
            self.pw_cl = to_gpu(pw, self.pw_cl, queue = self.queue, dtype = np.float32)
        
        #g, c = self.F_dri.PCWKF(P, K)
        self.cl_code.PwCKF(self.queue, (i1-i0, r1-r0, d1-d0,), None,
            self.F_cl.data,
            self.F2_cl.data,
            self.F3_cl.data,
            self.K_cl.data,
            self.w_cl.data, 
            self.pw_cl.data, 
            self.C_cl.data, 
            self.W_cl.data, 
            self.B_cl.data, 
            self.P_cl.data, 
            d0,
            i0
        )
                
        size = np.prod(shape)
        cl.enqueue_copy(self.queue, self.F_dri[:size], self.F_cl.data)
        cl.enqueue_copy(self.queue, self.F2_dri[:size], self.F2_cl.data)
        cl.enqueue_copy(self.queue, self.F3_dri[:size], self.F3_cl.data)
        self.queue.finish()
        gW  = self.F_dri[:size].reshape(shape)
        cW  = self.F2_dri[:size].reshape(shape)
        cwW = self.F3_dri[:size].reshape(shape)

        return gW, cW, cwW
        
            
    def calculate_F(self, d0, d1, r0, r1, i0, i1):
        #t0 = time.time()
        self.cl_code.calculate_F_dri_v1(self.queue, (i1-i0, r1-r0, d1-d0,), None,
            self.F_cl.data,
            self.w_cl.data, 
            self.C_cl.data, 
            self.W_cl.data, 
            self.B_cl.data, 
            d0,
            i0
        )
        #self.queue.finish()
        #print(f'time for cl code: {time.time() - t0}')
