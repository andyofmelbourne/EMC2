"""
F_dri = w_d C_i W_ri + B_di
"""
import numpy as np

import pyopencl as cl
import pyopencl.array 

code = """
// F_dri = w_d C_i W_ri + B_di
__kernel void calculate_F_dri_v0 (
    global float *F_dri,  
    global float *w_d, 
    global float *C_i, 
    global float *W_ri, 
    global float *B_di, 
    const int rotation_offset,
    const int pixel_offset,
    const int frame_offset)
{{
    int r        = get_global_id(0);
    int i        = get_global_id(1);
    int rotation = rotation_offset + r;
    int pixel    = pixel_offset    + i;
     
    int chunk_size_i = get_global_size(1);
    
    float R_l[9];
    
    int j;
    for (j=0; j<9; j++) {{
        R_l[j] = R[9*rotation + j];
    }}
    
    float4 coord ;
    float4 W;
    
    float qxl = qx[pixel];
    float qyl = qy[pixel];
    float qzl = qz[pixel];
    
    coord.x = i0 + (R_l[0] * qxl + R_l[1] * qyl + R_l[2] * qzl) / dq;
    coord.y = i0 + (R_l[3] * qxl + R_l[4] * qyl + R_l[5] * qzl) / dq;
    coord.z = i0 + (R_l[6] * qxl + R_l[7] * qyl + R_l[8] * qzl) / dq;
    
    j = W_offset + r * chunk_size_i + i;
    
    // get flattened I index
    out_n[j] = convert_int_rte(coord.x) * M * M + convert_int_rte(coord.y) * M + convert_int_rte(coord.z);
}}
"""

class Frames():
    
    def __init__(self, B_di, w_d, W_ri, **config):
        self.B_di = B_di
        self.W_ri = W_ri
        self.w_d  = w_d
        self.C    = config['C']

        self.frame_indices = B_di.data_getter.frame_indices
        self.pixel_indices = B_di.data_getter.pixel_indices
        self.r_indices     = W_ri.r_indices

        self.dtype = np.float32
        self.F_dri = None
    
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
        assert((shape[0] * shape[1] * shape[2]) < (2**31-1))
        return shape

    def make_buffer(self, shape):
        if self.F_dri is None or \
                self.F_dri.shape[0] < shape[0] or \
                self.F_dri.shape[1] < shape[1] or \
                self.F_dri.shape[2] < shape[2] :
            self.F_dri = np.empty(shape, dtype = self.dtype)
    
    def __getitem__(self, key):
        shape = self.parse_key(key)
        
        self.make_buffer(shape)
                
        dd, dr, di = shape
        self.F_dri[:dd, :dr, :di]  = self.w_d[key[0], None, None] * self.C[None, None, key[2]] * self.W_ri[key[1:]] 
        self.F_dri[:dd, :dr, :di] += self.B_di[(key[0],key[2])][:, None, :]
        
        return self.F_dri
