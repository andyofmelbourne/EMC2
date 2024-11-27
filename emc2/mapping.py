import numpy as np

from .tomograms import Tomograms

import pyopencl as cl
import pyopencl.array 

code_3D = """
        __kernel void mapping_nearest_3D_v0 (
            global int   *out_n,  
            global float *R, 
            global float *qx, 
            global float *qy, 
            global float *qz, 
            const int M,
            const float i0,
            const float dq,
            const int rotation_offset,
            const int pixel_offset,
            const int W_offset)
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
            
            coord.x = (R_l[0] * qxl + R_l[1] * qyl + R_l[2] * qzl) / dq;
            coord.y = (R_l[3] * qxl + R_l[4] * qyl + R_l[5] * qzl) / dq;
            coord.z = (R_l[6] * qxl + R_l[7] * qyl + R_l[8] * qzl) / dq;
            
            // apply symmetry operations here
            {symmetry}
            //j = W_offset + r * chunk_size_i + i;
            //out_n[j] = convert_int_rte(i0 + coord.x) * M * M + convert_int_rte(i0 + coord.y) * M + convert_int_rte(i0 + coord.z);
        }}
"""
        
code_2D = """
        __kernel void mapping_nearest_static_v0 (
            global int *out_n,  
            global float *qx, 
            global float *qy, 
            const int M,
            const float i0,
            const float dq,
            const int pixel_offset,
            const int W_offset)
        {{
            int i        = get_global_id(0);
            int pixel    = pixel_offset + i;
             
            int chunk_size_i = get_global_size(1);
            
            float2 coord ;
            float4 W;
            
            coord.x = qx[pixel] / dq;
            coord.y = qy[pixel] / dq;
            
            int j;
            int r = 0;
            {symmetry}   
            //int j = W_offset + i;
            //out_n[j] = convert_int_rte(i0 + coord.x) * M + convert_int_rte(i0 + coord.y);
        }}

        __kernel void mapping_nearest_2D_v0 (
            global int *out_n,  
            global float *R, 
            global float *qx, 
            global float *qy, 
            const int M,
            const float i0,
            const float dq,
            const int rotation_offset,
            const int pixel_offset,
            const int W_offset)
        {{
            int r        = get_global_id(0);
            int i        = get_global_id(1);
            int rotation = rotation_offset + r;
            int pixel    = pixel_offset    + i;
             
            int chunk_size_i = get_global_size(1);
            
            float R_l[4];
            
            int j;
            for (j=0; j<4; j++) {{
                R_l[j] = R[4*rotation + j];
            }}
            
            float2 coord ;
            
            coord.x = (R_l[2] * qx[pixel] + R_l[3] * qy[pixel]) / dq;
            coord.y = (R_l[0] * qx[pixel] + R_l[1] * qy[pixel]) / dq;
                
            {symmetry}   
            //j = W_offset + r * chunk_size_i + i;
            //out_n[j] = convert_int_rte(i0 + coord.x) * M + convert_int_rte(i0 + coord.y);
        }}
"""

class Mapping(Tomograms):
    """
    return an array like M_ri

    this yields the raveled pixel coordinates 
    for the class corresponding to the r-index

    when symmetry is applied there are repeated 
    entries at a given 'r' for each symmetry
    
    Thus a second array is provided to index the relevant 
    r-index for each n-index returned

    n_li, r_l = M_ri[:100, :]

    I[class[r_l[l]], n_li[l, i]] approx. W_ri[r_l[0], i]
    """

    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)
            
        # change from tomograms
        # to flattened model indices
        self.dtype = np.int32
        
        self.M = np.int32(kwargs['model_length'])
        # only applies for 3D models but better than
        # nothing for now
        assert(self.M**3 < np.iinfo(np.int32).max)
        
        # get symmetry order 
        # number of symmetry opperations to apply to each frame
        self.order = []
        for d, symmetry in zip(self.dimensions, self.symmetry) :
            if symmetry == 'P1': 
                self.order.append(1)
            
            elif symmetry == 'inversion':
                self.order.append(2)
            
            elif symmetry == 'D6' and d == 2:
                self.order.append(12)
            
            elif symmetry == 'D6' and d == 3:
                self.order.append(24)
            
            else :
                err = f'symmetry "{symmetry}" not supported'
                raise ValueError(err)
        
        # compile code for each symmetry opperator
        self.code = {}
        for d, symmetry in zip(self.dimensions, self.symmetry):
            if d == 2 :
                if symmetry == 'P1':
                    s = """
                    j = r * chunk_size_i + i;
                    out_n[j] = convert_int_rte(i0 + coord.x) * M + convert_int_rte(i0 + coord.y);
                    """
                
                elif symmetry == 'inversion' :
                    s = """
                    j = r * chunk_size_i + i;
                    out_n[j] = convert_int_rte(i0 + coord.x) * M + convert_int_rte(i0 + coord.y);
                    
                    j = W_offset + r * chunk_size_i + i;
                    out_n[j] = convert_int_rte(i0 - coord.x) * M + convert_int_rte(i0 - coord.y);
                    """
                
                # just to see what it looks like in 2D
                elif symmetry == 'D6' :
                    # add inversion symmetry
                    s = """
                    float c = 0.5;
                    float s = 0.8660254037844386;
                    float x, y;
                     
                    j = r * chunk_size_i + i - W_offset;
                    
                    // 6 x pi / 3 rotations about z-axis
                    for (i=0; i<6; i++) {
                        x = coord.x ;
                        y = coord.y ;
                        coord.x = x * c - y * s;
                        coord.y = x * s + y * c;
                        
                        j += W_offset;
                        out_n[j] = convert_int_rte(i0 + coord.x) * M  + convert_int_rte(i0 + coord.y);
                            
                        j += W_offset;
                        out_n[j] = convert_int_rte(i0 - coord.x) * M  + convert_int_rte(i0 - coord.y);
                    }
                    """
                
                self.code[(d, symmetry)] = cl.Program(
                    self.context, 
                    code_2D.format(symmetry = s)
                    ).build()
            
            if d == 3 :
                if symmetry == 'P1':
                    s = """
                    j = r * chunk_size_i + i;
                    out_n[j] = convert_int_rte(i0 + coord.x) * M * M + convert_int_rte(i0 + coord.y) * M + convert_int_rte(i0 + coord.z);
                    """
                
                elif symmetry == 'inversion' :
                    s = """
                    j = r * chunk_size_i + i;
                    out_n[j] = convert_int_rte(i0 + coord.x) * M * M + convert_int_rte(i0 + coord.y) * M + convert_int_rte(i0 + coord.z);
                    
                    j = W_offset + r * chunk_size_i + i;
                    out_n[j] = convert_int_rte(i0 - coord.x) * M * M + convert_int_rte(i0 - coord.y) * M + convert_int_rte(i0 - coord.z);
                    """
                
                elif symmetry == 'D6' :
                    # add inversion symmetry
                    s = """
                    float c = 0.5;
                    float s = 0.8660254037844386;
                    float x, y;
                     
                    j = r * chunk_size_i + i - W_offset;
                    
                    // 2 x pi rotations about x-axis
                    for (int k=0; k<2; k++) {
                        y       = -coord.y ;
                        coord.z = -coord.z ;
                    
                    // 6 x pi / 3 rotations about z-axis
                    for (i=0; i<6; i++) {
                        x = coord.x ;
                        y = coord.y ;
                        coord.x = x * c - y * s;
                        coord.y = x * s + y * c;
                        
                        j += W_offset;
                        out_n[j] = convert_int_rte(i0 + coord.x) * M * M + convert_int_rte(i0 + coord.y) * M + convert_int_rte(i0 + coord.z);
                            
                        j += W_offset;
                        out_n[j] = convert_int_rte(i0 - coord.x) * M * M + convert_int_rte(i0 - coord.y) * M + convert_int_rte(i0 - coord.z);
                    }}
                    """
                
                self.code[(d, symmetry)] = cl.Program(
                    self.context, 
                    code_3D.format(symmetry = s)
                    ).build()
        
        self.n_cl = None
        self.n    = None
                
    
    def make_buffer(self, shape, c):
        if self.n_cl is None or \
                self.n_cl.shape[0] < shape[0] or \
                self.n_cl.shape[1] != shape[1]:
            print('allocating gpu array:', shape, shape[0]*shape[1]*4/1024**2, 'mb')
            self.n_cl = cl.array.empty(self.queue, shape, dtype = self.dtype)
            self.n    = np.empty(shape, dtype = self.dtype)
    
    def __getitem__(self, key):
        """
        put pixels in the first dimension for efficient summing
        W_ri 
        only suports slicing:
            self[10:20, 100:20]
        """
        r_start, r_stop, pixel_start, pixel_stop, shape = self.parse_key(key)
        
        # only allow calling over r-range with a single class
        # and no other changes (such as q)
        c = self.class_r[r_start]
        #print(self.class_r[r_start: r_stop-1])
        #print(self.q_r[r_start: r_stop-1])
        assert(np.all(self.change_state[r_start: r_stop-1] == 0))
        
        # add symmetry to shape
        shape = list(shape)
        shape[0] = self.order[c] * shape[0]
        shape = tuple(shape)
        
        self.make_buffer(shape, c)
        
        self.n_cl = self.calculate_mapping(c, r_start, r_stop, pixel_start, pixel_stop)
        cl.enqueue_copy(self.queue, self.n[:shape[0]], self.n_cl.data)
        return self.n[:shape[0]].reshape(self.order[c], r_stop-r_start, pixel_stop-pixel_start)
    
    # replace calculate tomograms
    def calculate_mapping(self, c, r0, r1, i0, i1):
        """
        we should evaluate in chunks or r
        such that the q-values and classes do not change
        """
        # no changes are allowed in r range
        c  = self.class_r[r0]
        q  = self.q_r[r0]
        d  = self.dimensions[c]
        ro = self.rotation_orders[c]
        dr = r1-r0 
        W_offset           = np.int32(dr * (i1-i0))
        orientation_offset = np.int32(self.orientation_r[r0])

        if d == 2 and ro == 0 :
            code = self.code[d, self.symmetry[c]]
            
            code.mapping_nearest_static_v0(self.queue, (i1-i0,), None,
                    self.n_cl.data,
                    self.qxy[q][0].data, 
                    self.qxy[q][1].data, 
                    self.M,
                    self.i0, 
                    self.dq,
                    np.int32(i0),
                    W_offset)
        
        elif d == 2 and ro > 0 :
            code = self.code[d, self.symmetry[c]]
            
            code.mapping_nearest_2D_v0(self.queue, (r1-r0, i1-i0), None,
                    self.n_cl.data,
                    self.rotation_matrices[(d, ro)].data,
                    self.qxy[q][0].data, 
                    self.qxy[q][1].data, 
                    self.M,
                    self.i0, 
                    self.dq,
                    orientation_offset,
                    np.int32(i0),
                    W_offset)
         
        elif d == 3 and ro > 0 :
            code = self.code[d, self.symmetry[c]]
            
            code.mapping_nearest_3D_v0(self.queue, (r1-r0, i1-i0), None,
                    self.n_cl.data,
                    self.rotation_matrices[(d, ro)].data,
                    self.qxy[q][0].data, 
                    self.qxy[q][1].data, 
                    self.qxy[q][2].data, 
                    self.M,
                    self.i0, 
                    self.dq,
                    orientation_offset,
                    np.int32(i0),
                    W_offset)
        
        return self.n_cl

