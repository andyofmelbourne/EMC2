"""
There are two types of tomograms to make: 
    - ones that depend on the frame number
        - per pattern geometry
        - per pattern wavelength 
    
    - ones that do not depend on frame number

In both cases, we might have 2D or 3D mappings

Here we will take care of the orientations 

r -> class, rotation
W_ri = I[class, R . q]

"""
import numpy as np

import pyopencl as cl
import pyopencl.array 

from . import orientations 
from .utils_cl import to_gpu

def get_rotation_matrices(queue = None, context = None, rotation_order = 10, dimensions = 3, **kwargs):
    if dimensions == 3 :
        R_cl = orientations.get_rotations_3D(rotation_order, queue, context)
    
    elif dimensions == 2 :
        R_cl = orientations.get_rotations_2D(rotation_order, queue, context)
    
    else :
        raise ValueError(f'dimension {dimension} not supported')
    return R_cl


code = """
        constant sampler_t interpolation = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_{interpolation} ;
        
        // W_ri = I[class, R_r . [qx_i, qy_i]]
        __kernel void calculate_tomograms_2D_v0 (
            global float *Wout,  
            __read_only image2d_array_t I, 
            global float *R, 
            global float *qx, 
            global float *qy, 
            const float i0,
            const float dq,
            const int r_offset,
            const int i_offset,
            const int rotations)
        {{
            int r        = r_offset + get_global_id(0);
            int rotation = r % rotations;
            int class    = r / rotations;
            int pixel    = i_offset + get_global_id(1);
             
            int chunk_size_i = get_global_size(1);
            
            float R_l[4];
            
            int i;
            for (i=0; i<4; i++) {{
                R_l[i] = R[4*rotation + i];
            }}
            
            float4 coord ;
            float4 W;
            
            coord.x = i0 + (R_l[2] * qx[pixel] + R_l[3] * qy[pixel]) / dq + 0.5;
            coord.y = i0 + (R_l[0] * qx[pixel] + R_l[1] * qy[pixel]) / dq + 0.5;
            coord.z = class ;
                
            W = read_imagef(I, interpolation, coord);
            
            Wout[r * chunk_size_i + pixel - i_offset] = {Wmap};
        }}
        """

class Tomograms():
    def __init__(self, models_I, **config):
        queue = config['queue']
        
        # rotation matrices
        self.R_cl = get_rotation_matrices(**config)
        
        self.per_pattern = False
        
        self.models = models_I
        
        self.pixels = config['pixels']
        
        self.qx_cl = to_gpu(config['q'][0], queue = queue)
        self.qy_cl = to_gpu(config['q'][1], queue = queue)
        
        self.rotations = np.int32(self.R_cl.shape[0])
        self.classes   = config['models']
        
        self.pixel_indices    = np.arange(self.pixels)
        self.r_indices        = np.arange(config['models'] * self.rotations) 
        self.class_indices    = self.r_indices // self.rotations
        self.rotation_indices = self.r_indices % self.rotations
        
        #self.rotation_indices_cl = to_gpu(self.rotation_indices, queue = queue)
        #self.class_indices_cl    = to_gpu(self.class_indices,    queue = queue)
        
        self.shape = self.rotation_indices.shape + self.pixel_indices.shape
        self.dtype = np.float32
        self.dimensions = config['dimensions']
        
        self.W_cl = None

        self.queue   = queue 
        self.context = config['context'] 
        
        if config['interpolation_forward'] == 'linear':
            self.interpolation = 'LINEAR'
        elif config['interpolation_forward'] == 'nearest':
            self.interpolation = 'NEAREST'
        else :
            raise ValueError(f'forward interpolation strategy {interpolation_forward} not supported')
        
        if self.dimensions == 2 :
            self.compile_2D()
        else :
            raise ValueError(f'dimension = {self.dimensions} not supported')
    
    def set_log(self, log = True):
        if log :
            self.cl_code = self.cl_code_log
        else :
            self.cl_code = self.cl_code_normal
    
    def compile_2D(self):
        # compile code twice
        # once for computing W_ir 
        # once for computing log(W_ir)
        self.cl_code_normal = cl.Program(self.context, code.format(interpolation = self.interpolation, Wmap = 'W.x')).build()
        self.cl_code_log    = cl.Program(self.context, code.format(interpolation = self.interpolation, Wmap = 'log(W.x)')).build()
        
        self.cl_code = self.cl_code_normal
        
    def __getitem__(self, key):
        """
        put pixels in the first dimension for efficient summing
        W_ri 
        only suports slicing:
            self[10:20, 100:20]
        """
        assert(isinstance(key, tuple))
        assert(len(key) == 2)
         
        r_start = np.int32(self.r_indices[key[0]][0])
        r_stop  = np.int32(1+self.r_indices[key[0]][-1])
        
        pixel_start = np.int32(self.pixel_indices[key[1]][0])
        pixel_stop  = np.int32(1+self.pixel_indices[key[1]][-1])
        
        shape = (r_stop - r_start, pixel_stop - pixel_start)
        if self.W_cl is None or \
                self.W_cl.shape[0] < shape[0] or \
                self.W_cl.shape[1] < shape[1]:
            self.W_cl = cl.array.empty(self.queue, shape, dtype = np.float32)
        
        if self.dimensions == 2 :
            self.W_cl = self.calculate_tomograms_2D(r_start, r_stop, pixel_start, pixel_stop)
        return self.W_cl
            
    def calculate_tomograms_2D(self, r0, r1, i0, i1):
        """
        """
        self.cl_code.calculate_tomograms_2D_v0(self.queue, (r1-r0, i1-i0), None,
                self.W_cl.data,
                self.models.I_cl, 
                self.R_cl.data, 
                self.qx_cl.data, 
                self.qy_cl.data, 
                self.models.i0, 
                self.models.dq,
                np.int32(r0),
                np.int32(i0),
                self.rotations)
        return self.W_cl
                
