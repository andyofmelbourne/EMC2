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
import sys

import pyopencl as cl
import pyopencl.array 

from . import orientations 
from .utils_cl import to_gpu
from . import utils
from .model import load_models_cl

def get_rotation_matrices(queue = None, context = None, rotation_order = 10, dimensions = 3, **kwargs):
    if dimensions == 3 and rotation_order > 0 :
        R_cl = orientations.get_rotations_3D(rotation_order, queue, context)
    
    elif dimensions == 2 and rotation_order > 0 :
        R_cl = orientations.get_rotations_2D(rotation_order, queue, context)
    
    elif dimensions == 2 and rotation_order == 0 :
        R_cl = None
    
    else :
        raise ValueError(f'could not reconsile dimension {dimensions} and rotation_order {rotation_order}')
    
    return R_cl


code = """
        constant sampler_t interpolation = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_{interpolation} ;
        
        __kernel void mapping_3D_v0 (
            global int *out_n,  
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
            
            coord.x = i0 + (R_l[0] * qxl + R_l[1] * qyl + R_l[2] * qzl) / dq;
            coord.y = i0 + (R_l[3] * qxl + R_l[4] * qyl + R_l[5] * qzl) / dq;
            coord.z = i0 + (R_l[6] * qxl + R_l[7] * qyl + R_l[8] * qzl) / dq;
            
            j = W_offset + r * chunk_size_i + i;
            
            // get flattened I index
            out_n[j] = convert_int_rte(coord.x) * M * M + convert_int_rte(coord.y) * M + convert_int_rte(coord.z);
        }}
        
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
            
            coord.x = i0 + qx[pixel] / dq;
            coord.y = i0 + qy[pixel] / dq;
                
            int j = W_offset + i;
             
            // get flattened I index
            out_n[j] = convert_int_rte(coord.x) * M + convert_int_rte(coord.y);
        }}

        __kernel void mapping_nearest_2D_v0 (
            global int *out_n,  
            global float *R, 
            global float *qx, 
            global float *qy, 
            const int M, // side length of model
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
            
            coord.x = i0 + (R_l[2] * qx[pixel] + R_l[3] * qy[pixel]) / dq;
            coord.y = i0 + (R_l[0] * qx[pixel] + R_l[1] * qy[pixel]) / dq;
                
            j = W_offset + r * chunk_size_i + i;
        
            // get flattened I index
            out_n[j] = convert_int_rte(coord.x) * M + convert_int_rte(coord.y);
        }}
        
        
        // W_ri = I[class, R_r . [qx_i, qy_i]]
        __kernel void calculate_tomograms_2D_v0 (
            global float *Wout,  
            __read_only image2d_t I, 
            global float *R, 
            global float *qx, 
            global float *qy, 
            const float i0,
            const float dq,
            global int *ro,
            const int pixel_offset,
            const int W_offset)
        {{
            int r        = get_global_id(0);
            int i        = get_global_id(1);
            int rotation = ro[r];
            int pixel    = pixel_offset    + i;
             
            int chunk_size_i = get_global_size(1);
            
            float R_l[4];
            
            int j;
            for (j=0; j<4; j++) {{
                R_l[j] = R[4*rotation + j];
            }}
            
            float2 coord ;
            float4 W;
            
            coord.x = i0 + (R_l[2] * qx[pixel] + R_l[3] * qy[pixel]) / dq + 0.5;
            coord.y = i0 + (R_l[0] * qx[pixel] + R_l[1] * qy[pixel]) / dq + 0.5;
                
            W = read_imagef(I, interpolation, coord);
            
            j = W_offset + r * chunk_size_i + i;
            
            // Wout[j] = W.x;
            {Wmap}
        }}

        __kernel void calculate_tomograms_static_v0 (
            global float *Wout,  
            __read_only image2d_t I, 
            global float *qx, 
            global float *qy, 
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
            
            coord.x = i0 + qx[pixel] / dq + 0.5;
            coord.y = i0 + qy[pixel] / dq + 0.5;
                
            W = read_imagef(I, interpolation, coord);
            
            int j = W_offset + i;
            
            // Wout[j] = W.x;
            {Wmap}
        }}

        __kernel void calculate_tomograms_3D_v0 (
            global float *Wout,  
            __read_only image3d_t I, 
            global float *R, 
            global float *qx, 
            global float *qy, 
            global float *qz, 
            const float i0,
            const float dq,
            global int *ro,
            const int pixel_offset,
            const int W_offset)
        {{
            int r        = get_global_id(0);
            int i        = get_global_id(1);
            int rotation = ro[r];
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
            
            coord.x = i0 + (R_l[0] * qxl + R_l[1] * qyl + R_l[2] * qzl) / dq + 0.5;
            coord.y = i0 + (R_l[3] * qxl + R_l[4] * qyl + R_l[5] * qzl) / dq + 0.5;
            coord.z = i0 + (R_l[6] * qxl + R_l[7] * qyl + R_l[8] * qzl) / dq + 0.5;
                
            W = read_imagef(I, interpolation, coord);
            
            j = W_offset + r * chunk_size_i + i;
            
            // Wout[j] = W.x;
            {Wmap}
        }}
        """

class Tomograms_log():
    def __init__(self, tomo):
        self.tomo = tomo
    
    def __getitem__(self, key):
        self.tomo.set_log(True)
        out = self.tomo.__getitem__(key)
        self.tomo.set_log(False)
        return out

    
class Tomograms():
    """
    We want to allow for tomograms with different:
        - q mappings (pointing or wavelength fluctuations)
        - dimensions
        - rotation orders

    Rotations = {
        (dimension, rotation_order): Rs_cl
    }
    for each unique combination of (dimension, rotation_order)
    """
    def __init__(self, models_I, cpu = False, **config):
        queues  = config['queues']
        context = config['context']
        self.config = config
        
        # rotation matrices
        # -----------------
        self.dimensions      = utils.int_to_list(config['models'], config['dimensions']    , 'dimensions')
        self.rotation_orders = utils.int_to_list(config['models'], config['rotation_order'], 'rotation_order')
        self.symmetry        = utils.int_to_list(config['models'], config['symmetry']      , 'symmetry')
        
        self.symmetry_unique, self.symmetry_index = np.unique(self.symmetry, return_inverse = True)
        
        dr = set(zip(self.dimensions, self.rotation_orders))
        
        self.rotation_matrices_q = len(queues) * [{}]
        for q, queue in enumerate(queues):
            for (d, r) in set(dr):
                self.rotation_matrices_q[q][(d, r)] = get_rotation_matrices(
                    queue   = queue, 
                    context = context, 
                    rotation_order = r,
                    dimensions     = d
                )
        
        
        ##########
        self.rotations = 1 # hack
        self.pixel_indices  = np.arange(config['pixels'])
        self.update_mask(np.ones(config['pixels'], dtype = bool), config)
        ##########

        # ------------------------------------------------------
        # r is scalar integer that indexes: q, class, orientation
        #   - the number of r values is called "rotations"
        #   - the q           for each r value is called "q_r"
        #   - the class       for each r value is called "class_r"
        #   - the orientation for each r value is called "orientation_r"
        #   - the symmetry index for each r value is called "symmetry_r"
        # ------------------------------------------------------
        r = 0
        qrs = []
        crs = []
        ors = []
        srs = []
            
        for c in range(config['models']) :
            for q in range(len(self.xy_offset)) :
                R = self.rotation_matrices_q[0][(self.dimensions[c], self.rotation_orders[c])] 
                
                if R is None :
                    r = 1
                else :
                    r = R.shape[0]
                
                s = self.symmetry_index[c]
                
                srs += r * [s]
                qrs += r * [q]
                crs += r * [c]
                ors += range(r)
        
        self.q_r           = np.array(qrs)
        self.class_r       = np.array(crs)
        self.orientation_r = np.array(ors)
        self.symmetry_r    = np.array(srs)
         
        self.rotations = np.int32(len(self.q_r))
        self.classes   = config['models']
        
        # find the start and stop values within which q class don't change
        # these are the indices of r where class or q has a new value
        self.change_state = np.diff(self.q_r) + np.diff(self.class_r)
        self.changes = 1 + np.where(self.change_state)[0]
        
        self.r_indices     = np.arange(self.rotations)
        
        self.shape = (self.rotations, self.pixels)
        
        self.dtype = np.float32
        
        self.W_cl_q = len(queues) * [None]
        
        self.queues  = queues
        self.context = config['context'] 
        self.active_queue = 0
        
        if config['interpolation_forward'] == 'linear':
            self.interpolation = 'LINEAR'
        elif config['interpolation_forward'] == 'nearest':
            self.interpolation = 'NEAREST'
        else :
            raise ValueError(f'forward interpolation strategy {interpolation_forward} not supported')

        self.log = Tomograms_log(self)
        self.cpu = cpu

        # load models
        # -----------
        self.load_models(models_I)
        
        self.compile()

    def load_models(self, models_I):
        self.dq     = models_I.dq
        self.i0     = models_I.i0
        self.models_q = []
        for queue in self.queues :
            self.models_q.append(load_models_cl(models_I.dimensions, models_I.I, queue, self.context))
    
    def set_log(self, log = True):
        if log :
            self.cl_code = self.cl_code_log
        else :
            self.cl_code = self.cl_code_normal
    
    def update_mask(self, pixels, config):
        queues = config['queues']
        
        self.pixels         = len(pixels)
        self.pixel_indices0 = pixels.copy()
        self.shape = (self.rotations, self.pixels)
        
        # per pixel q values
        # ------------------
        # this links the lists to each other! 
        # https://stackoverflow.com/questions/12791501/why-does-this-code-for-initializing-a-list-of-lists-apparently-link-the-lists-to
        #self.qxy_q     = len(queues) * [[]]
        
        self.qxy_q     = [[] for i in range(len(queues))]
        self.xy_offset = []
        xyz = config['xyz']
        if config['pointing_fluctuations'] :
            N, step = config['pointing_fluctuations'] 
            for n in (np.arange(N) - (N//2)):
                for m in (np.arange(N) - (N//2)):
                    xyz2 = xyz.copy()
                    xyz2[0] += step * n
                    xyz2[1] += step * m
                    q = utils.calc_q(config['wavelength'], xyz2)
                    
                    self.xy_offset.append( [n * step, m * step] ) 
                    for qi, queue in enumerate(queues):
                        self.qxy_q[qi].append((
                            to_gpu(q[0][pixels], queue = queue),
                            to_gpu(q[1][pixels], queue = queue),
                            to_gpu(q[2][pixels], queue = queue),
                        ))
        else :
            self.xy_offset.append( [0, 0] ) 
            for qi, queue in enumerate(queues):
                self.qxy_q[qi].append((
                    to_gpu(config['q'][0][pixels], queue = queue),
                    to_gpu(config['q'][1][pixels], queue = queue),
                    to_gpu(config['q'][2][pixels], queue = queue),
                ))
    
    def compile(self):
        # compile code twice
        # once for computing W_ir 
        # once for computing log(W_ir)
        wmap     = 'Wout[j] = W.x;'
        wmap_log = """
        if (W.x > 0.) 
            Wout[j] = log(W.x);
        else 
            Wout[j] = 0.;
        """
        self.cl_code_normal = cl.Program(self.context, code.format(interpolation = self.interpolation, Wmap = wmap)).build()
        self.cl_code_log    = cl.Program(self.context, code.format(interpolation = self.interpolation, Wmap = wmap_log)).build()
        
        self.cl_code = self.cl_code_normal

    def parse_key(self, key):
        assert(isinstance(key, tuple))
        assert(len(key) == 2)
         
        rs     = self.r_indices[key[0]]
        pixels = self.pixel_indices[key[1]]
        
        if pixels.shape != self.pixel_indices0.shape or not np.allclose(pixels, self.pixel_indices0):
            self.update_mask(pixels, self.config)
        
        shape = (len(rs), len(pixels))
        
        # because I use int32 for indexing 
        assert((shape[0] * shape[1]) < (2**31-1))
        
        return rs, pixels, shape

    def make_buffer(self, shape):
        # we need a new cl buffer if the pixels change
        # or the number of r's increases
        if self.W_cl is None or self.W_cl.shape[0] < shape[0] or self.W_cl.shape[1] != shape[1]:
            
            self.W_cl = cl.array.empty(self.queue, shape, dtype = self.dtype)
            
            if self.cpu :
                self.W = np.empty(shape, dtype = self.dtype)
    
    def __getitem__(self, key):
        """
        put pixels in the first dimension for efficient summing
        W_ri 
        only suports slicing:
            self[10:20, 100:20]
        """
        rs, pixels, shape = self.parse_key(key)

        # set active queue 
        self.active_queue      = (self.active_queue + 1) % len(self.queues)
        self.queue             = self.queues[self.active_queue]
        self.W_cl              = self.W_cl_q[self.active_queue]
        self.qxy               = self.qxy_q[self.active_queue]
        self.models            = self.models_q[self.active_queue]
        self.rotation_matrices = self.rotation_matrices_q[self.active_queue]
        
        self.make_buffer(shape)
        
        # making a new buffer each time seems to help with threading
        #self.W_cl_q[self.active_queue] = self.W_cl
        
        self.W_cl, self.event = self.calculate_tomograms(rs, pixels)
        
        if self.cpu :
            cl.enqueue_copy(self.queue, self.W[:shape[0]], self.W_cl.data)
            return self.W[:shape[0]]
        else :
            return self.W_cl
            
    def calculate_tomograms(self, rs0, pixels):
        """
        we should evaluate in chunks or r
        such that the q-values and classes do not change
        """
        # these are the indices of r where class or q has a new value
        #self.changes = 1 + np.where(np.diff(self.q_indices) + np.diff(self.class_indices))[0]

        rs_out = []
        
        # rs0 is the list of r indices, not neccessarily continuous
        rs = utils.get_chunks(rs0[0], rs0[-1]+1, self.changes)
        
        W_offset = np.int32(0)
        for r00, r11 in rs :
            rs_chunk = np.ascontiguousarray(rs0[ (rs0 >= r00) * (rs0 < r11) ].astype(np.int32))
             
            if len(rs_chunk) == 0 :
                continue
            
            c  = self.class_r[r00]
            q  = self.q_r[r00]
            d  = self.dimensions[c]
            ro = self.rotation_orders[c]
            dr = len(rs_chunk)
            #orientation_offset = np.int32(self.orientation_r[r00])
            #W_offset           = np.int32(dr * len(pixels))
            orientation_r    = self.orientation_r[rs_chunk]
            orientation_r_cl = to_gpu(orientation_r, queue = self.queue)
            
            if d == 2 and ro == 0 :
                event = self.cl_code.calculate_tomograms_static_v0(self.queue, (len(pixels),), None,
                        self.W_cl.data,
                        self.models[c], 
                        self.qxy[q][0].data, 
                        self.qxy[q][1].data, 
                        self.i0, 
                        self.dq,
                        np.int32(0),
                        W_offset)
            
            elif d == 2 and ro > 0 :
                event = self.cl_code.calculate_tomograms_2D_v0(self.queue, (dr, len(pixels)), None,
                        self.W_cl.data,
                        self.models[c], 
                        self.rotation_matrices[(d, ro)].data,
                        self.qxy[q][0].data, 
                        self.qxy[q][1].data, 
                        self.i0, 
                        self.dq,
                        orientation_r_cl.data,
                        np.int32(0),
                        W_offset)
             
            elif d == 3 and ro > 0 :
                event = self.cl_code.calculate_tomograms_3D_v0(self.queue, (dr, len(pixels)), None,
                        self.W_cl.data,
                        self.models[c], 
                        self.rotation_matrices[(d, ro)].data,
                        self.qxy[q][0].data, 
                        self.qxy[q][1].data, 
                        self.qxy[q][2].data, 
                        self.i0, 
                        self.dq,
                        orientation_r_cl.data,
                        np.int32(0),
                        W_offset)
            
            rs_out.append(rs_chunk.copy())
            W_offset += np.int32(dr * len(pixels))

        #assert(np.allclose(np.concatenate(rs_out).ravel(), rs0))
        #print(W_offset, len(rs0), len(pixels), len(rs0) * len(pixels), file = sys.stderr)
        #assert(W_offset == (len(rs0) * len(pixels)))
        return self.W_cl, event
