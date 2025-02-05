import numpy as np
import pyopencl as cl
mf = cl.mem_flags


from .utils_cl import to_gpu, to_gpu_2D_image, to_gpu_3D_image

"""
So it seems that we should avoid float3 data dtype: https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_C.html#alignment-of-types

I might get rid of this dynamic buffer stuff, it's messy, and if we want to optimise we can do it manually
"""

def num_to_array(x):
    if not hasattr(x, '__len__'):
        return np.array([x])
    else :
        return x

def compare(x, y):
    """
    if x is None
    or has a different shape to y
    or has different values to y
    return False

    y must be a numpy array
    """
    if x is None :
        return False 
    elif x.shape != y.shape :
        return False
    elif not np.allclose(x, y) :
        return False
    return True
    

code = """
constant sampler_t interpolation = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_{interpolation} ;

float4 mapping (
    global float4 *M,
    global float4 *r
)
{{
    float4 vec = r[0];
    float4 n   = (float4)0.;
    
    // offset r - dr
    vec = vec - M[0];
    
    // normalise (r-dr) / |r-dr|
    vec = normalize(vec);
    
    // dot product A . (r-dr) / |r-dr|
    n.x = dot(M[2], vec);
    n.y = dot(M[3], vec);
    n.z = dot(M[4], vec);
    
    // offset A . (r-dr) / |r-dr| + b
    n = n + M[1];
    
    return n;
}}

//M = [s, r, (dr, b, A)]
__kernel void mapping_3D_v0 (
    global float4 *out_n,  
    global float4 *M, 
    global float4 *r,
    global int *rs
)
{{
    int rot = get_global_id(1);
    int i   = get_global_id(0);
    int I   = get_global_size(0);
    
    out_n[rot * I + i] = mapping(M + rs[rot] * 5, r + i);
}}

__kernel void tomo_2D (
    __read_only image2d_t I, 
    global float *out,
    global float4 *M, 
    global float4 *r,
    global int *rs
)
{{
    int rot    = get_global_id(1);
    int i      = get_global_id(0);
    int pixels = get_global_size(0);
    
    float4 n = mapping(M + rs[rot] * 5, r + i);
    
    float4 W = read_imagef(I, interpolation, (float2)(n.x, n.y));
    //out[rot * pixels + i] = W.x;
    out[rot * pixels + i] = {Wmap};
}}


__kernel void tomo_3D (
    __read_only image3d_t I, 
    global float *out,
    global float4 *M, 
    global float4 *r,
    global int *rs
)
{{
    int rot    = get_global_id(1);
    int i      = get_global_id(0);
    int pixels = get_global_size(0);
    
    float4 n = mapping(M + rs[rot] * 5, r + i);
    
    float4 W = read_imagef(I, interpolation, n);
    //out[rot * pixels + i] = W.x;
    out[rot * pixels + i] = {Wmap};
}}

"""


class Mapper():
    """
    A class for mapping pixels to voxels
    
    n_sri = A_sr . (r-dr) / |r-dr| + b_sr
    """
    
    def __init__(self, dimensions, xyz, mapping_matrix, context, queue, interpolation = 'linear'):
        
        if interpolation == 'linear':
            self.interpolation = 'LINEAR'
        
        elif interpolation == 'nearest':
            self.interpolation = 'NEAREST'
        
        else :
            raise ValueError(f'forward interpolation strategy {interpolation} not supported')
        
        self.mapping_matrix = mapping_matrix
    
        self.xyz            = np.zeros((4, xyz.shape[1]), dtype = xyz.dtype)
        self.xyz[:3, :]     = xyz
        
        shape = mapping_matrix.shape
        self.mapping_matrix = np.zeros((shape[0], shape[1], shape[2], 4), dtype = mapping_matrix.dtype)
        self.mapping_matrix[:, :, :, :3] = mapping_matrix
        
        self.symmetry = np.arange(mapping_matrix.shape[0])
        self.rs       = np.arange(mapping_matrix.shape[1])
        self.pixels   = np.arange(xyz.shape[1])

        self.shape = (mapping_matrix.shape[0], mapping_matrix.shape[1], xyz.shape[1], 3)
        
        self.context = context
        self.queue   = queue
        
        self.last_pixel_inds    = None
        self.last_symmetry_inds = None
        self.last_r_inds        = None
        
        self.xyz_i_cl           = None
        self.M_cl               = None
        self.n_sri_cl           = None
        self.rs_cl              = None
        
        # keep full mapping matrix on gpu
        self.update_M_buffer()
        
        self.code     = cl.Program(self.context, code.format(interpolation = self.interpolation, Wmap = 'W.x')).build()
        self.code_log = cl.Program(self.context, code.format(interpolation = self.interpolation, Wmap = 'log(W.x)')).build()
    
    def update_xyz_buffer(self, pixel_inds):
        if not compare(self.last_pixel_inds, pixel_inds):
            xyz_i         = np.ascontiguousarray(np.transpose(self.xyz[:, pixel_inds]).astype(np.float32))
            self.xyz_i_cl = cl.Buffer(self.context, mf.READ_ONLY, xyz_i.nbytes)
            cl.enqueue_copy(self.queue, self.xyz_i_cl, xyz_i)
            self.last_pixel_inds = pixel_inds

    def update_M_buffer(self):
        # these are the mapping-vectors for each s,r pair (symmetries, rotations, 5) of type float3
        M    = np.ascontiguousarray(self.mapping_matrix.astype(np.float32))
        self.M_cl = cl.Buffer(self.context, mf.READ_ONLY, M.nbytes)
        cl.enqueue_copy(self.queue, self.M_cl, M)

    def update_n_buffer(self, size):
        if (not self.n_sri_cl) or ((4*size) > self.n_sri.size) :
            # make output buffer
            self.n_sri    = np.empty((4 * size,), dtype = np.float32)
            self.n_sri_cl = cl.Buffer(self.context, mf.WRITE_ONLY, self.n_sri.nbytes)
    
    def update_rs_buffer(self, symmetry_inds, r_inds):
        if (not self.rs_cl) or \
           (not compare(self.last_symmetry_inds, symmetry_inds)) or \
           (not compare(self.last_r_inds, r_inds)) :
            
            rs = len(self.rs) * symmetry_inds[:, None] + r_inds[None, :]
            rs = np.ascontiguousarray(rs.astype(np.int32))
            self.rs_cl = cl.Buffer(self.context, mf.READ_ONLY, rs.nbytes)
            cl.enqueue_copy(self.queue, self.rs_cl, rs)
            
            self.last_symmetry_inds = symmetry_inds
            self.last_r_inds        = r_inds

    def parse_key(self, key):
        if len(key) > 0 :
            symmetry_inds = num_to_array(self.symmetry[key[0]])
        
        if len(key) > 1 :
            r_inds = num_to_array(self.rs[key[1]])
        else :
            r_inds = self.rs

        if len(key) > 2 :
            pixel_inds = num_to_array(self.pixels[key[2]])
        else :
            pixel_inds = self.pixels

        size = len(symmetry_inds) * len(r_inds) * len(pixel_inds)
        return symmetry_inds, r_inds, pixel_inds, size

    def update_buffers(self, symmetry_inds, r_inds, pixel_inds, size):
        self.update_xyz_buffer(pixel_inds)
        self.update_n_buffer(size)
        self.update_rs_buffer(symmetry_inds, r_inds)
    
    def __getitem__(self, key):
        symmetry_inds, r_inds, pixel_inds, size = self.parse_key(key)
        
        self.update_buffers(symmetry_inds, r_inds, pixel_inds, size)
        
        event = self.code.mapping_3D_v0(
            self.queue,
            (len(pixel_inds), len(symmetry_inds) * len(r_inds)),
            None,
            self.n_sri_cl,  
            self.M_cl, 
            self.xyz_i_cl,
            self.rs_cl
        )
        
        cl.enqueue_copy(self.queue, self.n_sri, self.n_sri_cl)
        
        out   = self.n_sri[: 4 * size]
        shape = (len(symmetry_inds), len(r_inds), len(pixel_inds), 4)
        out   = out.reshape(shape)[:, :, :, :3]
        return out


class Tomograms():
    """
    A class for evaluating model tomorgams

    Does not include symmetry operations
    """
    def __init__(self, mapper, model):
        self.model   = model
        self.mapper  = mapper
        self.queue   = mapper.queue
        self.context = mapper.context
        
        self.shape = mapper.shape[1:-1]
        self.size  = np.prod(self.shape)
        self.dtype = np.float32
        
        self.W_ri_cl = None
        self.cpu      = True
        self.log      = False
        
        self.update_model()
        self.set_log(self.log)

    def set_log(self, log = False):
        self.log = log
        d = self.model.ndim
        
        if log :
            code = self.mapper.code_log
        else :
            code = self.mapper.code
        
        if (d == 3) :
            self.tomo = code.tomo_3D
        elif (d == 2) :
            self.tomo = code.tomo_2D
        
    
    def update_model(self):
        d = self.model.ndim
        if d == 3 :
            self.I_cl = to_gpu_3D_image(self.model, self.queue, self.context)
        
        elif d == 2 :
            self.I_cl = to_gpu_2D_image(self.model, self.queue, self.context)
        
        else :
            raise ValueError(f'could parse dimension {d}')
    
    def update_W_buffer(self, shape):
        if (self.W_ri_cl is None) or (shape != self.W_ri.shape) :
            # make output buffer
            self.W_ri    = np.empty(shape, dtype = np.float32)
            #self.W_sri_cl = cl.Buffer(self.context, mf.WRITE_ONLY, self.W_sri.nbytes)
            # pyclblast.gemm needs arrays
            self.W_ri_cl = cl.array.empty(self.queue, shape, dtype = self.dtype)
    
    def get(self):
        cl.enqueue_copy(self.queue, self.W_ri, self.W_ri_cl.data)
        
        return self.W_ri
    
    def __getitem__(self, key):
        key = (0,) + key
        symmetry_inds, r_inds, pixel_inds, size = self.mapper.parse_key(key)
        self.buffer_size  = size
        self.buffer_shape = (len(r_inds), len(pixel_inds))
        
        self.mapper.update_xyz_buffer(pixel_inds)
        self.mapper.update_rs_buffer(symmetry_inds, r_inds)
        self.update_W_buffer(self.buffer_shape)
        
        self.event = self.tomo(
            self.queue,
            (len(pixel_inds), len(symmetry_inds) * len(r_inds)),
            None,
            self.I_cl,  
            self.W_ri_cl.data,  
            self.mapper.M_cl, 
            self.mapper.xyz_i_cl,
            self.mapper.rs_cl
        )

        if self.cpu :
            out = self.get()
        else :
            out = self.W_ri_cl
        
        return out
