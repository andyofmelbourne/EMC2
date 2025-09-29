import numpy as np
import logging
import pyopencl as cl
from .utils_cl import to_gpu_2D_image, to_gpu_3D_image

logger = logging.getLogger(__name__)

mf = cl.mem_flags

"""
So it seems that we should avoid float3 data dtype:
    https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_C.html#alignment-of-types

I might get rid of this dynamic buffer stuff, it's messy,
and if we want to optimise we can do it manually
"""


def num_to_array(x):
    if not hasattr(x, '__len__'):
        return np.array([x])
    else:
        return x


def compare(x, y):
    """
    if x is None
    or has a different shape to y
    or has different values to y
    return False

    y must be a numpy array
    """
    if x is None:
        return False
    elif x.shape != y.shape:
        return False
    elif not np.allclose(x, y):
        return False
    return True


code = """
constant sampler_t interpolation =
CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_{interpolation} ;

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

// M = [s, r, (dr, b, A)]
// return x,y,z,_ voxel coordinates
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


// return ravelled voxel coordinates
__kernel void mapping_2D_n(
    global int *out_n,
    global float4 *M,
    global float4 *r,
    global int *rs,
    const int N
)
{{
    int rot = get_global_id(1);
    int i   = get_global_id(0);
    int I   = get_global_size(0);

    float4 n;

    n = mapping(M + rs[rot] * 5, r + i);

    int m = N * convert_int_rte(n.x);
    m += convert_int_rte(n.y);
    out_n[rot * I + i] = m;
}}


// return raveled voxel coordinates
__kernel void mapping_3D_n(
    global int *out_n,
    global float4 *M,
    global float4 *r,
    global int *rs,
    const int N
)
{{
    int rot = get_global_id(1);
    int i   = get_global_id(0);
    int I   = get_global_size(0);

    float4 n;

    n = mapping(M + rs[rot] * 5, r + i);

    int m = N * N * convert_int_rte(n.x);
    m += N * convert_int_rte(n.y);
    m += convert_int_rte(n.z);
    out_n[rot * I + i] = m;
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

    float4 n = mapping(M + rs[rot] * 5, r + i) + (float)0.5;

    float4 W = read_imagef(I, interpolation, (float2)(n.x, n.y));

    int j = rot * pixels + i;
    //out[j] = W.x;
    {Wmap};
}}


__kernel void tomo_3D (
    __read_only image3d_t I,
    global float *out,
    global float4 *M,
    global float4 *r, // 3 dimension pixel coords
    global int *rs // list of r-indices
)
{{
    int rot    = get_global_id(1);
    int i      = get_global_id(0);
    int pixels = get_global_size(0);

    float4 n = mapping(M + rs[rot] * 5, r + i) + (float)0.5;

    float4 W = read_imagef(I, interpolation, n);

    int j = rot * pixels + i;
    //out[j] = W.x;
    {Wmap};
}}


// out = K_di log(F_dri)
__kernel void calculate_K_logF_dri_3D (
    __read_only image3d_t I,
    global float *out,
    global float4 *M, // mapping matrix
    global float4 *r, // 3 dimension pixel coords
    global int *rs,    // list of r-indices
    global uchar *K_di,
    global float *w_d,
    global float *C_i,
    global float *B_di,
    const int w_offset,
    const int C_offset,
    const int D)
{{
    int rot = get_global_id(1);
    int i = get_global_id(0);
    int pixels = get_global_size(0);
    int rotations = get_global_size(1);

    float F, W_ri;

    // calculate tomogram W_ri
    float4 n = mapping(M + rs[rot] * 5, r + i) + (float)0.5;
    float4 W = read_imagef(I, interpolation, n);
    W_ri = W.x;

    // calculate F_dri
    for (int d=0; d<D; d++) {{
        F = w_d[d + w_offset] * C_i[i + C_offset] *
            W_ri + B_di[d * pixels + i];

        int j = d * rotations * pixels + rot * pixels + i;

        if (F>0.)
            out[j] = (float)K_di[d * pixels + i] * log(F);
        else
            out[j] = 0.;
    }}
}}


// out = K_di log(F_dri)
__kernel void calculate_K_logF_dri_2D (
    __read_only image2d_t I,
    global float *out,
    global float4 *M, // mapping matrix
    global float4 *r, // 3 dimension pixel coords
    global int *rs,    // list of r-indices
    global uchar *K_di,
    global float *w_d,
    global float *C_i,
    global float *B_di,
    const int w_offset,
    const int C_offset,
    const int D)
{{
    int rot = get_global_id(1);
    int i = get_global_id(0);
    int pixels = get_global_size(0);
    int rotations = get_global_size(1);

    float F, W_ri;

    // calculate tomogram W_ri
    float4 n = mapping(M + rs[rot] * 5, r + i) + (float)0.5;
    float4 W = read_imagef(I, interpolation, (float2)(n.x, n.y));
    W_ri = W.x;

    // calculate F_dri
    for (int d=0; d<D; d++) {{
        F = w_d[d + w_offset] * C_i[i + C_offset] *
            W_ri + B_di[d * pixels + i];

        int j = d * rotations * pixels + rot * pixels + i;

        if (F>0.)
            out[j] = (float)K_di[d * pixels + i] * log(F);
        else
            out[j] = 0.;
    }}
}}


// out = F_dri = w_d C_i W_ri + B_di
__kernel void calculate_F_dri_3D (
    __read_only image3d_t I,
    global float *out,
    global float4 *M, // mapping matrix
    global float4 *r, // 3 dimension pixel coords
    global int *rs,    // list of r-indices
    global float *w_d,
    global float *C_i,
    global float *B_di,
    const int w_offset,
    const int C_offset,
    const int D)
{{
    int rot = get_global_id(1);
    int i = get_global_id(0);
    int pixels = get_global_size(0);
    int rotations = get_global_size(1);

    float F, W_ri;

    // calculate tomogram W_ri
    float4 n = mapping(M + rs[rot] * 5, r + i) + (float)0.5;
    float4 W = read_imagef(I, interpolation, n);
    W_ri = W.x;

    // calculate F_dri
    for (int d=0; d<D; d++) {{
        F = w_d[d + w_offset] * C_i[i + C_offset] *
            W_ri + B_di[d * pixels + i];

        int j = d * rotations * pixels + rot * pixels + i;

        out[j] = F;
    }}
}}


// out = F_dri = w_d C_i W_ri + B_di
__kernel void calculate_F_dri_2D (
    __read_only image2d_t I,
    global float *out,
    global float4 *M, // mapping matrix
    global float4 *r, // 3 dimension pixel coords
    global int *rs,    // list of r-indices
    global float *w_d,
    global float *C_i,
    global float *B_di,
    const int w_offset,
    const int C_offset,
    const int D)
{{
    int rot = get_global_id(1);
    int i = get_global_id(0);
    int pixels = get_global_size(0);
    int rotations = get_global_size(1);

    float F, W_ri;

    // calculate tomogram W_ri
    float4 n = mapping(M + rs[rot] * 5, r + i) + (float)0.5;
    float4 W = read_imagef(I, interpolation, (float2)(n.x, n.y));
    W_ri = W.x;

    // calculate F_dri
    for (int d=0; d<D; d++) {{
        F = w_d[d + w_offset] * C_i[i + C_offset] *
            W_ri + B_di[d * pixels + i];

        int j = d * rotations * pixels + rot * pixels + i;

        out[j] = F;
    }}
}}

"""


class Mapper():
    """
    A class for mapping pixels to voxels

    n_sri = A_sr . (r-dr) / |r-dr| + b_sr
    """

    def __init__(
        self, dimensions, model_width, xyz, mapping_matrix,
        context, queue, interpolation='linear'
    ):
        if interpolation == 'linear':
            self.interpolation = 'LINEAR'

        elif interpolation == 'nearest':
            self.interpolation = 'NEAREST'

        else:
            raise ValueError(
                f'forward interpolation strategy {interpolation} not supported'
            )

        self.mapping_matrix = mapping_matrix

        self.xyz = np.zeros((4, xyz.shape[1]), dtype=xyz.dtype)
        self.xyz[:3, :] = xyz

        shape = mapping_matrix.shape

        self.mapping_matrix = np.zeros(
            (shape[0], shape[1], shape[2], 4),
            dtype=mapping_matrix.dtype
        )

        self.mapping_matrix[:, :, :, :3] = mapping_matrix

        self.symmetry = np.arange(mapping_matrix.shape[0])

        self.rs = np.arange(mapping_matrix.shape[1])

        self.pixels = np.arange(xyz.shape[1])

        self.shape = (
            mapping_matrix.shape[0], mapping_matrix.shape[1], xyz.shape[1]
        )
        self.dtype = np.int32

        self.model_width = np.int32(model_width)

        self.cpu = True

        self.context = context
        self.queue = queue

        self.last_pixel_inds = None
        self.last_symmetry_inds = None
        self.last_r_inds = None

        self.xyz_i_cl = None
        self.M_cl = None
        self.n_sri_cl = None
        self.rs_cl = None

        # keep full mapping matrix on gpu
        self.update_M_buffer()

        wmap = 'out[j] = W.x;'
        wmap_log = """
        if (W.x > 0.)
            out[j] = log(W.x);
        else
            out[j] = 0.;
        """

        self.code = cl.Program(
            self.context,
            code.format(
                interpolation=self.interpolation,
                Wmap=wmap)
        ).build()

        self.code_log = cl.Program(
            self.context,
            code.format(
                interpolation=self.interpolation,
                Wmap=wmap_log)
        ).build()

        if dimensions == 2:
            self.calculate_mapping = self.code.mapping_2D_n
        elif dimensions == 3:
            self.calculate_mapping = self.code.mapping_3D_n
        else:
            raise ValueError(f'{dimensions=} not supported')

    def update_xyz_buffer(self, pixel_inds):
        if not compare(self.last_pixel_inds, pixel_inds):
            xyz_i = np.ascontiguousarray(
                np.transpose(self.xyz[:, pixel_inds]).astype(np.float32)
            )

            self.xyz_i_cl = cl.Buffer(self.context, mf.READ_ONLY, xyz_i.nbytes)

            cl.enqueue_copy(self.queue, self.xyz_i_cl, xyz_i)

            self.last_pixel_inds = pixel_inds

    def update_M_buffer(self):
        # these are the mapping-vectors for each s,r pair
        # (symmetries, rotations, 5) of type float3
        M = np.ascontiguousarray(self.mapping_matrix.astype(np.float32))
        self.M_cl = cl.Buffer(self.context, mf.READ_ONLY, M.nbytes)
        cl.enqueue_copy(self.queue, self.M_cl, M)

    def update_n_buffer(self, size):
        if (not self.n_sri_cl) or (size > self.n_sri.size):
            # make output buffer
            self.n_sri = np.empty((size,), dtype=np.int32)
            self.n_sri_cl = cl.Buffer(
                self.context, mf.WRITE_ONLY, self.n_sri.nbytes)

    def update_rs_buffer(self, symmetry_inds, r_inds):
        if (not self.rs_cl) or \
           (not compare(self.last_symmetry_inds, symmetry_inds)) or \
           (not compare(self.last_r_inds, r_inds)):

            rs = len(self.rs) * symmetry_inds[:, None] + r_inds[None, :]
            rs = np.ascontiguousarray(rs.astype(np.int32))
            self.rs_cl = cl.Buffer(self.context, mf.READ_ONLY, rs.nbytes)
            cl.enqueue_copy(self.queue, self.rs_cl, rs)

            self.last_symmetry_inds = symmetry_inds
            self.last_r_inds = r_inds

    def parse_key(self, key):
        if len(key) > 0:
            symmetry_inds = num_to_array(self.symmetry[key[0]])

        if len(key) > 1:
            r_inds = num_to_array(self.rs[key[1]])
        else:
            r_inds = self.rs

        if len(key) > 2:
            pixel_inds = num_to_array(self.pixels[key[2]])
        else:
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

        self.event = self.calculate_mapping(
            self.queue,
            (len(pixel_inds), len(symmetry_inds) * len(r_inds)),
            None,
            self.n_sri_cl,
            self.M_cl,
            self.xyz_i_cl,
            self.rs_cl,
            self.model_width
        )

        if self.cpu:
            cl.enqueue_copy(self.queue, self.n_sri, self.n_sri_cl)

            out = self.n_sri[: size]
            shape = (len(symmetry_inds), len(r_inds), len(pixel_inds))
            out = out.reshape(shape)
        else:
            out = self.n_sri_cl

        return out


class Tomograms():
    """
    A class for evaluating model tomorgams

    Does not include symmetry operations
    """
    def __init__(self, mapper, model):
        self.model = model
        self.mapper = mapper
        self.queue = mapper.queue
        self.context = mapper.context

        self.shape = mapper.shape[1:]
        self.size = np.prod(self.shape)
        self.dtype = np.float32

        self.W_ri_cl = None
        self.cpu = True
        self.log = False

        self.update_model()
        self.set_log(self.log)

    def set_log(self, log=False):
        self.log = log
        d = self.model.ndim

        if log:
            code = self.mapper.code_log

        else:
            code = self.mapper.code

        if (d == 3):
            self.tomo = code.tomo_3D

        elif (d == 2):
            self.tomo = code.tomo_2D

    def update_model(self):
        d = self.model.ndim
        if d == 3:
            self.I_cl = to_gpu_3D_image(self.model, self.queue, self.context)

        elif d == 2:
            self.I_cl = to_gpu_2D_image(self.model, self.queue, self.context)

        else:
            raise ValueError(f'could parse dimension {d}')

    def update_W_buffer(self, shape):
        if (self.W_ri_cl is None) or (shape != self.W_ri.shape):
            # make output buffer
            self.W_ri = np.empty(shape, dtype=np.float32)
            # self.W_sri_cl = cl.Buffer(
            #   self.context, mf.WRITE_ONLY, self.W_sri.nbytes)
            # pyclblast.gemm needs arrays
            self.W_ri_cl = cl.array.empty(self.queue, shape, dtype=self.dtype)

    def get(self):
        cl.enqueue_copy(self.queue, self.W_ri, self.W_ri_cl.data)
        return self.W_ri

    def __getitem__(self, key):
        key = (0,) + key
        symmetry_inds, r_inds, pixel_inds, size = self.mapper.parse_key(key)
        self.buffer_size = size
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

        if self.cpu:
            out = self.get()
        else:
            out = self.W_ri_cl

        return out


class Frames():
    """
    F_dri = w_d C_i W_ri + B_di

    K_logF_F_dr = sum_i K_di logF_dri - F_dr
    """

    def __init__(
        self, context, queue, K_di, B_di, w_d, model,
        M_srn, xyz_i, C_i, interpolation='linear'
    ):
        self.context = context
        self.queue = queue

        if interpolation == 'linear':
            self.interpolation = 'LINEAR'

        elif interpolation == 'nearest':
            self.interpolation = 'NEAREST'

        else:
            raise ValueError(f'forward interpolation strategy'
                             f'{interpolation} not supported')

        wmap = 'out[j] = W.x;'
        self.code = cl.Program(
            self.context,
            code.format(
                interpolation=self.interpolation,
                Wmap=wmap)
        ).build()

        self.I_n_cl = None
        self.xyz_i_cl = None
        self.K_di_cl = None
        self.B_di_cl = None
        self.C_i_cl = None
        self.w_d_cl = None
        self.M_srn_cl = None
        self.rs_cl = None
        self.F_dri_cl = None
        self.w_offset = np.int32(0)
        self.C_offset = np.int32(0)

        self.K_di = K_di
        self.B_di = B_di
        self.w_d = w_d
        self.model = model
        self.M_srn = M_srn
        self.C_i = C_i
        self.xyz_i = xyz_i

        self.shape = (K_di.shape[0], M_srn.shape[1], K_di.shape[1])
        self.dtype = np.float32
        self.size = np.prod(self.shape)

        self.update_model(model)
        self.update_mapping(M_srn)
        self.update_C(C_i)
        self.update_xyz(xyz_i)

        if model.ndim == 2:
            self.calculate_K_logF_dri = self.code.calculate_K_logF_dri_2D
            self.calculate_F_dri = self.code.calculate_F_dri_2D

        elif model.ndim == 3:
            self.calculate_K_logF_dri = self.code.calculate_K_logF_dri_3D
            self.calculate_F_dri = self.code.calculate_F_dri_3D

    def update_frames(self, d0, d1):
        self.update_w(self.w_d[d0:d1])
        self.update_K(self.K_di[d0:d1, :])
        self.update_B(self.B_di[d0:d1, :])

    def update_rotations(self, r_inds):
        self.rotations = np.int32(r_inds.shape[0])

        rs = np.ascontiguousarray(r_inds.astype(np.int32))

        if self.rs_cl is None or self.rs_cl.size < rs.nbytes:
            self.rs_cl = cl.Buffer(self.context, mf.READ_ONLY, rs.nbytes)

        cl.enqueue_copy(self.queue, self.rs_cl, rs)

    def update_mapping(self, M_srn):
        shape = M_srn.shape

        M_c = np.zeros(
            (shape[0], shape[1], shape[2], 4),
            dtype=np.float32
        )

        M_c[:, :, :, :3] = M_srn

        if self.M_srn_cl is None or self.M_srn_cl.size < M_c.nbytes:
            self.M_srn_cl = cl.Buffer(self.context, mf.READ_ONLY, M_c.nbytes)

        cl.enqueue_copy(self.queue, self.M_srn_cl, M_c)

    def update_B(self, B_di):
        B_c = np.ascontiguousarray(B_di.astype(np.float32))

        if self.B_di_cl is None or self.B_di_cl.size < B_c.nbytes:
            self.B_di_cl = cl.Buffer(self.context, mf.READ_ONLY, B_c.nbytes)

        cl.enqueue_copy(self.queue, self.B_di_cl, B_c)

    def update_xyz(self, xyz_i):
        # xyz (x/y/z, pixel index) location of each pixel (float)
        # to xyz_i_cl (pixel index, x/y/z/w) (float4)
        xyz_c = np.zeros((4, xyz_i.shape[1]), dtype=np.float32)
        xyz_c[:3, :] = xyz_i

        xyz_c = np.ascontiguousarray(np.transpose(xyz_c).astype(np.float32))

        if self.xyz_i_cl is None or self.xyz_i_cl.size < xyz_c.nbytes:
            self.xyz_i_cl = cl.Buffer(self.context, mf.READ_ONLY, xyz_c.nbytes)

        cl.enqueue_copy(self.queue, self.xyz_i_cl, xyz_c)

    def update_K(self, K_di):
        self.frames, self.pixels = K_di.shape
        self.frames = np.int32(self.frames)
        self.pixels = np.int32(self.pixels)

        K_c = np.ascontiguousarray(K_di.astype(np.uint8))

        if self.K_di_cl is None or self.K_di_cl.size < K_c.nbytes:
            self.K_di_cl = cl.Buffer(self.context, mf.READ_ONLY, K_c.nbytes)

        cl.enqueue_copy(self.queue, self.K_di_cl, K_c)

    def update_w(self, w_d):
        w_c = np.ascontiguousarray(w_d.astype(np.float32))

        if self.w_d_cl is None or self.w_d_cl.size < w_c.nbytes:
            self.w_d_cl = cl.Buffer(self.context, mf.READ_ONLY, w_c.nbytes)

        cl.enqueue_copy(self.queue, self.w_d_cl, w_c)

    def update_C(self, C_i):
        C_c = np.ascontiguousarray(C_i.astype(np.float32))

        if self.C_i_cl is None or self.C_i_cl.size < C_c.nbytes:
            self.C_i_cl = cl.Buffer(self.context, mf.READ_ONLY, C_c.nbytes)

        cl.enqueue_copy(self.queue, self.C_i_cl, C_c)

    def update_model(self, model):
        d = model.ndim
        if d == 3:
            self.I_n_cl = to_gpu_3D_image(model, self.queue, self.context)

        elif d == 2:
            self.I_n_cl = to_gpu_2D_image(model, self.queue, self.context)

        else:
            raise ValueError(f'could parse dimension {d}')

    def K_logF_F(self, out=None):
        shape = (self.frames, self.rotations, self.pixels)

        if out is None:
            out = np.empty(shape, dtype=np.float32)

        size = out.nbytes

        if self.F_dri_cl is None or self.F_dri_cl.size < size:
            logger.info(f'creating F_dri buffer of size {size} bytes')
            self.F_dri_cl = cl.Buffer(self.context, mf.WRITE_ONLY, size)

        self.calculate_K_logF_dri(
            self.queue,
            (self.pixels, self.rotations),
            None,
            self.I_n_cl,
            self.F_dri_cl,
            self.M_srn_cl,
            self.xyz_i_cl,
            self.rs_cl,
            self.K_di_cl,
            self.w_d_cl,
            self.C_i_cl,
            self.B_di_cl,
            self.w_offset,
            self.C_offset,
            self.frames
        )

        cl.enqueue_copy(self.queue, out, self.F_dri_cl)
        return out

    def F(self, out=None):
        shape = (self.frames, self.rotations, self.pixels)

        if out is None:
            out = np.empty(shape, dtype=np.float32)

        size = out.nbytes

        if self.F_dri_cl is None or self.F_dri_cl.size < size:
            logger.info(f'creating F_dri buffer of size {size} bytes')
            self.F_dri_cl = cl.Buffer(self.context, mf.WRITE_ONLY, size)

        self.calculate_F_dri(
            self.queue,
            (self.pixels, self.rotations),
            None,
            self.I_n_cl,
            self.F_dri_cl,
            self.M_srn_cl,
            self.xyz_i_cl,
            self.rs_cl,
            self.w_d_cl,
            self.C_i_cl,
            self.B_di_cl,
            self.w_offset,
            self.C_offset,
            self.frames
        )

        cl.enqueue_copy(self.queue, out, self.F_dri_cl)
        return out
