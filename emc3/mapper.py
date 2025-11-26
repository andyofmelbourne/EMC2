import numpy as np
from . import orientations
from . import symmetry as sym

import pyopencl as cl
import pyopencl.array

def calculate_mapping_matrix(dimensions, wav, dq, rotation_order, i0,
                             scale=[1], offsets=[[0, 0, 0]], symmetry='P1'):
    """
    A function for calculating the mapping matrix that maps
    detector pixels to model voxels.

    n_sri = A_sr . (r_i-dr_r) / |r_i-dr_r| + b_sr
          = M_sr(r_i)

    where:
    A_sr = scale_r S_s . R_r / wav dq
    b_sr = - (A_sr . (0, 0, 1) + i0)

    n_sri = A_sr . rh_ri + b_sr
          = [a00 a01 a02] rhx   bx
            [a10 a11 a12].rhy + by
            [a20 a21 a22] rhz   bz

          = [a00 a01 a02 bx] rhx
            [a10 a11 a12 by].rhy
            [a20 a21 a22 bz] rhz
            [drx dry drz 1 ] 1

    On the gpu each r-index will have 4 float4 values:
        M_sr[0-4]
          = [a00 a01 a02 b0]
            [a10 a11 a12 b1]
            [a20 a21 a22 b2]
            [drx dry drz 1 ]

    the latter is more efficient on gpus
    (which treat float3 as float4 under the hood)

    Internally the mapping matrix will be indexed as:
        M_sr -> M_sjkl:
            s = symmetry index (S_s)
            j = offset index (dr_j)
            k = scale index (scale_k)
            l = orientation index (R_l)

    In 2D the mapping matrix will have the size but be filled with zeros where
    necessary:
        M_sr[0-4]
          = [a00 a01 0   b0]
            [a10 a11 0   b1]
            [0   0   0   0 ]
            [drx dry drz 1 ]
    """
    R_l = orientations.get_rotation_matrices(
            rotation_order=rotation_order,
            dimensions=dimensions)

    # get symmetry opperators
    S_s = sym.get_non_voxel_operators(dimensions, symmetry)

    # M_skl
    S, J, K, L = len(S_s), len(offsets), len(scale), len(R_l)

    shape_M = (S, J, K, L, 4, 4)
    M_sjkl = np.zeros(shape_M, dtype=np.float32)

    d = dimensions

    # loop over symmetry ops
    for s in range(S):
        # loop over offsets
        for j in range(J):
            # loop over scales
            for k in range(K):
                A = scale[k] * S_s[s] @ R_l / (wav * dq)

                if d == 2:
                    b = i0 - np.array([0, 0, 1])
                else:
                    b = i0 - A[:, :, 2]

                M_sjkl[s, j, k, :, :d, :d] = np.ascontiguousarray(A)
                M_sjkl[s, j, k, :, :3, 3] = b
                M_sjkl[s, j, k, :, 3, :3] = offsets[j]
                M_sjkl[s, j, k, :, 3, 3] = 1

    return M_sjkl



class Mapper():
    """
    A class for mapping detector pixels to model voxels

    n_sri = A_sr . (r_i-dr_r) / |r_i-dr_r| + b_sr
          = M_sr(r_i)

    Internally the mapping matrix will be indexed as:
        M_sr -> M_sjkl:
            s = symmetry index (S_s)
            j = offset index (dr_j)
            k = scale index (scale_k)
            l = orientation index (R_l)
    """
    def __init__(
            self,
            detector,
            model,
            rotation_order,
            scale=[1,],
            offsets=[[0, 0, 0]]
            ):
        self.xyz = detector.xyz
        self.dimensions = len(model.shape)

        # delay loading pixel coordinates
        self.r = None

        # make mapping matrix:
        # detector pixel -> model voxel
        self.M_sjkl = calculate_mapping_matrix(
                self.dimensions,
                detector.wavelength,
                model.dq,
                rotation_order,
                model.i0,
                scale=scale,
                offsets=offsets,
                symmetry=model.symmetry)

        S, J, K, L, _, _ = self.M_sjkl.shape

        self.model_shape = model.shape
        self.shape0 = (S, J*K*L)
        self.shape = self.shape0
        self.offsets = np.array(offsets)
        self.mask = None

    def load_coords(self, mask):
        if mask is self.mask:
            return

        # make sure it's ready to load onto gpu
        pixels = np.sum(mask)
        self.r = np.ones((pixels, 4), dtype=np.float32)
        self.r[:, :3]  = self.xyz[:, mask].T
        self.rhat = np.ones((pixels, 4), dtype=np.float32)
        self.pixels = pixels
        self.shape = self.shape0 + (pixels,)

    def calculate_mapping(self, r, s=0):
        """
        r = offset, sale & orientation index
        s = symmetry index
        """
        if self.r is None:
            raise ValueError('must run Mapper.load_coords(mask) first!')

        j, k, l = np.unravel_index(r, self.M_sjkl.shape[1:-2])

        self.rhat[:, :3] = self.r[:, :3] - self.offsets[j]
        self.rhat[:, :3] /= np.linalg.norm(self.rhat[:, :3], axis=1)[:, None]

        out = self.M_sjkl[s, j, k, l] @ self.rhat.T

        return out[:self.dimensions, :].T

    def calculate_mapping_ravel(self, r, s=0):
        n = self.calculate_mapping(r, s)
        n = np.rint(n).astype(int)
        n = np.ravel_multi_index(n.T, self.model_shape)
        return n

# should break this down for fusing with tomograms
mapper_code = """
__kernel void mapping(
    global {out_type} *n_ri,
    global float4 *r_i,
    global float4 *M_r,
    const int r_offset
)
{{
    int r = get_global_id(1);
    int i = get_global_id(0);
    int R = get_global_size(1);
    int I = get_global_size(0);

    int base = 4 * (r + r_offset);

    float4 v = r_i[i];

    // offset r - dr
    v = v - M_r[base + 3];

    // normalise (r-dr) / |r-dr|
    v = normalize(v);

    // restore translation bit
    v.w = 1.f;

    // map pixel to voxel
    float r0 = dot(M_r[base + 0], v);
    float r1 = dot(M_r[base + 1], v);
    float r2 = dot(M_r[base + 2], v);

    n_ri[r * I + i] = {out};
}}

__kernel void mapping_rlist(
    global {out_type} *n_ri,
    global float4 *r_i,
    global float4 *M_r,
    global int *rs
)
{{
    int r = get_global_id(1);
    int i = get_global_id(0);
    int R = get_global_size(1);
    int I = get_global_size(0);

    int base = 4 * rs[r];

    float4 v = r_i[i];

    // offset r - dr
    v = v - M_r[base + 3];

    // normalise (r-dr) / |r-dr|
    v = normalize(v);

    // restore translation bit
    v.w = 1.f;

    // map pixel to voxel
    float r0 = dot(M_r[base + 0], v);
    float r1 = dot(M_r[base + 1], v);
    float r2 = dot(M_r[base + 2], v);

    n_ri[r * I + i] = {out};
}}

__kernel void mapping_rlist_pixlist(
    global {out_type} *n_ri,
    global float4 *r_i,
    global float4 *M_r,
    global int *rs,
    global int *is
)
{{
    int r = get_global_id(1);
    int i = get_global_id(0);
    int R = get_global_size(1);
    int I = get_global_size(0);

    int base = 4 * rs[r];

    float4 v = r_i[is[i]];

    // offset r - dr
    v = v - M_r[base + 3];

    // normalise (r-dr) / |r-dr|
    v = normalize(v);

    // restore translation bit
    v.w = 1.f;

    // map pixel to voxel
    float r0 = dot(M_r[base + 0], v);
    float r1 = dot(M_r[base + 1], v);
    float r2 = dot(M_r[base + 2], v);

    n_ri[r * I + i] = {out};
}}
"""

class Mapper_cl():

    def __init__(self, mapper, context, queue):
        self.shape = mapper.shape

        N = mapper.model_shape[0]

        if mapper.dimensions == 2:
            out_vec_type='float2'
            out_vec = '(float2)(r0, r1)'
            out_ravel_type='int'
            out_ravel = f'{N} * convert_int_rte(r0) + convert_int_rte(r1)'

        elif mapper.dimensions == 3:
            out_vec_type='float4'
            out_vec = '(float4)(r0, r1, r2, (float)0.0)'
            out_ravel_type='int'
            out_ravel = f'{N} * {N} * convert_int_rte(r0) +\
                    {N} * convert_int_rte(r1) + convert_int_rte(r2)'

            # test
            # out_ravel = f'convert_int_rte(r0)'
            # out_ravel = f'convert_int_rte(v.w)'

        self.code_vec = cl.Program(
            context,
            mapper_code.format(
                out_type=out_vec_type,
                out=out_vec)
        ).build()

        self.code_ravel = cl.Program(
            context,
            mapper_code.format(
                out_type=out_ravel_type,
                out=out_ravel)
        ).build()

        self.queue = queue
        self.context = context
        self.mapper = mapper

        self.s_chunk_size = None
        self.r_chunk_size = None
        self.ravel = None
        self.buffers_loaded = False

    def load_buffers(self, r_chunk_size=1, ravel=False):
        s_chunk_size = 1
        if (
            ravel == self.ravel
            and s_chunk_size == self.s_chunk_size
            and r_chunk_size == self.r_chunk_size
        ):
            return

        if ravel:
            self.n_ri = np.empty((s_chunk_size, r_chunk_size,
                                  self.mapper.pixels),
                                 dtype=np.int32)
        else:
            self.n_ri = np.empty((s_chunk_size, r_chunk_size,
                                  self.mapper.pixels, self.mapper.dimensions),
                                 dtype=np.float32)

        mf = cl.mem_flags
        self.M_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.mapper.M_sjkl)
        self.r_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.mapper.r)
        self.n_cl = cl.Buffer(self.context, mf.READ_WRITE,
                              self.n_ri.nbytes)

        self.ravel = ravel
        self.s_chunk_size = s_chunk_size
        self.r_chunk_size = r_chunk_size
        self.buffers_loaded = True

    def calculate_mapping(self, s, r00, r11, ravel=False, cpu=True, n_cl=None):
        assert (self.buffers_loaded)

        if ravel:
            code = self.code_ravel
        else:
            code = self.code_vec

        if n_cl is None:
            n_cl = self.n_cl

        r0 = r00 + s * self.mapper.shape[1]
        r1 = r11 + s * self.mapper.shape[1]

        assert((r1 - r0)<=self.r_chunk_size)
        assert(r1<=np.prod(self.mapper.M_sjkl.shape[:4]))

        self.event = cl.Kernel(code, 'mapping')(
            self.queue,
            (self.mapper.pixels, r1-r0),
            None,
            n_cl,
            self.r_cl,
            self.M_cl,
            np.int32(r0)
        )
        self.event.wait()

        if cpu:
            cl.enqueue_copy(self.queue, self.n_ri[0], self.n_cl)
            out = self.n_ri[0, :r1-r0]
        else:
            out = self.n_cl

        return out

    def calculate_mapping_rlist(self, s, rs, ravel=False, cpu=True, n_cl=None):
        if ravel:
            code = self.code_ravel
        else:
            code = self.code_vec

        if n_cl is None:
            n_cl = self.n_cl

        mf = cl.mem_flags
        o = s * self.mapper.shape[1]
        rs = np.ascontiguousarray((rs + o).astype(np.int32))
        rs_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=rs)

        assert(len(rs)<=self.r_chunk_size)
        assert(rs[-1]<=np.prod(self.mapper.M_sjkl.shape[:4]))

        self.event = cl.Kernel(code, 'mapping_rlist')(
            self.queue,
            (self.mapper.pixels, len(rs)),
            None,
            n_cl,
            self.r_cl,
            self.M_cl,
            rs_cl
        )
        self.event.wait()

        if cpu:
            cl.enqueue_copy(self.queue, self.n_ri, self.n_cl)
            out = self.n_ri[0, :len(rs)]
        else:
            out = self.n_cl

        return out


class Mapper_cl_cpu_sparse(Mapper_cl):

    def __init__(self, mapper, context, queue):
        super().__init__(mapper, context, queue)

    def calculate_mapping_rlist_pixlist(self, s, rs, pix, ravel=False):
        if ravel:
            code = self.code_ravel
        else:
            code = self.code_vec

        o = s * self.mapper.shape[1]
        r_s = np.ascontiguousarray((rs + o).astype(np.int32))
        i_s = np.ascontiguousarray(pix.astype(np.int32))

        n_ri = np.empty((len(rs), len(pix)), dtype=np.int32)

        assert(r_s[-1]<np.prod(self.mapper.M_sjkl.shape[:4]))
        assert(i_s[-1]<self.shape[-1])

        self.event = cl.Kernel(code, 'mapping_rlist_pixlist')(
            self.queue,
            (len(pix), len(rs)),
            None,
            cl.SVM(n_ri),
            cl.SVM(self.mapper.r),
            cl.SVM(self.mapper.M_sjkl),
            cl.SVM(r_s),
            cl.SVM(i_s)
        )
        self.event.wait()

        return n_ri

class Mapper_cl_gpu_sparse(Mapper_cl):

    def __init__(self, mapper, context, queue):
        self.buffer_size = 1
        super().__init__(mapper, context, queue)

    def calculate_mapping_rlist_pixlist(self, s, rs, pix, cpu=True, ravel=True):
        if ravel:
            code = self.code_ravel
        else:
            code = self.code_vec

        o = s * self.mapper.shape[1]
        r_s = np.ascontiguousarray((rs + o).astype(np.int32))
        i_s = np.ascontiguousarray(pix.astype(np.int32))

        assert((r_s[-1]+o)<np.prod(self.mapper.M_sjkl.shape[:4]))
        assert(i_s[-1]<self.shape[-1])

        mf = cl.mem_flags
        r_s_cl = cl.Buffer(self.context, mf.READ_ONLY |
                mf.COPY_HOST_PTR, hostbuf=r_s)
        i_s_cl = cl.Buffer(self.context, mf.READ_ONLY |
                mf.COPY_HOST_PTR, hostbuf=i_s)

        self.event = cl.Kernel(code, 'mapping_rlist_pixlist')(
            self.queue,
            (len(pix), len(rs)),
            None,
            self.n_cl,
            self.r_cl,
            self.M_cl,
            r_s_cl,
            i_s_cl
        )
        self.event.wait()

        size = len(rs) * len(pix)
        if cpu:
            cl.enqueue_copy(self.queue, self.n_ri[:size], self.n_cl)
            out = self.n_ri[:size].reshape((len(rs), len(pix)))
        else:
            out = self.n_cl

        return out

    def load_buffers(self, buffer_size, ravel=False):
        if (
            ravel == self.ravel
            and buffer_size == self.buffer_size
        ):
            return

        if ravel:
            self.n_ri = np.empty((buffer_size), dtype=np.int32)
        else:
            self.n_ri = np.empty((buffer_size, self.mapper.dimensions), dtype=np.int32)

        mf = cl.mem_flags
        self.M_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.mapper.M_sjkl)
        self.r_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.mapper.r)
        self.n_cl = cl.Buffer(self.context, mf.READ_WRITE,
                              self.n_ri.nbytes)

        self.ravel = ravel
        self.buffer_size = buffer_size
        self.buffers_loaded = True


