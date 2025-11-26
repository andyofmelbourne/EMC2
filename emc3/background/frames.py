import numpy as np

import pyopencl as cl
import pyopencl.array
from ..utils_cl import to_gpu_image


class Frames():

    def __init__(self, tomo, w_d, B_di):
        self.tomo = tomo
        self.w_d = np.ascontiguousarray(w_d.astype(np.float32))
        self.B_di = B_di

        self.shape = (len(w_d),) + tomo.shape
        self.size = np.prod(self.shape)
        self.ndim = len(self.shape)
        self.dtype = np.float32

    def calculate_frame(self, r, d):
        # C_i W_ri
        CW_ri = self.tomo.calculate_tomogram(r)
        # w_d C_i W_ri + B_di
        frame = self.w_d[d] * CW_ri + self.B_di[d][0]
        return frame

    def calculate_frame_sums(self, chunksize=1024):
        """
        We probably shouldn't ever calculate this
        """
        wsums_r = self.tomo.calculate_wsums(chunksize)

        # large array (is separable)
        # F_dr = sum_i w_d C_i W_ri + B_di
        #      = w_d wsums_r + B_d
        F_dr = self.w_d[:, None] * wsums_r[None, :] + self.B_di.data_sum[:, None]
        return F_dr


class Frames_cl():
    """
    store background + model in gpu memory
    """
    code_cl = """
    constant sampler_t interpolation =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_{interpolation} ;

    __kernel void frame(
        global float *F_dri,
        __read_only image{dim}d_t I_im,
        global float4 *r_i,
        global float4 *M_r,
        global float *C_i,
        global int *j_d,
        global float *w_d,
        global float *b_d,
        global float *B_ji,
        const int r_offset,
        const int d_offset,
        const int D
    )
    {{
        int r = get_global_id(1);
        int i = get_global_id(0);
        int R = get_global_size(1);
        int I = get_global_size(0);

        int d, d0, j;
        float W_ri, F;
        int base = 4 * (r + r_offset);

        float4 v = r_i[i];

        // offset r - dr
        v = v - M_r[base + 3];

        // normalise (r-dr) / |r-dr|
        v = normalize(v);

        // restore translation bit
        v.w = 1.f;

        // map pixel to voxel
        float r0 = dot(M_r[base + 0], v) + 0.5f;
        float r1 = dot(M_r[base + 1], v) + 0.5f;
        float r2 = dot(M_r[base + 2], v) + 0.5f;

        float4 W = read_imagef(I_im, interpolation, {n});
        W_ri = W.x * C_i[i];

        for (d = 0; d < D; d++){{
            d0 = d + d_offset;
            F = w_d[d0] * W_ri + b_d[d0] * B_ji[j_d[d0]*I + i];

            j = d * R * I + r * I + i;
            F_dri[j] = {out};
        }}
    }}
    """

    def __init__(self, frames, context, queue):
        self.frames = frames
        self.queue = queue
        self.context = context

        dim = len(frames.tomo.model.shape)
        self.shape = frames.shape
        self.size = frames.size
        self.ndim = frames.ndim
        self.dtype = np.float32

        if dim == 2:
            n = '(float2)(r0, r1)'
        elif dim == 3:
            n = '(float4)(r0, r1, r2, (float)0.0)'

        code = self.code_cl

        self.code = cl.Program(
            context,
            code.format(
                interpolation=frames.tomo.interpolation_forward.upper(),
                dim=str(dim),
                n=n,
                out='F')
        ).build()

        self.code_log = cl.Program(
            context,
            code.format(
                interpolation=frames.tomo.interpolation_forward.upper(),
                dim=str(dim),
                n=n,
                out='log(fmax(1e-8f, F))')
        ).build()

    def load_buffers(self, r_chunk_size=1, d_chunk_size=1):
        size = d_chunk_size * r_chunk_size * self.shape[2]

        # make sure we don't overflow int32 indexing (I should use uint32)
        assert (size < np.iinfo(np.int32).max)

        self.I_im = to_gpu_image(self.frames.tomo.model.data, self.queue, self.context)

        self.F_dri = np.empty((d_chunk_size * r_chunk_size * self.shape[2]),
                             dtype=np.float32)

        mf = cl.mem_flags
        self.C_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.tomo.C_i)

        self.M_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.tomo.mapper.M_sjkl)

        self.r_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.tomo.mapper.r)

        self.w_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.w_d)

        self.j_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.B_di.j_d)

        self.b_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.B_di.b_d)

        self.B_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.frames.B_di.B_ji)

        self.F_cl = cl.Buffer(self.context, mf.READ_WRITE,
                              self.F_dri.nbytes)


    def calculate_frame(self, d0, d1, r0, r1, log=False, cpu=True):
        if log:
            code = self.code_log
        else:
            code = self.code

        F_cl = self.F_cl

        self.event = cl.Kernel(code, 'frame')(
            self.queue,
            (self.shape[2], r1-r0),
            None,
            F_cl,
            self.I_im,
            self.r_cl,
            self.M_cl,
            self.C_cl,
            self.j_cl,
            self.w_cl,
            self.b_cl,
            self.B_cl,
            np.int32(r0),
            np.int32(d0),
            np.int32(d1-d0)
        )
        self.event.wait()

        if cpu:
            shape = (d1-d0, r1-r0, self.shape[2])
            size = np.prod(shape)
            cl.enqueue_copy(self.queue, self.F_dri[:size], self.F_cl)
            out = self.F_dri[:size].reshape(shape)
        else:
            out = self.F_cl

        return out


class Calc_logR():
    code_logR_cl = """
    // r_block_size must equal R
    __kernel void calc_logR (
        global float *W_ri,
        global uchar *K_di,
        global float *logR_dr,
        global float *w_d,
        global float *B_di,
        const int R,
        const int I
    )
    {{
        int d = get_global_id(0);
        int D = get_global_size(0);

        float F, K, B;
        int i, r;

        local float t[{r_block_size}];

        for (r = 0; r < R; r++){{
            t[r] = 0.;
        }}

        for (i = 0; i < I; i++){{
            K = (float)K_di[d * I + i];
            if (K > 0.) {{
                B = B_di[d*I + i];
                for (r = 0; r < R; r++){{
                    F = w_d[d] * W_ri[I * r + i] + B;
                    if (F > 0.) {{
                        F = log(fmax(1e-8f, F));
                        // logR_dr[R * d + r] += K * F;
                        t[r] += K * F;
                    }}
                }}
            }}
        }}

        for (r = 0; r < R; r++){{
            logR_dr[R * d + r] = t[r];
        }}
    }}
    """
    def __init__(self, frames, context, queue):
        self.context = context
        self.queue = queue
        self.frames = frames

    def calculate_logR(self, d0, d1, dr, W_ri, K_di):
        """
        K_di is already chunked
        W_cl is already chunked (gpu)
        """
        assert (K_di.flags.c_contiguous)
        assert (K_di.size <= self.size)
        assert (np.issubdtype(K_di.dtype, np.uint8))

        B = self.frames.B_di
        self.B_di = B.b_d[d0:d1, None] * B.B_ji[B.j_d[d0:d1]]
        assert (self.B_di.flags.c_contiguous)
        assert (np.issubdtype(self.B_di.dtype, np.float32))

        self.w_d = self.frames.w_d[d0:d1]
        assert (self.w_d.flags.c_contiguous)

        self.event = cl.Kernel(self.code_logR, 'calc_logR')(
            self.queue,
            (d1-d0,),
            (1,),
            cl.SVM(W_ri),
            cl.SVM(K_di),
            cl.SVM(self.logR_dr),
            cl.SVM(self.w_d),
            cl.SVM(self.B_di),
            np.int32(dr),
            np.int32(K_di.shape[1])
        )

        self.event.wait()

        shape = (d1-d0, dr)
        size = np.prod(shape)
        out = self.logR_dr[:size].reshape(shape)
        return out

    def load_logR_buffers(self, r_chunk_size=1, d_chunk_size=1, I_chunk_size=1):
        mf = cl.mem_flags
        self.size = d_chunk_size * r_chunk_size * I_chunk_size

        self.logR_dr = np.empty((d_chunk_size * r_chunk_size),
                             dtype=np.float32)

        self.code_logR = cl.Program(
            self.context,
            self.code_logR_cl.format(r_block_size=r_chunk_size)
        ).build()
