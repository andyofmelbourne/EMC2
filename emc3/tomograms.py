import numpy as np
from scipy.interpolate import interpn


import pyopencl as cl
import pyopencl.array
from .utils_cl import to_gpu_image
from .utils import chunker

class Tomograms():

    def __init__(self, mapper, model, C_i, fluence=None, interpolation_forward='linear'):
        self.mapper = mapper
        self.model = model
        self.fluence = fluence
        self.interpolation_forward = interpolation_forward

        self.C_i = np.ascontiguousarray(C_i.astype(np.float32))

        self.shape = (mapper.shape[1], C_i.shape[0])

        # store linear indices of the model
        n = np.arange(model.shape[0])
        self.n_model = len(model.shape) * (n,)

    def calculate_tomogram(self, r):
        n = self.mapper.calculate_mapping(r)

        out = interpn(
                self.n_model,
                self.model.data,
                n,
                method=self.interpolation_forward,
                bounds_error=True,
                fill_value=0)

        return out * self.C_i

    def calculate_wsums(self, chunksize=1024):
        chunksize = min(chunksize, self.shape[0])

        W_ri = np.empty((chunksize, self.shape[1]), dtype=np.float32)

        wsums_r = np.empty(self.shape[0], dtype=np.float32)

        for r0, r1, dr in chunker(chunksize, self.shape[0]):
            for r in range(r0, r1):
                W_ri[r-r0] = self.calculate_tomogram(r)

            wsums_r[r0:r1] = np.sum(W_ri[:dr], axis=1)
        return wsums_r


class Tomograms_cl():

    def __init__(self, tomo, context, queue):
        # should combine kernels in future
        code = """
        constant sampler_t interpolation =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_{interpolation} ;

        __kernel void tomo(
            global float *W_ri,
            __read_only image{dim}d_t I_im,
            global float4 *r_i,
            global float4 *M_r,
            global float *C_i,
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
            float r0 = dot(M_r[base + 0], v) + 0.5f;
            float r1 = dot(M_r[base + 1], v) + 0.5f;
            float r2 = dot(M_r[base + 2], v) + 0.5f;

            float4 W = read_imagef(I_im, interpolation, {n});
            W_ri[r * I + i] = {out} * C_i[i];
        }}

        __kernel void tomo_list(
            global float *W_ri,
            __read_only image{dim}d_t I_im,
            global float4 *r_i,
            global float4 *M_r,
            global float *C_i,
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
            float r0 = dot(M_r[base + 0], v) + 0.5f;
            float r1 = dot(M_r[base + 1], v) + 0.5f;
            float r2 = dot(M_r[base + 2], v) + 0.5f;

            float4 W = read_imagef(I_im, interpolation, {n});
            W_ri[r * I + i] = {out} * C_i[is[i]];
        }}
        """
        dim = len(tomo.model.shape)
        self.shape = tomo.shape

        if dim == 2:
            n = '(float2)(r0, r1)'
        elif dim == 3:
            n = '(float4)(r0, r1, r2, (float)0.0)'

        self.code = cl.Program(
            context,
            code.format(
                interpolation=tomo.interpolation_forward.upper(),
                dim=str(dim),
                n=n,
                out='W.x')
        ).build()

        self.code_log = cl.Program(
            context,
            code.format(
                interpolation=tomo.interpolation_forward.upper(),
                dim=str(dim),
                n=n,
                out='log(fmax(1e-8f, W.x))')
        ).build()

        self.queue = queue
        self.context = context
        self.tomo = tomo

    def load_buffers(self, r_chunk_size=1, buffer_size=None):
        self.I_im = to_gpu_image(self.tomo.model.data, self.queue, self.context)

        if buffer_size is not None:
            self.W_ri = np.empty((buffer_size), dtype=np.float32)
            self.buffer_size = buffer_size
        else:
            self.W_ri = np.empty((r_chunk_size, self.tomo.shape[1]),
                                 dtype=np.float32)

        print(f'{self.tomo.C_i.shape=} {self.tomo.C_i.dtype=} {self.tomo.C_i.nbytes=}')
        mf = cl.mem_flags
        self.C_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.tomo.C_i)
        self.M_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.tomo.mapper.M_sjkl)
        self.r_cl = cl.Buffer(self.context, mf.READ_ONLY |
                              mf.COPY_HOST_PTR, hostbuf=self.tomo.mapper.r)
        self.W_cl = cl.Buffer(self.context, mf.READ_WRITE,
                              self.W_ri.nbytes)

    def get_W_cl(self, dr):
        self.event.wait()
        cl.enqueue_copy(self.queue, self.W_ri, self.W_cl)
        out = self.W_ri[:dr]
        return out

    def calculate_tomogram(self, r0, r1, log=False, cpu=True, W_cl=None):
        if log:
            code = self.code_log
        else:
            code = self.code

        if W_cl is None:
            W_cl = self.W_cl

        print(f'{r1-r0=} {self.tomo.shape[1]=}')
        print(f'{W_cl=}')
        print(f'{self.r_cl=}')
        print(f'{self.M_cl=}')
        self.event = cl.Kernel(code, 'tomo')(
            self.queue,
            (self.tomo.shape[1], r1-r0),
            None,
            W_cl,
            self.I_im,
            self.r_cl,
            self.M_cl,
            self.C_cl,
            np.int32(r0)
        )

        if cpu:
            out = self.get_W_cl(r1 - r0)
        else:
            out = self.W_cl

        return out

    def calculate_tomogram_rlist_pixlist(self, rs, pix, log=False, cpu=True, W_cl=None):
        if log:
            code = self.code_log
        else:
            code = self.code

        if W_cl is None:
            W_cl = self.W_cl

        mf = cl.mem_flags
        r_s = np.ascontiguousarray(rs.astype(np.int32))
        i_s = np.ascontiguousarray(pix.astype(np.int32))

        # assume increasing
        assert (r_s[-1]<self.shape[0])
        assert (i_s[-1]<self.shape[-1])

        r_s_cl = cl.Buffer(self.context, mf.READ_ONLY |
                mf.COPY_HOST_PTR, hostbuf=r_s)
        i_s_cl = cl.Buffer(self.context, mf.READ_ONLY |
                mf.COPY_HOST_PTR, hostbuf=i_s)

        self.event = cl.Kernel(code, 'tomo_list')(
            self.queue,
            (len(pix), len(rs)),
            None,
            W_cl,
            self.I_im,
            self.r_cl,
            self.M_cl,
            self.C_cl,
            r_s_cl,
            i_s_cl
        )

        size = len(rs) * len(pix)
        if size > self.W_ri.size:
            print()
            print()
            print(f'buffer size too small! {len(rs)=} {len(pix)=} {self.W_ri.size=}')
            print(f'{self.buffer_size=} {id(self)=}')
            print()
            print()

        if cpu:
            self.event.wait()
            cl.enqueue_copy(self.queue, self.W_ri.ravel()[:size], self.W_cl)
            print(f'{size=} {self.W_ri.shape=} {self.W_ri.size=}')
            out = self.W_ri.ravel()[:size].reshape((len(rs), len(pix)))
        else:
            out = self.W_cl

        return out

    def calculate_wsums(self, chunksize=1024, r0=0, r1=None):
        if r1 is None:
            r1 = self.shape[0]

        R = r1-r0

        chunksize = min(chunksize, R)

        self.load_buffers(chunksize)

        wsums_r = np.empty(self.shape[0], dtype=np.float32)

        for r00, r11, dr in chunker(chunksize, R):
            W_ri = self.calculate_tomogram(r0+r00, r0+r11, cpu=True)

            wsums_r[r0+r00:r0+r11] = np.sum(W_ri[:dr], axis=1)
            assert(np.all(wsums_r[r0+r00:r0+r11]>0))

        return wsums_r

