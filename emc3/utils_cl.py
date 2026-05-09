import os
import warnings
import pathlib

# redirect pyopencl compiler cache to /tmp to use fast local storage and avoid
# network filesystem contention on HPC clusters
_cl_cache_base = pathlib.Path('/tmp') / os.environ.get('USER', 'user')
_cl_cache_base.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(_cl_cache_base))

import pyopencl as cl
import pyopencl.array
import pyopencl.elementwise as elwise
import pyclblast
import numpy as np
import logging

warnings.filterwarnings('ignore', category=cl.CompilerWarning)

logger = logging.getLogger(__name__)

def get_devices(
        device_type='gpu',
        fallback=True,
        preferred_platform_gpu='nvidia',
        preferred_platform_cpu='intel'
        ):
    logger.debug('loading opencl context (gpu prefered)')

    if device_type.lower()=='gpu':
        device_type = cl.device_type.GPU
        preferred_platform = preferred_platform_gpu.lower()
    elif device_type.lower()=='cpu':
        device_type = cl.device_type.CPU
        preferred_platform = preferred_platform_cpu.lower()
    else:
        device_type = cl.device_type.ALL

    # find an opencl device (preferably a GPU)
    # in one of the available platforms
    done = False
    for p in cl.get_platforms():
        devices = p.get_devices(device_type)
        if (len(devices) > 0) and (preferred_platform in p.name.lower()):
            done = True
            break

    if not done:
        for p in cl.get_platforms():
            devices = p.get_devices(device_type)
            if (len(devices) > 0):
                done = True
                break

    if not done and fallback:
        for p in cl.get_platforms():
            devices = p.get_devices()
            if len(devices) > 0:
                break

    if len(devices) == 0:
        raise ValueError('could not any devices')

    logger.debug(f'number of devices: {len(devices)}')
    logger.debug(f'devices: {devices}')

    return devices

def opencl_init(device_no=0):
    devices = get_devices(device_type='gpu')
    device = devices[device_no % len(devices)]

    context = cl.Context([device])

    queue = cl.CommandQueue(context, device)
    return {'context': context, 'queue': queue, 'device': device}


def opencl_init_cpu(rank=0):
    """find an opencl device if available"""
    devices = get_devices(device_type='cpu')

    context = cl.Context(devices)
    queue = cl.CommandQueue(context)
    return {'context': context, 'queue': queue}


def dot(A, x):
    """
    for testing
    """
    opencl_stuff = opencl_init()
    queue = opencl_stuff['queue']

    A_cl = cl.array.empty(
        queue,
        A.shape,
        dtype=np.float32
    )

    x_cl = cl.array.empty(
        queue,
        x.shape,
        dtype=np.float32
    )

    y = np.empty(A.shape[0], dtype=np.float32)

    y_cl = cl.array.empty(
        queue,
        y.shape,
        dtype=np.float32
    )

    cl.enqueue_copy(queue, A_cl.data,
                    np.ascontiguousarray(A.astype(np.float32)))
    cl.enqueue_copy(queue, x_cl.data,
                    np.ascontiguousarray(x.astype(np.float32)))

    pyclblast.gemv(
        queue,
        A.shape[0],
        A.shape[1],
        A_cl,
        x_cl,
        y_cl,
        A.shape[1]
    )

    cl.enqueue_copy(queue, y, y_cl.data)
    return y

def dot_cl(A_cl, x_cl):
    pass



# these are much faster than pyopencl's packaged routines for some reason
def to_gpu(ar, ar_cl=None, queue=None, dtype=None):
    if dtype is None:
        # prefer single precision
        if ar.dtype == np.float64:
            dtype = np.float32
        elif ar.dtype == np.int64:
            dtype = np.int32
        else:
            dtype = ar.dtype

    if ar_cl is None:
        ar_cl = cl.array.empty(queue, ar.shape, dtype=dtype)

    cl.enqueue_copy(queue, ar_cl.data, np.ascontiguousarray(ar.astype(dtype)))
    return ar_cl


def to_cpu(ar_cl, ar=None, queue=None, dtype=None):
    if dtype is None:
        dtype = ar_cl.dtype

    if ar is None:
        ar = np.empty(ar_cl.shape, dtype=dtype)

    cl.enqueue_copy(queue, ar, ar_cl.data)
    return ar


def to_gpu_2D_image_stack(ar, queue=None, context=None):
    # copy I as an opencl "image" for bilinear sampling
    shape = ar.shape
    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    flags = cl.mem_flags.READ_ONLY
    I_cl = cl.Image(
        context, flags, image_format,
        shape=shape[::-1], is_array=True
    )

    cl.enqueue_copy(
        queue, dest=I_cl, src=ar,
        origin=(0, 0, 0), region=shape[::-1]
    )
    return I_cl

def to_gpu_image(ar, queue, context):
    if len(ar.shape) == 2:
        return to_gpu_2D_image(ar, queue, context)
    elif len(ar.shape) == 3:
        return to_gpu_3D_image(ar, queue, context)

def to_gpu_2D_image(ar, queue, context):
    # copy I as an opencl "image" for bilinear sampling
    shape = ar.shape
    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    flags = cl.mem_flags.READ_ONLY

    I_cl = cl.Image(
        context, flags, image_format,
        shape=shape[::-1], is_array=False
    )

    cl.enqueue_copy(
        queue,
        dest=I_cl,
        src=np.ascontiguousarray(ar.T.astype(np.float32)),
        origin=(0, 0), region=shape[::-1]
    )
    return I_cl

def to_gpu_3D_image(ar, queue, context):
    # copy I as an opencl "image" for trilinear sampling
    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    flags = cl.mem_flags.READ_ONLY

    I_cl = cl.Image(context, flags, image_format, shape=ar.shape[::-1])

    cl.enqueue_copy(
        queue,
        I_cl,
        np.ascontiguousarray(ar.T.astype(np.float32)),
        is_blocking=True,
        origin=(0, 0, 0),
        region=ar.shape[::-1]
    )
    return I_cl

class Bincount_cl():

    def __init__(
        self,
        length,
        buffer_shape,
        queue=None,
        context=None
    ):
        self.out = np.zeros(length, dtype=np.int32)
        self.buffer = np.zeros(buffer_shape, dtype=np.int32)

        self.cl_code = cl.Program(context, self.code()).build()
        self.queue = queue
        self.event = None

    def add(self, buffer):
        N = np.int32(buffer.size)
        if self.event is not None:
            self.event[0].wait()

        self.buffer[:N] = buffer.ravel()

        event = cl.Kernel(self.cl_code, 'add')(
                self.queue, (1,), (1,),
                cl.SVM(self.buffer),
                cl.SVM(self.out),
                N
        )
        self.event = [event]

    def code(self):
        return """
        // single worker single group
        __kernel void add (
            global int *array,
            global int *out,
            const  int N
        ) {
        for (int n=0; n<N; n++){
            if (array[n] >= 0)
                out[array[n]] += 1;
        }
        }
        """


class ScatterAdd_cl():
    """
    Parallel weighted scatter-add (histogram) on a CPU OpenCL device.

    Divides the input into n_kernels = max_compute_units // workers_per_kernel
    chunks.  Each chunk is dispatched to its own queue with a private
    accumulator, so CAS contention is limited to workers_per_kernel writers.
    On a laptop (few cores) n_kernels degrades to 1, matching the original
    single-kernel behaviour.

    Usage:
        sa = ScatterAdd_cl(out_size, context=context)
        sa.add(n_sri, N_ri, N_n)   # N_n updated in-place
        sa.add(n_sri, D_ri, D_n)
    """

    _src = """
    inline void atomic_addf(__global float *ptr, float delta) {
        __global unsigned int *uptr = (__global unsigned int *)ptr;
        unsigned int old_bits, new_bits;
        do {
            old_bits = *uptr;
            new_bits = as_uint(as_float(old_bits) + delta);
        } while (atomic_cmpxchg(uptr, old_bits, new_bits) != old_bits);
    }

    __kernel void scatter_add(
        __global const int   *indices,
        __global const float *weights,
        __global       float *out,
        const long start,
        const long end
    ) {
        long gid    = get_global_id(0);
        long stride = get_global_size(0);
        for (long i = start + gid; i < end; i += stride) {
            int idx = indices[i];
            if (idx >= 0)
                atomic_addf(&out[idx], weights[i]);
        }
    }
    """

    def __init__(self, out_size, context=None, workers_per_kernel=4):
        max_cu = context.devices[0].max_compute_units
        self.n_kernels       = max(1, max_cu // workers_per_kernel)
        self.workers_per_kernel = workers_per_kernel
        self.out_size        = out_size
        self._accs           = np.zeros((self.n_kernels, out_size), dtype=np.float32)
        self._queues         = [cl.CommandQueue(context)
                                for _ in range(self.n_kernels)]
        self._prog           = cl.Program(context, self._src).build()

    def add(self, indices_2d, weights_2d, out):
        """
        indices_2d : int32 numpy array, shape (r_chunk, I)
        weights_2d : float32 numpy array, shape (r_chunk, I)
        out        : numpy array, shape (out_size,)  — updated in-place
        """
        idx = np.ascontiguousarray(indices_2d.ravel(), dtype=np.int32)
        w   = np.ascontiguousarray(weights_2d.ravel(), dtype=np.float32)
        N = idx.size
        chunk = (N + self.n_kernels - 1) // self.n_kernels
        self._accs[:] = 0.0

        events = []
        for k in range(self.n_kernels):
            start = k * chunk
            end   = min(start + chunk, N)
            if start >= N:
                break
            events.append(cl.Kernel(self._prog, 'scatter_add')(
                self._queues[k], (self.workers_per_kernel,), None,
                cl.SVM(idx), cl.SVM(w), cl.SVM(self._accs[k]),
                np.int64(start), np.int64(end)
            ))

        for e in events:
            e.wait()

        out += self._accs[:len(events)].sum(axis=0).astype(out.dtype)


class Solve_axbc_cl():

    def __init__(
        self,
        fill_value=0.,
        ftol=1e-2,
        xtol=1e-3,
        maxiters=1000,
        debug=False,
        queue=None,
        context=None
    ):
        self.fill_value = np.float64(fill_value)
        self.ftol = np.float64(ftol)
        self.xtol = np.float64(xtol)
        self.maxiters = np.int64(maxiters)

        self.cl_code = cl.Program(context, self.code()).build()
        self.queue = queue
        self.flag = np.empty(1, dtype=np.int64)

        self.debug = debug

        self.flags = {
            0: f'zero length input returning {fill_value}',
            1: f'negative c-value returning {fill_value}',
            2: 'success',
            3: 'fmax is less than 0 returning 0',
            4: 'Warning maximum iterations exceeded!',
            5: 'analytic solution for N=1',
            6: 'analytic solution for N=2',
            7: 'all b terms are zero setting x to xmax = sum(a) / c'
        }

    def solve(self, a, b, c, out, d):
        event = self.cl_code.solve(
            self.queue, (1,), (1,),
            cl.SVM(a),
            cl.SVM(b),
            np.float64(c),
            cl.SVM(out),
            cl.SVM(self.flag),
            self.fill_value,
            self.ftol,
            self.xtol,
            self.maxiters,
            np.int64(a.size),
            np.int64(d)
        )

        if self.debug:
            self.queue.finish()
            logger.debug(f'{d=} {out[d]=} {self.flags[self.flag[0]]}')
        return event

    def code(self):
        return """
        // optimised for cpu with one worker per group
        // assume all a >  0
        //            b >= 0
        __kernel void solve (
            global double *a,
            global double *b,
            const  double  c,
            global double *out,
            global int    *flag,
            const  double fill_value,
            const  double ftol,
            const  double xtol,
            const  long maxiters,
            const  long I,
            const  long d
        ) {

        double x;

        if (I == 0) {
            out[d]  = fill_value;
            // print(f'zero length input returning {fill_value}')
            flag[0] = 0;
            return;
        }

        if (c <= 0.) {
            out[d]  = fill_value;
            // print(f'negative c-value {c} returning {fill_value}')
            flag[0] = 1;
            return;
        }

        if (I == 1) {
            out[d]  = a[0] / c - b[0];
            flag[0] = 5;          // 'analytic solution for N=1'
            return;
        }

        if (I == 2) {
            double ap, bp, cp;
            ap      = -c;
            bp      = a[0] + a[1] - c * (b[0] + b[1]);
            cp      = a[0] * b[1] + a[1] * b[0] - c * b[0] * b[1];
            out[d]  = (-bp - sqrt(pown(bp, 2) - 4. * ap * cp)) / (2. * ap);
            flag[0] = 6;          // 'analytic solution for N=2'
            return;
        }

        long i, j ;

        double amax = 0.;
        double xmin = 0.;
        double fmax = -1.;
        double xmax = 0.; // np.sum(a)/c
        double fmin = 0.; // np.sum(a / (xmax + b)) - c
        long bzeros = 0;

        for (i=0; i<I; i++) {
            xmax += a[i];

            if (b[i] == 0.)
                bzeros += 1;

            if (a[i]>amax)
                amax = a[i];
        }
        xmax /= c;

        for (i=0; i<I; i++)
            fmin += a[i] / (xmax + b[i]) ;

        fmin -= c;

        if (bzeros == I) {

            out[d] = xmax;
            // 'all b terms are zero setting x to xmax = sum(a) / c'
            flag[0] = 7;
            return;

        } else if (bzeros > 0) {

            xmin = amax / c ;

        } else {

            for (i=0; i<I; i++)
                fmax += a[i] / b[i] ;
            fmax -= c;

            if (fmax <= 0) {
                // print(f'fmax {fmax} is less than 0 returning 0')
                flag[0] = 3;
                out[d]  = 0.;
                return;
            }
        }

        // fmax = fmax or (np.sum(a / (xmin + b)) - c)
        if (fmax < 0.) {
            for (i=0; i<I; i++)
                fmax += a[i] / (xmin + b[i]) ;
            fmax -= c;
        }

        // begin line search
        x = xmax / 2.;

        double fn, fpn, xb, step;

        for (i=0; i < maxiters; i++) {
            // Newton's method
            // x_n+1 = x_n - f(x_n) / f'(x_n)
            fn  = 0.;
            fpn = 0.;
            for (j = 0; j < I; j++) {
                xb   = x + b[j];
                fn  += a[j] / xb ;
                fpn += a[j] / pown(xb, 2) ;
            }
            fn -= c;
            fpn = -fpn;

            step = - fn / fpn;
            x += step;
            x = clamp(x, xmin, xmax);

            if (
                (fabs(step) < (xtol * x))
                && (fabs(fn) < (ftol * (fmax-fmin)))
            ) {
                out[d]  = xmax ;
                flag[0] = 2;
                return ;
            }
        }

        out[d]  = x;
        flag[0] = 4;
        }
        """

def get_fourier_gaussian_kernel(ctx, shape, sigma, factor=1.):
    # Pre-calculate constants to pass as scalars
    nz, ny, nx = shape
    # Gaussian in Fourier space: G(u, v, w) = exp(-2 * pi^2 * sigma^2 * (u^2 + v^2 + w^2))
    # where u, v, w are frequency coordinates from -0.5 to 0.5
    scale = -2.0 * (np.pi ** 2) * (sigma ** 2)

    return elwise.ElementwiseKernel(
        ctx,
        "cfloat_t *data",
        f"""
        // 1. Get 3D indices from global ID i
        int z = i / ({nx} * {ny});
        int y = (i / {nx}) % {ny};
        int x = i % {nx};

        // 2. Convert to frequency coordinates [-0.5, 0.5]
        // This handles the fftshift logic implicitly
        float u = (x < {nx//2}) ? (float)x / {nx} : (float)(x - {nx}) / {nx};
        float v = (y < {ny//2}) ? (float)y / {ny} : (float)(y - {ny}) / {ny};
        float w = (z < {nz//2}) ? (float)z / {nz} : (float)(z - {nz}) / {nz};

        // 3. Calculate Gaussian term
        float freq_sq = u*u + v*v + w*w;
        float gaussian = {factor} * exp({scale} * freq_sq);

        // 4. In-place multiply the Fourier-transformed data
        data[i] = cfloat_rmul(gaussian, data[i]);
        """,
        "fourier_gaussian_multiplier",
        preamble="#include <pyopencl-complex.h>",
    )


def crop_fft_indices_to_grid(raveled_fft_indices, original_shape, values):
    """
    Converts raveled FFT-shifted indices into native indices, crops to a
    tight grid, and returns the result.

    Args:
        raveled_fft_indices: 1D array of indices from an fftshifted array
        original_shape: Tuple (nz, ny, nx) of the full array
        values: 1D array of values at those indices

    Returns:
        grid: Smallest 3D numpy array containing the values
        offset: Tuple (z_min, y_min, x_min) in the full array
    """
    # 1. Unravel the indices to FFT-shifted (centered) coordinates
    # These indices match a grid where zero-freq is at original_shape // 2
    coords = np.array(np.unravel_index(raveled_fft_indices, original_shape))

    # 2. Shift coordinates back to native (standard) NumPy indexing
    # Standard order (np.fft.fft) expects zero at the beginning.
    # We use (coord + (size + 1) // 2) % size to reverse the shift logic.
    native_coords = []
    for i, size in enumerate(original_shape):
        shift = (size + 1) // 2
        # Apply the circular shift to restore native 0-based indexing
        native_coords.append((coords[i] + shift) % size)

    native_coords = np.stack(native_coords, axis=-1)

    # 3. Determine the bounding box in the native coordinate system
    mins = native_coords.min(axis=0)
    maxs = native_coords.max(axis=0)

    # 4. Create the tightly cropped grid
    new_shape = tuple(maxs - mins + 1)
    grid = np.zeros(new_shape, dtype=values.dtype)

    # 5. Fill the grid using local (cropped) coordinates
    local_coords = native_coords - mins
    grid[tuple(local_coords.T)] = values

    return grid, tuple(mins)
