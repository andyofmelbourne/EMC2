import sys
import pyopencl as cl
import pyopencl.array
import numpy as np


def opencl_init(device_no=0):
    # find an opencl device (preferably a GPU) in one of the available platforms
    done = False
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.GPU)
        if (len(devices) > 0) and ('NVIDIA' in p.name):
            done = True
            break

    if not done :
        for p in cl.get_platforms():
            devices = p.get_devices(cl.device_type.GPU)
            if (len(devices) > 0) :
                break
        
    if len(devices) == 0 :
        for p in cl.get_platforms():
            devices = p.get_devices()
            if len(devices) > 0:
                break
    
    print('number of devices:', len(devices), file=sys.stderr)
    print('devices:', devices, file=sys.stderr)
    sys.stdout.flush()

    device = devices[device_no % len(devices)]
    
    context = cl.Context(devices)
    
    # one queue for each device (maybe make 2 per device later)
    queue  = cl.CommandQueue(context, device)
    # for testing
    #queues  = queues + [cl.CommandQueue(context, device) for device in devices]
    return {'context': context, 'queue': queue, 'device': device}


def opencl_init_cpu(rank=0):
    """find an opencl device (preferably a GPU) if available"""
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.CPU)
        if (len(devices) > 0):
            break

    print('number of devices:', len(devices), file=sys.stderr)
    print('devices:', devices, file=sys.stderr)
    sys.stdout.flush()

    context = cl.Context(devices)
    # queue   = cl.CommandQueue(
    #    context,
    #    properties = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
    # )
    queue = cl.CommandQueue(context)
    return {'context': context, 'queue': queue}


# these are much faster than pyopencl's packaged routines for some reason
def to_gpu(ar, ar_cl = None, queue = None, dtype = None):
    if dtype is None :
        # prefer single precision
        if ar.dtype == np.float64 :
            dtype = np.float32
        elif ar.dtype == np.int64 :
            dtype = np.int32
        else :
            dtype = ar.dtype
    
    if ar_cl is None :
        ar_cl    = cl.array.empty(queue, ar.shape, dtype = dtype)
    
    cl.enqueue_copy(queue, ar_cl.data, np.ascontiguousarray(ar.astype(dtype)))
    return ar_cl

def to_cpu(ar_cl, ar = None, queue = None, dtype = None):
    if dtype is None :
        dtype = ar_cl.dtype
    
    if ar is None :
        ar = np.empty(ar_cl.shape, dtype = dtype)
    
    cl.enqueue_copy(queue, ar, ar_cl.data)
    return ar

def to_gpu_2D_image_stack(ar, queue = None, context = None):
    # copy I as an opencl "image" for bilinear sampling
    shape        = ar.shape
    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    flags        = cl.mem_flags.READ_ONLY
    I_cl         = cl.Image(context, flags, image_format, 
                            shape = shape[::-1], is_array = True)
    
    cl.enqueue_copy(queue, dest = I_cl, src = ar, 
                    origin = (0, 0, 0), region = shape[::-1])
    return I_cl

def to_gpu_2D_image(ar, queue = None, context = None):
    # copy I as an opencl "image" for bilinear sampling
    shape        = ar.shape
    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    flags        = cl.mem_flags.READ_ONLY
    I_cl         = cl.Image(context, flags, image_format, 
                            shape = shape[::-1], is_array = False)
    
    cl.enqueue_copy(queue, dest = I_cl, src = np.ascontiguousarray(ar.T.astype(np.float32)), 
                    origin = (0, 0), region = shape[::-1])
    return I_cl

def to_gpu_3D_image(ar, queue = None, context = None):
    # copy I as an opencl "image" for trilinear sampling
    shape        = ar.shape
    image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    flags        = cl.mem_flags.READ_ONLY
    I_cl         = cl.Image(context, flags, image_format, shape=ar.shape[::-1])
    cl.enqueue_copy(queue, I_cl, np.ascontiguousarray(ar.T.astype(np.float32)), is_blocking=True, origin=(0, 0, 0), region=ar.shape[::-1])
    return I_cl


class Solve_axbc_cl():
    
    def __init__(self, fill_value = 0., ftol = 1e-2, xtol = 1e-3, maxiters = 1000, debug = False, queue = None, context = None):
        
        self.fill_value = np.float64(fill_value)
        self.ftol       = np.float64(ftol)
        self.xtol       = np.float64(xtol)
        self.maxiters   = np.int64(maxiters)
        
        self.cl_code = cl.Program(context, self.code()).build()
        self.queue   = queue
        self.flag    = np.empty(1, dtype = np.int64)

        self.debug = debug

        self.flags = {
            0: f'zero length input returning {fill_value}',
            1: f'negative c-value returning {fill_value}',
            2: 'success',
            3: 'fmax is less than 0 returning 0',
            4: f'Warning maximum iterations exceeded!',
            5: 'analytic solution for N=1',
            6: 'analytic solution for N=2',
            7: 'all b terms are zero setting x to xmax = sum(a) / c'
        }
    
    def solve(self, a, b, c, out, d):
        event = self.cl_code.solve(self.queue, (1,), (1,),
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
        
        if self.debug :
            self.queue.finish()
            if out[d] > 10 :
                print(f'{d=} {out[d]=} {self.flags[self.flag[0]]}')
            """
            if self.flag[0] == 4:
                #print(a)
                #print(b)
                #print(c)
                xmax = 
                print(np.sum(a) / c)
                print(np.sum(a/b) - c)
            """
            sys.stdout.flush()
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
            flag[0] = 0;          // print(f'zero length input returning {fill_value}')
            return;
        }
            
        if (c <= 0.) {
            out[d]  = fill_value;
            flag[0] = 1;          // print(f'negative c-value {c} returning {fill_value}')
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
            flag[0] = 7;        // 'all b terms are zero setting x to xmax = sum(a) / c'
            return;
            
        } else if (bzeros > 0) {
            
            xmin = amax / c ;
        
        } else {

            for (i=0; i<I; i++) 
                fmax += a[i] / b[i] ;
            fmax -= c;
            
            if (fmax <= 0) {
                flag[0] = 3; // print(f'fmax {fmax} is less than 0 returning 0')
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
            
            if ((fabs(step) < (xtol * x)) && (fabs(fn) < (ftol * (fmax-fmin)))) {
                out[d]  = xmax ;
                flag[0] = 2;
                return ;
            }
        }
        
        out[d]  = x;
        flag[0] = 4; // print(f'Warning maximum iterations exceeded! {x=} {xmin=} {xmax=} {fmax=} {fn=} {fpn=}') 
        }
        """

