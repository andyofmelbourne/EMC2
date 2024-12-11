import sys
import pyopencl as cl
import pyopencl.array 
import numpy as np

def opencl_init(rank):
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
    
    print('number of devices:', len(devices))
    print(rank, 'my device:', devices[rank % len(devices)])
    sys.stdout.flush()
    
    context = cl.Context([devices[rank % len(devices)]])
    queue   = cl.CommandQueue(context)
    return {'context': context, 'queue': queue}

def opencl_init_cpu(rank):
    # find an opencl device (preferably a GPU) in one of the available platforms
    done = False
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.CPU)
        if (len(devices) > 0) :
            break
    
    print('number of devices:', len(devices))
    print(rank, 'my device:', devices[rank % len(devices)])
    sys.stdout.flush()
    
    context = cl.Context([devices[rank % len(devices)]])
    queue   = cl.CommandQueue(context)
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
