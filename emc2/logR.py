import numpy as np
import math

def chunker(chunksize, size):
    D      = math.ceil(size/chunksize)
    dstart = np.arange(D) * chunksize
    dstop  = np.clip(dstart + chunksize, 0, size)
    dd     = dstop - dstart
    return zip(dstart, dstop, dd)

def reduce(ar, axis):
    return np.sum(ar, axis=axis)

def logR_cpu_nompi(
    mapper, 
    data_getter, 
    frame_chunks = 128, 
    rotation_chunks = 128, 
    pixel_chunks = 128,
    dtype = np.float32):
    """
    Calculate logR_dr = sum_i logR_dri
    """
    logR     = np.empty((mapper.D, mapper.R), dtype = dtype)
    logR_buf = np.empty((rotaion_chunks,), dtype = float)
    frames   = np.empty((frame_chunks, mapper.I), dtype = data_getter.dtype)
    
    for d0, d1, _ in chunker(frame_chunks, mapper.D):
        frames = data_getter.data(d0, d1)
        
        for r0, r1, dr in chunker(rotation_chunks, mapper.R):
            logR_buf[:dr] = 0
            
            for i0, i1, _ in chunker(pixel_chunks, mapper.I):
                logR_buf[:dr] += reduce(mapper.logR(frames[:, i0:i1], d0, d1, r0, r1, i0, i1), axis = 1)
             
            logR[d0: d1, r0: r1] = logR_buf[:dr]
    
    return logR
