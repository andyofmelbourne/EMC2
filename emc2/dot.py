import numpy as np
import tqdm
import time

import pyopencl as cl
import pyopencl.array 

gpu_precision = np.float32

# find an opencl device (preferably a GPU) in one of the available platforms
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
    if len(devices) > 0:
        break
    
if len(devices) == 0 :
    for p in cl.get_platforms():
        devices = p.get_devices()
        if len(devices) > 0:
            break

print(devices)
context = cl.Context(devices)
queue   = cl.CommandQueue(context)

import pyclblast


N = 2048
A = np.random.random((N, N)).astype(np.float32)
B = np.random.random((N, N)).astype(np.float32)
C = np.empty((N, N), dtype=np.float32)


A_cl = cl.array.to_device(queue, A)
B_cl = cl.array.to_device(queue, B)
C_cl = cl.array.to_device(queue, C)

t0 = time.time()
T = 10
for i in range(T):
    C[:] = np.dot(A_cl.get().T, B_cl.get())
t1 = time.time()
print(f'iterations per second: {1/(t1-t0)/T}')
#A *= 1e-40

t0 = time.time()
T = 10
for i in range(T):
    pyclblast.gemm(queue, N, N, N, A_cl, B_cl, C_cl, N, N, N, a_transp = True, b_transp = False)
queue.finish()
t1 = time.time()
print(f'iterations per second gemm: {1/(t1-t0)/T}')

print(100 * np.max((C - C_cl.get())/ C))
#assert(np.allclose(c_cl.get(), A.dot(b)))
