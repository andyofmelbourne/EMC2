import pyopencl as cl
import pyopencl.array
import pyclblast
from emc2 import utils
from emc2 import utils_cl
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# I don't think this is safe
def calculate_wsums_r(C_i, W_ri, r00, r11, r_chunk_size=1024):
    # calculate tomogram sums
    R = r11-r00
    assert (r00 >= 0)
    assert (r11 <= W_ri.shape[0])
    wsums_r = np.zeros(R, dtype=float)
    W_ri.cpu = True

    r_iter = tqdm(
        utils.chunker(r_chunk_size, R, offset=r00),
        desc='calculating tomogram sums',
        disable=False
    )

    for r0, r1, dr in r_iter:
        W = W_ri[r0:r1, :]
        wsums_r[r0-r00:r1-r00] = np.sum(C_i * W, axis=-1)

    return wsums_r


def calculate_K_dot_W_gpu(W_ri, K_di, r_chunk_size=2048, d_chunk_size=2048):
    D, I = K_di.shape
    R = W_ri.shape[0]

    r_chunk_size = min(r_chunk_size, R)
    d_chunk_size = min(d_chunk_size, D)

    P_dr = np.zeros((D, R), dtype=float)
    W_ri.set_log(True)
    W_ri.cpu = False
    queue = W_ri.queue

    d_iter = tqdm(
        utils.chunker(d_chunk_size, D),
        desc='calculating dot product K . W'
    )

    P_dr_cl = cl.array.empty(
        queue,
        (d_chunk_size, r_chunk_size),
        dtype=np.float32
    )

    P_dr_ch = np.empty((d_chunk_size, r_chunk_size), dtype=np.float32)

    for d0, d1, dd in d_iter:
        K = K_di[d0:d1, :]
        K_cl = utils_cl.to_gpu(K, queue=queue, dtype=np.float32)

        r_iter = tqdm(
            utils.chunker(r_chunk_size, R),
            desc='looping over r',
            leave=False
        )

        for r0, r1, dr in r_iter:
            # non-blocking
            W_cl = W_ri[r0:r1, :]

            # non-blocking
            pyclblast.gemm(
                queue, dd, dr, I, K_cl, W_cl, P_dr_cl, I,
                I, r_chunk_size, a_transp=False, b_transp=True
            )

            # blocking
            cl.enqueue_copy(queue, P_dr_ch, P_dr_cl.data)
            P_dr[d0:d1, r0:r1] = P_dr_ch[:d1-d0, :r1-r0]

    W_ri.set_log(False)
    return P_dr


def calculate_K_dot_W(W_ri, K_di, r_chunk_size=1024, d_chunk_size=256):
    D, I = K_di.shape
    R, _ = W_ri.shape
    P_dr = np.zeros((D, R), dtype=float)
    W_ri.set_log(True)

    def dot(P_dr, K, W_cl, d0, d1, r0, r1):
        W = W_cl.get()
        P_dr[d0:d1, r0:r1] = np.dot(K[:d1-d0], W[:r1-r0].T)

    d_iter = tqdm(
        utils.chunker(d_chunk_size, D),
        desc='calculating dot product K . W'
    )

    r_iter = tqdm(
        utils.chunker(r_chunk_size, R),
        desc='looping over r',
        leave=False
    )

    max_depth = 4

    executor = ThreadPoolExecutor()
    dot_events = []

    for d0, d1, dd in d_iter:
        K = K_di[d0:d1, :]
        for r0, r1, dr in r_iter:
            # non-blocking
            W_cl = W_ri[r0:r1, :]

            # non-blocking
            dot_events.append(
                executor.submit(dot, P_dr, K, W_cl, d0, d1, r0, r1)
            )

            if len(dot_events) == max_depth or r1 == R:
                executor.shutdown()
                dot_events = []
                executor = ThreadPoolExecutor()

    W_ri.set_log(False)
    return P_dr


def calc_logR(K_di, W_ri, C_i, w_d, wsums_r, **config):
    D, I = K_di.shape
    R = W_ri.shape[0]

    # calculate KW_dr = sum_i K_di W_ri
    # P_dr1 = calculate_K_dot_W(W_ri, K_di)
    P_dr = calculate_K_dot_W_gpu(W_ri, K_di)

    # get cpu context and queue
    cl_cpu_stuff = utils_cl.opencl_init_cpu(0)
    queue_cpu = cl_cpu_stuff['queue']

    logger.debug('\nCompiling cpu code for offset and normalisation of P_dr')
    cl_cpu_code = cl.Program(
        cl_cpu_stuff['context'],
        code.format(rotations=R)
    ).build()

    # offset logR (just use cpu)
    # logR_dr = KlogW_dr - K_d log(wsums_r)
    # logR_dr = KlogW_dr - w_d wsums_r
    # logR_dr = KlogW_dr - wsums_r

    for i in tqdm(range(1), desc='applying offset to logR'):
        # logR_dr = \sum_i K_di logW_ri - K_d log(sum_i C_i W_ri)
        # -------------------------------------------------------
        if (
            config['likelihood'] == 'Poisson_fluence_free'
            and config['frame_model'] == 'basic'
        ):
            event = cl_cpu_code.logR_Klog_wsums(
                queue_cpu,
                (D, R),
                None,
                cl.SVM(P_dr),
                cl.SVM(K_di.photon_sums),
                cl.SVM(wsums_r)
            )

        # logR_dr = \sum_i K_di logW_ri - sum_i C_i W_ri
        # -------------------------------------------------------
        elif (
            config['likelihood'] == 'Poisson' and
            config['frame_model'] == 'basic'
        ):
            event = cl_cpu_code.logR_wsums(
                queue_cpu,
                (D, R),
                None,
                cl.SVM(P_dr),
                cl.SVM(wsums_r)
            )

        # logR_dr = \sum_i K_di logW_ri - w_d sum_i C_i W_ri
        # -------------------------------------------------------
        elif (
            config['likelihood'] == 'Poisson' and
            config['frame_model'] == 'fluence'
        ):
            event = cl_cpu_code.logR_w_wsums(
                queue_cpu,
                (D, R),
                None,
                cl.SVM(P_dr),
                cl.SVM(w_d),
                cl.SVM(wsums_r)
            )

    event.wait()
    queue_cpu.finish()

    return P_dr, wsums_r


code = """
    // logR_dr = KlogW_dr - K_d log(wsums_r)
    // optimised for gpu with one worker per d and r
    __kernel void logR_Klog_wsums (
        global double *KlogW_dr,
        global long   *K_d,
        global double *wsums_r
    ) {{
        int d = get_global_id(0);
        int r = get_global_id(1);
        int R = get_global_size(1);

        KlogW_dr[d * R + r] -= (double)K_d[d] * log(wsums_r[r]);
    }}

    // logR_dr = KlogW_dr - w_d wsums_r
    __kernel void logR_w_wsums (
        global double *KlogW_dr,
        global double *w_d,
        global double *wsums_r
    ) {{
        int d = get_global_id(0);
        int r = get_global_id(1);
        int R = get_global_size(1);

        KlogW_dr[d * R + r] -= w_d[d] * wsums_r[r];
    }}

    // logR_dr = KlogW_dr - wsums_r
    __kernel void logR_wsums (
        global double *KlogW_dr,
        global double *wsums_r
    ) {{
        int d = get_global_id(0);
        int r = get_global_id(1);
        int R = get_global_size(1);

        KlogW_dr[d * R + r] -= wsums_r[r];
    }}

"""
