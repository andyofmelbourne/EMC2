from emc2 import utils

import numpy as np
from tqdm import tqdm
import logging

# testing
import pyopencl as cl
import pyopencl.array
import pyclblast
import sys
from . import utils_cl

mf = cl.mem_flags

logger = logging.getLogger(__name__)

r"""
F_dri = C_i w_d W_ri + B_di
F_dr  = sum_i (C_i w_d W_ri + B_di)
F_dr  = w_d sum_i (C_i W_ri) + sum_i (B_di)
F_dr  = w_d Wsums_r + Bsums_d

general:
logR_dr = \sum_i K_di logF_dri - F_dr

over fluence:
logR_dr = \sum_i K_di logF_dri - K_d logF_dr
"""

code = """
// r_block_size must equal R
__kernel void calc_logR (
    global float *W_ri,
    global float *B_di,
    global float *K_di,
    global float *logR_dr,
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
        K = K_di[d * I + i];
        if (K > 0.) {{
            B = B_di[d * I + i];
            for (r = 0; r < R; r++){{
                F = W_ri[I * r + i] + B;
                if (F > 0.) {{
                    F = log(F);
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


def calc_logR(wsums_r, w_d, K_di, B_di, C_i, W_ri_getter, likelihood='basic'):
    """
    Try:
        calculate W_ri
        calculate B'_di = B_di / (w_d C_i)

        logF_dri = log(w_d C_i W_ri + B_di)
                 = log(W_ri + B'_di) + log(w_d) + log(C_i)
                 = logF'_dri + log(w_d) + log(C_i)

        logR_dr = sum_i K_di logF_dri
                = sum_i K_di logF'_dri
                + log(w_d) K_d
                + sum_i K_di log(C_i)

    calculate in blocks:
        loop d block:
            load B'_di_cl
            load K_di_cl

            loop r block:
                load W_ri_cl

                loop d
                    loop r chunk
                        calc logF'_dri
                        calc logR_r = logF'_dri
    """
    D, I = K_di.shape
    R = wsums_r.shape[0]

    logR_dr = np.zeros((D, R), dtype=float)

    d_block_size = min(256, D)
    r_block_size = min(64, R)

    # calculate logR offset
    # ---------------------
    desc1 = 'calculating sum_i F_dri = w_d Wsums_r + Bsums_d'
    desc2 = 'calculating K_d logF_dr'
    for _ in tqdm(range(1), desc=desc1):
        F_dr = np.outer(w_d, wsums_r) \
            + B_di.background_sums[:, None]

        if likelihood == 'Poisson_fluence_free':
            for d in tqdm(range(D), desc=desc2, leave=False):
                F_dr[d] = K_di.photon_sums[d] * np.log(F_dr[d])

    t = K_di.photon_sums[:D] * np.log(w_d)
    t += np.dot(K_di[:, :], np.log(C_i))
    F_dr -= t[:, None]
    # ---------------------

    # setup block iterations
    # ----------------------
    opencl_stuff = utils_cl.opencl_init_cpu()
    queue = opencl_stuff['queue']
    context = opencl_stuff['context']
    code_cl = cl.Program(
        context,
        code.format(r_block_size=r_block_size)
    ).build()

    d_block_iter = tqdm(
        utils.chunker(d_block_size, D),
        desc=f'calculating dot product K . logF over d-chunks {d_block_size}'
    )

    W_ri_getter.cpu = True

    logR_dr_buf = np.empty((d_block_size * r_block_size), dtype=np.float32)
    # ----------------------

    # start block iterations
    # ----------------------
    for d0, d1, dd in d_block_iter:
        B2_di = B_di[d0: d1, :] / w_d[d0: d1, None] / C_i
        K_di_block = K_di[d0: d1, :]

        B2_di = np.ascontiguousarray(B2_di.astype(np.float32))
        K_di_block = np.ascontiguousarray(K_di_block.astype(np.float32))

        r_block_iter = tqdm(
            utils.chunker(r_block_size, R),
            leave=False
        )

        for r0, r1, dr in r_block_iter:
            W_ri = np.ascontiguousarray(
                W_ri_getter[r0: r1, :].astype(np.float32)
            )

            code_cl.calc_logR(
                queue,
                (dd,),
                (1,),
                cl.SVM(W_ri),
                cl.SVM(B2_di),
                cl.SVM(K_di_block),
                cl.SVM(logR_dr_buf),
                np.int32(dr),
                np.int32(I)
            )
            queue.finish()
            logR_dr[d0: d1, r0: r1] = logR_dr_buf[:dd*dr].reshape((dd, dr))

    logR_dr -= F_dr

    return logR_dr


def calc_logR_old(wsums_r, F_dri, likelihood='basic'):
    D, R, I = F_dri.shape

    logR_dr = np.zeros((D, R), dtype=float)

    d_chunk_size = min(64, D)
    r_chunk_size = min(64, R)

    # see how long it takes to generate all KlogF - F
    # F_buf = np.empty((d_chunk_size, r_chunk_size, I), dtype=np.float32)
    # F_dri.update_rotations(np.arange(R))
    # for d0, d1, dd in d_iter:
    #     F_dri.update_frames(d0, d1)
    #     F_dri.K_logF_F(out = F_buf)

    desc1 = 'calculating sum_i F_dri = w_d Wsums_r + Bsums_d'
    desc2 = 'calculating K_d logF_dr'
    for _ in tqdm(range(1), desc=desc1):
        F_dr = np.outer(F_dri.w_d, wsums_r) \
            + F_dri.B_di.background_sums[:, None]

        if likelihood == 'Poisson_fluence_free':
            for d in tqdm(range(D), desc=desc2, leave=False):
                F_dr[d] = F_dri.K_di.photon_sums[d] * np.log(F_dr[d])

    d_iter = tqdm(
        utils.chunker(d_chunk_size, D),
        desc=f'calculating dot product K . logF over d-chunks {d_chunk_size}'
    )

    F_buf = np.empty((d_chunk_size * r_chunk_size * I), dtype=np.float32)
    for d0, d1, dd in d_iter:
        F_dri.update_frames(d0, d1)

        r_iter = tqdm(
            utils.chunker(r_chunk_size, R),
            desc=f'looping over r-chunks {r_chunk_size}', leave=False
        )

        for r0, r1, dr in r_iter:
            F_dri.update_rotations(np.arange(r0, r1))

            F_dri.K_logF_F(out=F_buf)

            shape = (dd, dr, I)
            size = dd * dr * I
            temp = F_buf[:size].reshape(shape)

            # logger.info(f'mean of sum_i frame : '
            #             f'{np.mean(np.sum(temp, axis=-1))}')

            logR_dr[d0:d1, r0:r1] = np.sum(temp, axis=-1)

    logR_dr -= F_dr

    return logR_dr


# sparse
def calc_logR_test(w_d, wsums_r, B_di, K_di, C_i, W_ri, likelihood='basic'):
    D, I = K_di.shape
    R = W_ri.shape[0]

    logR_dr = np.zeros((D, R), dtype=float)

    desc1 = 'calculating sum_i F_dri = w_d Wsums_r + Bsums_d'
    desc2 = 'calculating K_d logF_dr'
    for _ in tqdm(range(1), desc=desc1):
        F_dr = np.outer(w_d, wsums_r) \
            + B_di.background_sums[:, None]

        if likelihood == 'Poisson_fluence_free':
            for d in tqdm(range(D), desc=desc2, leave=False):
                F_dr[d] = K_di.photon_sums[d] * np.log(F_dr[d])

    d_iter = tqdm(
        range(D),
        desc='calculating dot product K . logF over frames'
    )

    W_ri.cpu = True
    for d in d_iter:
        Kd_i, pixels = K_di.sparse(d)
        Bd_i = B_di.sparse(d, pixels)
        Cd_i = C_i[pixels]

        for r in tqdm(range(R)):
            Wd_ri = W_ri[r:r+1, pixels]
            F_dri = w_d[d] * Cd_i * Wd_ri[0] + Bd_i

            m = F_dri > 0

            logR_dr[d, r] = np.sum(Kd_i[m] * np.log(F_dri[m])) - F_dr[d, r]

    return logR_dr


# cpu intensive
def calc_logR_test2(w_d, wsums_r, B_di, K_di, C_i, W_ri, likelihood='basic'):
    D, I = K_di.shape
    R = W_ri.shape[0]

    logR_dr = np.zeros((D, R), dtype=float)

    desc1 = 'calculating sum_i F_dri = w_d Wsums_r + Bsums_d'
    desc2 = 'calculating K_d logF_dr'
    for _ in tqdm(range(1), desc=desc1):
        F_dr = np.outer(w_d, wsums_r) \
            + B_di.background_sums[:, None]

        if likelihood == 'Poisson_fluence_free':
            for d in tqdm(range(D), desc=desc2, leave=False):
                F_dr[d] = K_di.photon_sums[d] * np.log(F_dr[d])

    r_iter = tqdm(
        range(R),
        desc='calculating dot product K . logF over rotations'
    )

    W_ri.cpu = True
    for r in r_iter:
        Wd_ri = W_ri[r:r+1, :][0]

        for d in tqdm(range(D), leave=False):
            Kd_i = K_di[d:d+1, :][0]
            Bd_i = B_di[d:d+1, :][0]

            F_dri = w_d[d] * C_i * Wd_ri + Bd_i

            m = F_dri > 0

            logR_dr[d, r] = np.sum(Kd_i[m] * np.log(F_dri[m])) - F_dr[d, r]

    return logR_dr
