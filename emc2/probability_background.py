from emc2 import utils

import numpy as np
from tqdm import tqdm
import logging

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


def calc_logR(wsums_r, F_dri, likelihood='basic'):
    D, R, I = F_dri.shape

    logR_dr = np.zeros((D, R), dtype=float)

    d_chunk_size = 64
    r_chunk_size = 64

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
