"""
Here are a collection of scripts for actually calculating things
"""

import numpy as np
import time

from . import utils_cl
from .tomograms import Tomograms, Tomograms_cl
from .likelihood import Likelihood

def calculate_logR_cl(config):

    """
    basic all in memory cpu process
    """
    for c in config['classes']:
        if not c['update_logR']:
            continue

        c['mapper'].load_coords(c['P_data'].mask)

        tomos = Tomograms(
                c['mapper'],
                c['model'],
                c['P_data'].C_i,
                c['fluence'])

        L = Likelihood(tomos, c['P_data'], **c)

        c['logR_dr'] = calculate_logR_class_0(L)


def calculate_logR_class_0(L):
    """
    all in memory cpu + opencl process
    """
    cl = utils_cl.opencl_init(device_no=0)
    tomos_cl = Tomograms_cl(L.tomo, cl['context'], cl['queue'])

    t0 = time.time()
    L.wsums_r = tomos_cl.calculate_wsums()

    R = L.tomo.shape[0]
    tomos_cl.load_buffers(r_chunk_size=R)

    # calculate tomograms
    W_ri = tomos_cl.calculate_tomogram(0, R, log=True, cpu=True)
    # print('tomo time:', time.time() - t0)

    # get data
    K_di = L.K_di[:]

    # calculate dot product
    t0 = time.time()
    logR_dr = np.dot(K_di[:], W_ri.T)
    # print('dot time:', time.time() - t0)

    # offset
    L.offset(logR_dr)

    return logR_dr



def update_I_cl(config):
    for c in config['classes']:
        if not c['update_model']:
            continue

        c['model'].data = update_I_class_0(c)

    models = [c['model'].data for c in config['classes']]
    dq = config['classes'][0]['model'].dq
    utils.save_model_slices(
            models,
            dq,
            config['working_directory']
            )


def update_I_class_0(c):
    P_dr = c['P_dr']
    C_i = c['data'].C_i
    w_d = c['fluence']
    likelihood = c['likelihood']
    frame_model = c['frame_model']
    ksums = c['data'].data_sum
    mapper = c['mapper']
    D, R = P_dr.shape
    I = c['data'].shape[1]
    N = c['model'].shape[0]

    N_n = np.zeros(c['model'].data.size, dtype=float)
    D_n = np.zeros(c['model'].data.size, dtype=float)

    assert (D == c['data'].shape[0])

    mapper.load_coords(c['data'].mask)

    tomos = Tomograms(
            mapper,
            c['model'],
            c['data'].C_i,
            c['fluence'])

    cl = utils_cl.opencl_init(device_no=0)

    tomos_cl = Tomograms_cl(tomos, cl['context'], cl['queue'])

    wsums_r = tomos_cl.calculate_wsums()

    R = tomos.shape[0]
    tomos_cl.load_buffers(r_chunk_size=R)

    t0 = time.time()
    N_ri = np.dot(P_dr.T, c['data'][:])
    # print('dot time', time.time() - t0)

    if (
        likelihood == 'Poisson' and
        frame_model == 'basic'
    ):
        D_ri = C_i[None, :] * \
                np.sum(P_dr, axis=0)[:, None]

    elif (
        likelihood == 'Poisson'
        and frame_model == 'fluence'
    ):
        D_ri = C_i[None, :] * np.dot(w_d, P_dr)[:, None]

    elif (
        likelihood == 'Poisson_fluence_free'
        and frame_model == 'basic'
    ):
        N_ri *= wsums_r[:, None]
        D_ri = C_i[None, :] * \
            np.dot(ksums, P_dr)[:, None]

    else:
        raise ValueError(f'could not parse likelihood {class_c.likelihood}'
                         f'and frame_model {class_c.frame_model}')

    t0 = time.time()
    n_sri = np.empty(mapper.shape + (I,), dtype=int)

    for s in range(mapper.shape[0]):
        for r in range(mapper.shape[1]):
            n_sri[s, r] = mapper.calculate_mapping_ravel(r, s)
    # print('mapper time', time.time() - t0)

    # now merge N_ri and D_ri to I-space
    if c['maximise'] == 'W':
        D_ri[D_ri == 0] = 1.
        N_ri /= D_ri
        D_ri[:] = 1.

    for s in tqdm(range(n_sri.shape[0]), leave=False):
        for r in range(n_sri.shape[1]):
            N_n += np.bincount(
                n_sri[s, r],
                N_ri[r],
                minlength=c['model'].data.size
            )

            D_n += np.bincount(
                n_sri[s, r],
                D_ri[r],
                minlength=c['model'].data.size
            )

    # apply symmetry
    i0 = N // 2

    sym = symmetry.Symmetry(
            c['model'].i0,
            c['model'].shape,
            c['model'].symmetry
    )

    N_n = sym.apply_symmetry(
        N_n.reshape(c['model'].shape),
    )

    D_n = sym.apply_symmetry(
        D_n.reshape(c['model'].shape),
    )

    # I = N / D
    D_n[D_n == 0] = 1.
    N_n /= D_n

    return N_n
