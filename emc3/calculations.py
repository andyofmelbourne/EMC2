"""
Here are a collection of scripts for actually calculating things

Calculate likelihood per pixel

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
    cl = utils_cl.opencl_init(device_no=0)
    dot_time = 0
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

        c['logR_dr'], t = calculate_logR_class_0(L, cl)
        dot_time += t

    print('K . W time:', dot_time)


def calculate_logR_class_0(L, cl):
    """
    all in memory cpu + opencl process
    """
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
    t = time.time() - t0
    # print('K . W time:', time.time() - t0)

    # offset
    L.offset(logR_dr)

    return logR_dr, t


def calculate_logR_class_0_c(c):
    if not c['update_logR']:
        return

    c['mapper'].load_coords(c['P_data'].mask)

    tomos = Tomograms(
            c['mapper'],
            c['model'],
            c['P_data'].C_i,
            c['fluence'])

    L = Likelihood(tomos, c['P_data'], **c)

    c['logR_dr'] = calculate_logR_class_0(L)


if __name__ == '__main__':
    import sys, pickle
    config_fnam = sys.argv[1]
    class_id = int(sys.argv[2])

    config = pickle.load(open(config_fnam, 'rb'))

    c = config['classes'][class_id]
    c['P_data'].load_from_file()

    logR_dr = calculate_logR_class_0_c(c)

