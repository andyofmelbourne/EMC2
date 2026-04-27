"""
Here are a collection of scripts for actually calculating things

Calculate likelihood per pixel

"""

import numpy as np
import time
from pathlib import Path

from . import utils_cl
from . import utils
from .tomograms import Tomograms, Tomograms_cl
from .likelihood import Likelihood
from .update_models import gpu_dot

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

        c['logR_dr'], wsums_r, t = calculate_logR_class_0(L, cl)
        dot_time += t

    print('K . W time:', dot_time)


def calculate_logR_class_0(L, cl, d_chunk_size=1024, r_chunk_size=1024):
    """
    all in memory cpu + opencl process
    """
    R = L.tomo.shape[0]
    D = L.K_di.shape[0]

    tomos_cl = Tomograms_cl(L.tomo, cl['context'], cl['queue'])

    r_chunk_size = min(r_chunk_size, R)

    t0 = time.time()
    L.wsums_r = tomos_cl.calculate_wsums(chunksize=r_chunk_size)

    tomos_cl.load_buffers(r_chunk_size=r_chunk_size)

    logR_dr = np.zeros((D, R), dtype=np.float32)

    for r0, r1, dr in utils.chunker(r_chunk_size, R):
        # calculate tomograms
        W_ri = tomos_cl.calculate_tomogram(r0, r1, log=True, cpu=True)

        # test
        """
        R0 = np.dot(L.K_di[:], W_ri.T) - L.K_di.data_sum[:, None] * np.log(L.wsums_r[r0:r1])[None, :]
        print()
        wsums_r = L.wsums_r
        print(f'{np.min(W_ri)=} {np.max(W_ri)=} {np.mean(W_ri)=}')
        print(f'{np.min(wsums_r[r0:r1])=} {np.max(wsums_r[r0:r1])=} {np.mean(wsums_r[r0:r1])=}')
        print(f'{np.min(np.log(wsums_r[r0:r1]))=} {np.max(np.log(wsums_r[r0:r1]))=} {np.mean(np.log(wsums_r[r0:r1]))=}')
        W_ri -= np.log(L.wsums_r[r0:r1])[:, None]
        print(f'{np.min(W_ri)=} {np.max(W_ri)=} {np.mean(W_ri)=}')
        R1 = np.dot(L.K_di[:], W_ri.T)
        print(f'{np.linalg.norm(R0)=} {np.linalg.norm(R1)=} {np.linalg.norm(R0-R1)=}')
        print()
        # print('tomo time:', time.time() - t0)
        """

        for d0, d1, dr in utils.chunker(d_chunk_size, D):
            # get data
            K_di = L.K_di[d0:d1].astype(np.float32)

            # calculate dot product
            t0 = time.time()
            # logR_dr = np.dot(K_di[:], W_ri.T)

            logR_dr_dev, evt = gpu_dot(
                K_di,
                W_ri,
                cl['queue'], a_transp=False, b_transp=True)
            evt.wait()

            logR_dr[d0:d1, r0:r1] += logR_dr_dev.get()
            # print('K . W time:', time.time() - t0)

    t = time.time() - t0

    # offset
    L.offset(logR_dr)

    return logR_dr, L.wsums_r, t


def calculate_logR_class_0_c(c, cl):
    if not c['update_logR']:
        return

    c['mapper'].load_coords(c['P_data'].mask)

    tomos = Tomograms(
            c['mapper'],
            c['model'],
            c['P_data'].C_i,
            c['fluence'])

    L = Likelihood(tomos, c['P_data'], **c)

    return calculate_logR_class_0(L, cl)

def calculate_logR_subprocess(config_file, config, p_per_device=2, cids=None):
    """
    calculate logR in a separate process
    2 processes for each device
    write logR to file, return file names
    """
    import subprocess, sys
    from . import utils_cl, utils

    # class ids
    if cids is None:
        cids = [i for i in range(len(config['classes'])) if config['classes'][i]['update_logR']]
    else:
        cids = [ci for ci in cids if config['classes'][ci]['update_logR']]

    devices = utils_cl.get_devices(device_type='gpu')

    # number parallel processes
    nproc = p_per_device * len(devices)

    # make cids_str = '0,1,2 3,4,5 6,7,8 9'
    cids_str = []
    for s in np.array_split(cids, nproc):
        cids_str.append(','.join(s.astype(str)))
    cids_str = ' '.join(cids_str)

    cmd = f"time parallel --halt now,fail=1 python -m emc3.calculations {config_file} {{}} {{%}} ::: {cids_str}"
    print(cmd)
    p = subprocess.Popen(
        cmd,
        text=True,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
        )
    p.wait()

    if p.returncode != 0:
        raise ValueError('something went wrong with call')

    return True


"""
load config
calculate logR for selected classes
    this helps to reduce io and initialisation overhead
write to file

use platform and device specified on command line for multi gpu
"""
if __name__ == '__main__':
    import sys, pickle, h5py

    config_fnam = sys.argv[1]
    # class_id = int(sys.argv[2])
    class_ids = [int(c) for c in sys.argv[2].split(',')]
    device = int(sys.argv[3])

    config = pickle.load(open(config_fnam, 'rb'))
    wd = config['working_directory']

    # load fluence
    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]

    dot_time = 0
    for class_id in class_ids:
        c = config['classes'][class_id]

        if not c['update_logR']:
            continue

        # load model
        with h5py.File(c['model_file']) as f:
            c['model'].data = f['data'][()]

        # set fluence
        c['fluence'] = w_d

        # load data
        c['P_data'].load_from_file()

        cl = utils_cl.opencl_init(device_no=device)

        logR_dr, wsums_r, t = calculate_logR_class_0_c(c, cl)
        dot_time += t

        # save
        with h5py.File(c['logR_file'], 'w') as f:
            f['logR_dr'] = logR_dr

        with h5py.File(c['P_wsums_file'], 'w') as f:
            f['wsums_r'] = wsums_r

    print('K . W time:', dot_time)
