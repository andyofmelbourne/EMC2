"""
Here are a collection of scripts for actually calculating things

Calculate likelihood per pixel

"""

import numpy as np
import time
from pathlib import Path

from . import utils_cl
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
    # logR_dr = np.dot(K_di[:], W_ri.T)

    logR_dr_dev, evt = gpu_dot(
        K_di[:].astype(np.float32),
        W_ri,
        cl['queue'], a_transp=False, b_transp=True)
    evt.wait()
    t = time.time() - t0

    logR_dr = logR_dr_dev.get()
    # print('K . W time:', time.time() - t0)

    # offset
    L.offset(logR_dr)

    return logR_dr, t


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

def calculate_logR_subprocess(config_file, config, p_per_device=2):
    """
    calculate logR in a separate process
    2 processes for each device
    write logR to file, return file names
    """
    import subprocess, sys
    from . import utils_cl, utils

    # class ids
    cids = [i for i in range(len(config['classes'])) if config['classes'][i]['update_logR']]

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

    dot_time = 0
    for class_id in class_ids:
        c = config['classes'][class_id]

        if not c['update_logR']:
            continue

        # load model
        with h5py.File(c['model_file']) as f:
            c['model'].data = f['data'][()]

        # load data
        c['P_data'].load_from_file()

        cl = utils_cl.opencl_init(device_no=device)

        logR_dr, t = calculate_logR_class_0_c(c, cl)
        dot_time += t

        # save
        with h5py.File(c['logR_file'], 'w') as f:
            f['logR_dr'] = logR_dr

    print('K . W time:', dot_time)
