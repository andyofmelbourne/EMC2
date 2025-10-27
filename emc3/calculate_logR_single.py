"""
Here are a collection of scripts for actually calculating things

Calculate likelihood per pixel

"""

import numpy as np
import time
from pathlib import Path
import sys
import h5py

from . import utils_cl
from . import utils
from .tomograms import Tomograms, Tomograms_cl
from .likelihood import Likelihood
from .update_models import gpu_dot

from .calculations import *


def calculate_logR_class_0(L, cl, r0, r1, d_chunk_size=1024, r_chunk_size=1024):
    """
    all in memory cpu + opencl process
    """
    R0 = L.tomo.shape[0]
    D = L.K_di.shape[0]

    R = r1-r0

    tomos_cl = Tomograms_cl(L.tomo, cl['context'], cl['queue'])

    r_chunk_size = min(r_chunk_size, R)

    t0 = time.time()
    L.wsums_r = tomos_cl.calculate_wsums(chunksize=r_chunk_size, r0=r0, r1=r1)

    tomos_cl.load_buffers(r_chunk_size=r_chunk_size)

    logR_dr = np.zeros((D, R), dtype=np.float32)

    t = 0

    for r00, r11, dr in utils.chunker(r_chunk_size, R):
        # calculate tomograms
        W_ri = tomos_cl.calculate_tomogram(r0+r00, r0+r11, log=True, cpu=True)
        # print('tomo time:', time.time() - t0)

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

            logR_dr[d0:d1, r00:r11] += logR_dr_dev.get()
            # print('K . W time:', time.time() - t0)

            t += time.time() - t0

    # offset
    L.offset(logR_dr, rrange=[r0, r1])

    return logR_dr, L.wsums_r, t


def calculate_logR_class_0_c(c, cl, r0=None, r1=None):
    if not c['update_logR']:
        return

    R = c['mapper'].shape[1]

    if r0 is None:
        r0 = 0

    if r1 is None:
        r1 = self.R

    r1 = min(R, r1)
    r0 = max(0, r0)

    c['mapper'].load_coords(c['P_data'].mask)

    tomos = Tomograms(
            c['mapper'],
            c['model'],
            c['P_data'].C_i,
            c['fluence'])

    L = Likelihood(tomos, c['P_data'], **c)

    return calculate_logR_class_0(L, cl, r0, r1)

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
    assert(len(cids) == 1)

    c = config['classes'][0]
    class_id = c['class_id']

    R = c['mapper'].shape[1]
    D = c['P_data'].shape[0]

    if not c['update_logR']:
        return None

    devices = utils_cl.get_devices(device_type='gpu')

    # number parallel processes
    nproc = p_per_device * len(devices)

    # split calculation by r-chunks
    # str = '0-10000 10000-20000 20000-20323'
    r0_r1_str = []
    for s in np.array_split(np.arange(0, R), nproc):
        r0_r1_str.append('-'.join([str(s[0]), str(s[-1]+1)]))
    r0_r1_str = ' '.join(r0_r1_str)


    cmd = f"time parallel --halt now,fail=1 python -m emc3.calculate_logR_single {config_file} {{}} {{%}} ::: {r0_r1_str}"
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

    # gather chunks
    # load chunks
    fnams = Path('./').glob(f'class_logR_chunk_{class_id}_*.h5')

    with h5py.File(c['logR_file'], 'w') as f:
        f.create_dataset('logR_dr', shape=(D, R), dtype=float)
        for fnam in fnams:
            with h5py.File(str(fnam)) as g:
                logR_dr = g['logR_dr'][()]
                r0 = g['r0'][()]
                r1 = g['r1'][()]

            f['logR_dr'][:, r0:r1] = logR_dr

            # delete
            fnam.unlink()

    fnams = Path('./').glob(f'P_wsums_chunk_{class_id}_*.h5')

    with h5py.File(c['P_wsums_file'], 'w') as f:
        f.create_dataset('wsums_r', shape=(R,), dtype=float)
        for fnam in fnams:
            with h5py.File(str(fnam)) as g:
                wsums_r = g['wsums_r'][()]
                r0 = g['r0'][()]
                r1 = g['r1'][()]

            f['wsums_r'][r0:r1] = wsums_r

            # delete
            fnam.unlink()

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
    r0, r1 = [int(c) for c in sys.argv[2].split('-')]
    device = int(sys.argv[3])

    config = pickle.load(open(config_fnam, 'rb'))
    wd = config['working_directory']

    c = config['classes'][0]
    class_id = c['class_id']

    dot_time = 0

    if not c['update_logR']:
        sys.exit()

    # load model
    with h5py.File(c['model_file']) as f:
        c['model'].data = f['data'][()]

    # load data
    c['P_data'].load_from_file()

    cl = utils_cl.opencl_init(device_no=device)

    logR_dr, wsums_r, t = calculate_logR_class_0_c(c, cl, r0=r0, r1=r1)
    dot_time += t

    # save
    fnam = f'class_logR_chunk_{class_id}_{r0}_{r1}.h5'
    with h5py.File(fnam, 'w') as f:
        f['logR_dr'] = logR_dr
        f['r0'] = r0
        f['r1'] = r1

    fnam = f'P_wsums_chunk_{class_id}_{r0}_{r1}.h5'
    with h5py.File(fnam, 'w') as f:
        f['wsums_r'] = wsums_r[r0:r1]
        f['r0'] = r0
        f['r1'] = r1

    print('K . W time:', dot_time)

