"""
Here are a collection of scripts for actually calculating things

Calculate likelihood per pixel

"""

import numpy as np
import time
from pathlib import Path
import sys
import h5py
from tqdm import tqdm

from .. import utils_cl
from .. import utils
from .. import profiling
from ..tomograms import Tomograms, Tomograms_cl
from ..likelihood import Likelihood
from ..update_models import gpu_dot
from ..calculations import *
from ..data import BackCXI
from .frames import Frames, Frames_cl, Calc_logR, Calc_logR_back_fluence_free


@profiling.timed
def calculate_logR_class_0(K_di, frames, cl, r0, r1, d_chunk_size=64, r_chunk_size=256):
    """
    all in memory cpu + opencl process
    """
    D, R0, I = frames.shape

    R = r1-r0

    r_chunk_size = min(r_chunk_size, R)

    # calculate wsums_r
    tomos_cl = Tomograms_cl(frames.tomo, cl['context'], cl['queue'])
    wsums_r = tomos_cl.calculate_wsums(chunksize=r_chunk_size, r0=r0, r1=r1)
    wsums_r = wsums_r[r0:r1]

    # free memory (not sure if this works)
    del tomos_cl

    frames_cl = Frames_cl(frames, cl['context'], cl['queue'])
    frames_cl.load_buffers(r_chunk_size=r_chunk_size, d_chunk_size=d_chunk_size)

    logR_dr = np.zeros((D, R), dtype=np.float32)
    w_d = frames.w_d

    t = 0
    for r00, r11, dr in tqdm(utils.chunker(r_chunk_size, R)):
        for d0, d1, dd in utils.chunker(d_chunk_size, D):
            # get data
            K_chunk = K_di[d0:d1]

            # get frame models: F_dri = w_d C_i W_ri + B_di
            F_dri = frames_cl.calculate_frame(d0, d1, r0+r00, r0+r11, log=True, cpu=True)

            # calculate dot product
            # logR_dr = sum_i K_di F_dri - F_dr
            # F_dr = w_d wsums_r + B_d
            # we don't need B_d term (normalised away)
            t0 = time.time()
            logR = np.sum(K_chunk[:, None, :] * F_dri, axis=-1)
            logR -= w_d[d0:d1, None] * wsums_r[None, r00:r11] + frames.B_di.data_sum[d0:d1, None]

            logR_dr[d0:d1, r00:r11] += logR

            t += time.time() - t0

    return logR_dr, t


@profiling.timed
def calculate_logR_class_0_cpu(K_di, frames, cl, cl_cpu, r0, r1, d_chunk_size=1024, r_chunk_size=1024, fluence_free=False):
    """
    all in memory cpu + opencl process
    """
    t0 = time.time()
    D, R0, I = frames.shape

    R = r1-r0

    r_chunk_size = min(r_chunk_size, R)
    d_chunk_size = min(d_chunk_size, D)

    # calculate wsums_r
    tomos_cl = Tomograms_cl(frames.tomo, cl['context'], cl['queue'])
    print(f'\n\n{r_chunk_size=}\n\n')
    tomos_cl.load_buffers(r_chunk_size)

    if fluence_free:
        calc_logR = Calc_logR_back_fluence_free(frames, cl_cpu['context'], cl_cpu['queue'])
    else:
        calc_logR = Calc_logR(frames, cl_cpu['context'], cl_cpu['queue'])

    calc_logR.load_logR_buffers(r_chunk_size, d_chunk_size, K_di.shape[1])

    logR_dr = np.zeros((D, R), dtype=np.float32)
    w_d = frames.w_d

    for r00, r11, dr in tqdm(utils.chunker(r_chunk_size, R)):
        W_ri = tomos_cl.calculate_tomogram(r0+r00, r0+r11, log=False, cpu=True)

        wsums_r = np.sum(W_ri, axis=-1)

        for d0, d1, dd in utils.chunker(d_chunk_size, D):
            # get data
            K_chunk = K_di[d0:d1]

            if fluence_free:
                logR = calc_logR.calculate_logR(d0, d1, dr, W_ri, K_chunk, K_di.data_sum[d0: d1], wsums_r)
            else:
                print(f'{D=} {d_chunk_size=}')
                logR = calc_logR.calculate_logR(d0, d1, dr, W_ri, K_chunk)
                # logR_dr = sum_i K_di log(F_dri) - F_dr
                # F_dr = w_d wsums_r + B_d
                logR -= w_d[d0:d1, None] * wsums_r[None, :] #+ frames.B_di.data_sum[d0:d1, None]

            logR_dr[d0:d1, r00:r11] = logR

    t = time.time() - t0

    return logR_dr, t


@profiling.timed
def calculate_logR_class_0_c(c, cl, cl_cpu, r0=None, r1=None):
    if not c['update_logR']:
        return

    R = c['mapper'].shape[1]

    # load fluence
    with h5py.File(c['fluence_file']) as f:
        w_d = f['w_d'][()]
    c['fluence'] = w_d

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

    frames = Frames(tomos, w_d, c['P_data'].B_di)
    print(f'{frames.shape=}')

    fluence_free = (c['likelihood'] == 'fluence_free')

    return calculate_logR_class_0_cpu(c['P_data'], frames, cl, cl_cpu, r0, r1, fluence_free=fluence_free)


def calculate_logR_subprocess(config_file, config, p_per_device=2, cids=None):
    """
    calculate logR in a separate process
    2 processes for each device
    write logR to file, return file names
    """
    import subprocess, sys

    if cids is None:
        cids = range(len(config['classes']))

    for ci in cids:
        c = config['classes'][ci]
        class_id = c['class_id']

        R = c['mapper'].shape[1]
        D = c['P_data'].shape[0]

        if not c['update_logR']:
            continue

        devices = utils_cl.get_devices(device_type='gpu')

        # number parallel processes
        nproc = p_per_device * len(devices)

        # split calculation by r-chunks
        # str = '0-10000 10000-20000 20000-20323'
        r0_r1_str = []
        for s in np.array_split(np.arange(0, R), nproc):
            r0_r1_str.append('-'.join([str(s[0]), str(s[-1]+1)]))
        r0_r1_str = ' '.join(r0_r1_str)

        cmd = f"time parallel --halt now,fail=1 python -m emc3.background.calculate_logR_single {config_file} {ci} {{}} {{%}} ::: {r0_r1_str}"
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
        print(f'{fnams=}')

        with h5py.File(c['logR_file'], 'w') as f:
            f.create_dataset('logR_dr', shape=(D, R), dtype=float)
            for fnam in fnams:
                with h5py.File(str(fnam)) as g:
                    logR_dr = g['logR_dr'][()]
                    r0 = g['r0'][()]
                    r1 = g['r1'][()]

                f['logR_dr'][:, r0:r1] = logR_dr

                # delete
                print(f'deleting {fnam=}')
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
    from pathlib import Path

    config_fnam = sys.argv[1]
    ci = int(sys.argv[2])
    r0, r1 = [int(c) for c in sys.argv[3].split('-')]
    device = int(sys.argv[4])

    config = pickle.load(open(config_fnam, 'rb'))
    wd = config['working_directory']

    profiling.setup(Path(wd) / 'profile')

    c = config['classes'][ci]
    class_id = c['class_id']

    if not c['update_logR']:
        sys.exit()

    with profiling.cl_timed('io:load_data'):
        with h5py.File(c['model_file']) as f:
            c['model'].data = f['data'][()]
        c['P_data'].load_from_file()

    cl = utils_cl.opencl_init(device_no=device)
    cl_cpu = utils_cl.opencl_init_cpu()

    logR_dr, t = calculate_logR_class_0_c(c, cl, cl_cpu, r0=r0, r1=r1)

    with profiling.cl_timed('io:save_logR'):
        fnam = f'class_logR_chunk_{class_id}_{r0}_{r1}.h5'
        with h5py.File(fnam, 'w') as f:
            f['logR_dr'] = logR_dr
            f['r0'] = r0
            f['r1'] = r1

