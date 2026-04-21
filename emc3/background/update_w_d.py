import argparse
from pathlib import Path
import sys
import h5py
import pickle
import numpy as np
import os
from tqdm import tqdm
import subprocess

from ..utils import solve_axbc
from .. import utils_cl
from ..tomograms import Tomograms, Tomograms_cl

def _w_update(P_cdr, K_di, B_di, W_cri, Wsums_cr, d0, d1, update_b=False, w0_d=None):
    """
    P_cdr is chunked
    all others have absolute indices

    assume sparse P_dr and K_di
    we need:
        a_j = {Pd_r Kd_i}         for non-zero elements
        b_j = {Bd_i / (C_i W_ri}  for above elements
        c_d = sum_r Pd_r Wsums_r

    sum_j a_j / (w_d + b_j) - c_d = 0

    calculate W_ri on cpu? give it a go

    wmax = sum_i K_di / sum_r P_dr Wsums_r
         = sum_i K_di / sum_r P_dr Wsums_r

    or calculate W_ri on gpu with r and i list


    Optional, update background weighting factors:
    ---------------------------------------------
    B_di is a numpy array (1, pixels)

    F_dri = w_d C_i W_ri + b_d B_di (here B_di does not include weighting)

    assume sparse P_dr and K_di
    we need:
        a_j = {Pd_r Kd_i}           for non-zero elements
        b_j = {w_d C_i W_ri / B_di} for above elements
        c_d = B_d

    a_j is the same as above and we can re-use W_ri calculations
    """
    # we shouldn't need this as P_thresh is already
    # used to zero P_dr values where P_dr[d] < P_thresh max_r(P_dr[d])
    # P_thresh = 0.01
    D = d1 - d0
    I = K_di.shape[1]
    C = len(P_cdr)

    assert (len(P_cdr) == len(Wsums_cr))

    w_d = np.zeros(D, dtype=float)

    # np accelerates this over cpus
    c_d = np.zeros(D, dtype=float)
    for P_dr, Wsums_r in tqdm(
        zip(P_cdr, Wsums_cr),
        total=C,
        desc='calculating c_d = sum_r P_dr Wsums_r'
    ):
        c_d += np.dot(P_dr, Wsums_r)

    # make sparse P_dr
    # don't do a = C * [[]] this makes each list reference
    # the same object
    rs_cd = [[] for c in range(C)]
    ds_c = [[] for c in range(C)]
    Ps_cdr = [[] for c in range(C)]
    Nr = 0

    buffer_size_c = []
    for c in tqdm(range(C), desc='generating sparse P-matrix'):
        b_max = 0
        for d in range(D):
            rs = np.where(P_cdr[c][d] > 0)[0]
            Ps_cdr[c].append(P_cdr[c][d][rs])
            rs_cd[c].append(rs)
            Nr += len(rs)

            buffer_size = (len(rs)*K_di.data.litpix[d + d0])
            if buffer_size > b_max:
                b_max = buffer_size

        print(f'maximum W_cl buffersize:', c, b_max)
        sys.stdout.flush()

        buffer_size_c.append(b_max)
        if b_max > 0:
            W_cri[c].load_buffers(buffer_size=b_max)

    print(f'average number of tomograms per frame: {Nr/D:.3f}')

    # this is slower that single cpu (surprising) but I guess
    # the overhead with thread management
    # is larger than the actual average computation time.
    # solve = utils.Solve_axbc_cl(debug = True, queue = config['queue_cpu'],
    #           context = config['context_cpu'])
    # tqueue = utils.ThreadQueue(max_queue_depth = 32)
    # tqueue.submit(
    #     solve.solve, a, b, c_d[d], w_d, d
    # )
    # tqueue.shutdown()

    # optional: update background weighting factors
    # ---------------------------------------------
    if update_b:
        cb_d = np.sum(B_di.B_ji)
        b_d = np.zeros(D, dtype=float)

    # dominated by W_ri evaluation
    # could use threading to calculate W on different gpus simultaneously
    # or multiprocessing to call this funtion
    for di, d in tqdm(enumerate(range(d0, d1)), total=D, desc='updating w_d', disable=False):
        pixels, Kd_i = K_di.data.get_sparse(d)
        Bd_i = B_di[d][0][pixels]

        Psd_r = np.concatenate(
            [Ps_cdr[c][di] for c in range(C)]
        )

        Ws_ri = []
        for c in range(C):
            if len(rs_cd[c][di]) == 0:
                continue

            Ws_ri.append(W_cri[c].calculate_tomogram_rlist_pixlist(rs_cd[c][di], pixels))

        Ws_ri = np.concatenate(Ws_ri)
        Ws_ri = np.clip(Ws_ri, 1e-10, None)

        a = np.outer(Psd_r, Kd_i)
        b = Bd_i.astype(float) / Ws_ri

        a = np.ascontiguousarray(a.ravel().astype(np.float64))
        b = np.ascontiguousarray(b.ravel().astype(np.float64))
        a = np.clip(a, 0, None)
        b = np.clip(b, 0, None)

        if not np.all(b >= 0.):
            print(f'{b[b<0.]=}')
            print(f'{b=}')

        w_d[di] = solve_axbc(
            a, b, c_d[di], fill_value=0., ftol=1e-2, xtol=1e-3,
            maxiters=1000, algorithm='Newton', debug=False
        )

        w_d[di] = max(w_d[di], 1e-10)

        if update_b:
            b = w_d[di] * Ws_ri / B_di.B_ji[0][pixels]
            b = np.ascontiguousarray(b.ravel().astype(np.float64))
            b = np.clip(b, 0, None)

            b_d[di] = solve_axbc(
                a, b, cb_d, fill_value=0., ftol=1e-2, xtol=1e-3,
                maxiters=1000, algorithm='Newton', debug=False
            )

            b_d[di] = max(b_d[di], 1e-3)

    if update_b:
        return w_d, b_d
    else:
        return w_d, None


def calculate_wsums_cl(
        mapper=None,
        model=None,
        model_file=None,
        data=None,
        cl=None,
        **kwargs
        ):

    if cl is None:
        cl = utils_cl.opencl_init()

    # load model
    if model.data is None:
        with h5py.File(model_file) as f:
            model.data = f['data'][()]

    mapper.load_coords(data.mask)

    tomos = Tomograms(
            mapper,
            model,
            data.C_i)

    tomos_cl = Tomograms_cl(tomos, cl['context'], cl['queue'])

    wsums_r = tomos_cl.calculate_wsums()

    return wsums_r


def w_update(config, config_file, update_b=False, p_per_device=16):
    """
    - calculate wsums
    - calculate w_d split over frames
    """
    if not config['update_fluence']:
        print('update_fluence is False skipping w_d update')
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()

    cids = list(range(len(config['classes'])))
    my_classes = cids[rank::size]

    cl = utils_cl.opencl_init()

    for ci in my_classes:
        c = config['classes'][ci]
        wsums_r = calculate_wsums_cl(**c, cl=cl)

        # save
        with h5py.File(c['wsums_file'], 'w') as f:
            f['wsums_r'] = wsums_r

    comm.Barrier()

    # chunk over frames
    # -----------------
    devices = utils_cl.get_devices(device_type='gpu')

    # number parallel processes
    nproc = p_per_device

    # first split by rank then by nproc
    # ---------------------------------
    D = config['classes'][0]['data'].shape[0]
    d0_d1_rank = []
    d0_d1 = []
    for r, s in enumerate(np.array_split(np.arange(0, D), size)):
        d00, d11 = s[0], s[-1]+1
        d0_d1_rank.append([d00, d11])

        # str = '0-10000 10000-20000 20000-20323'
        d0_d1_str = []
        for s in np.array_split(np.arange(d00, d11), nproc):
            d0, d1 = s[0], s[-1]+1
            d0_d1.append([d0, d1])
            d0_d1_str.append('-'.join([str(d0), str(d1)]))
        d0_d1_str = ' '.join(d0_d1_str)

        if r == rank:
            my_d0_d1_str = d0_d1_str

    cmd = f"time parallel --halt now,fail=1 python -m emc3.background.update_w_d {config_file} {{}} {{%}} {update_b} ::: {my_d0_d1_str}"
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

    comm.Barrier()

    # merge
    if rank == 0:
        w_d = np.zeros(config['classes'][0]['data'].shape[0], dtype=float)
        b_d = np.zeros(config['classes'][0]['data'].shape[0], dtype=float)
        for d0, d1 in d0_d1:
            fnam = Path(f'fluence_chunk_{d0}_{d1}.h5')
            with h5py.File(str(fnam)) as f:
                w_d[d0:d1] = f['w_d'][()]
                if update_b:
                    b_d[d0:d1] = f['b_d'][()]

            fnam.unlink()

        w_d = np.clip(w_d, 1e-8, None)
        b_d = np.clip(b_d, 1e-8, None)

        with h5py.File(config['fluence_file'], 'w') as f:
            f['w_d'] = w_d
            if update_b:
                f['b_d'] = b_d

    comm.Barrier()

    # hack this into background getter
    #c = config['classes'][0]
    #with h5py.File(c['data'].B_di.fnam, 'a') as f:
    #    f['b_d'][:] = b_d


if __name__ == '__main__':
    config_fnam = sys.argv[1]
    d0, d1 = [int(c) for c in sys.argv[2].split('-')]
    device = int(sys.argv[3])
    update_b = sys.argv[4] == 'True'

    config = pickle.load(open(config_fnam, 'rb'))

    cl = utils_cl.opencl_init()

    # load stuff
    K_di = config['classes'][0]['data']
    B_di = K_di.B_di

    with h5py.File(config['fluence_file']) as f:
        w0_d = f['w_d'][()]

    P_cdr = []
    W_cri = []
    wsums_cr = []
    for c in config['classes']:
        with h5py.File(c['model_file']) as f:
            c['model'].data = f['data'][()]

        c['data'].load_from_file()
        c['mapper'].load_coords(c['data'].mask)

        tomos = Tomograms(
                c['mapper'],
                c['model'],
                c['data'].C_i)

        tomos_cl = Tomograms_cl(tomos, cl['context'], cl['queue'])

        W_cri.append(tomos_cl)

        with h5py.File(c['probability_matrix_file']) as f:
            P_cdr.append(f['P_dr'][d0: d1])

        with h5py.File(c['wsums_file']) as f:
            wsums_cr.append(f['wsums_r'][()])

    # main call
    w_d, b_d = _w_update(P_cdr, K_di, B_di, W_cri, wsums_cr, d0, d1, update_b=update_b, w0_d=w0_d)

    # save
    fnam = f'fluence_chunk_{d0}_{d1}.h5'
    with h5py.File(fnam, 'w') as f:
        f['w_d'] = w_d
        f['d0'] = d0
        f['d1'] = d1

        if update_b:
            f['b_d'] = b_d
