import sys
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .. import utils_cl
from .. import symmetry
from ..mapper import Mapper_cl_cpu_sparse

import pyopencl as cl
import subprocess
import pickle

def get_sparse_P_matrix(P_dr):
    D, R = P_dr.shape

    # make sparse P_dr
    print('generating sparse P-matrix (start)')
    rs_d = []
    Ps_dr = []
    Nr = 0

    for d in tqdm(range(D), desc='generating sparse P-matrix'):
        rs = np.where(P_dr[d] > 0)[0]
        Ps_dr.append(P_dr[d][rs])
        rs_d.append(rs)
        Nr += len(rs)

    # it's possible that no frames contribute to a given class
    print(f'average number of tomograms per frame: {Nr/D:.3f}')

    a = [min(P) for P in Ps_dr if len(P) > 0]
    b = [max(P) for P in Ps_dr if len(P) > 0]
    if len(a) > 0:
        print(f'minimum P value after threshold: '
                     f'{np.min(a)}')

        print(f'maximum P value after threshold: '
                     f'{np.max([max(P) for P in Ps_dr if len(P) > 0])}')
    else:
        print('P == 0 for all orientation for this class')

    print('generating sparse P-matrix (stop)')
    return Ps_dr, rs_d, Nr


def update_buffer_size(config, config_file, p_per_device=2, cids=None):
    # classes to process
    if cids is None:
        cids = list(range(len(config['classes'])))

    for ci in cids:
        c = config['classes'][ci]

        if not c['update_model']:
            cids.remove(ci)

    cids_str = ' '.join([str(i) for i in cids])

    # number parallel processes (how to choose?)
    devices = utils_cl.get_devices(device_type='gpu')
    nproc = p_per_device * len(devices)

    D = config['classes'][0]['data'].shape[0]

    # if there is only one class then split over frames
    d0_d1_str = []
    if len(cids) == 1:
        for s in np.array_split(np.arange(0, D), nproc):
            d0_d1_str.append('-'.join([str(s[0]), str(s[-1]+1)]))
        d0_d1_str = ' '.join(d0_d1_str)
    else:
        d0_d1_str = f'0-{D}'

    cmd = f"time parallel --retries 3 --halt now,fail=1 'python -m emc3.background.update_I_buffer_size {config_file} {{1}} {{2}} {{%}} 2>&1' ::: {cids_str} ::: {d0_d1_str}"
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

    for ci in cids:
        c = config['classes'][ci]
        cid = c['class_id']

        counts_n = np.zeros(c['model'].shape, dtype=int)
        fnams = Path('./').glob(f'class_{cid}_buffersize_*_*.h5')
        for fnam in fnams:
            with h5py.File(fnam) as f:
                counts_n += f['counts_n'][()]

            fnam.unlink()

        fnam_out = f'class_{cid}_buffersize.h5'

        # calculate counts per assymetric unit
        sym = symmetry.Symmetry(
            c['model'].shape[0]//2,
            c['model'].shape,
            symmetry=c['model'].symmetry
        )

        # n: raveled model voxel location
        # m: raveled asymmetric unit voxel location
        # n = n_asy[m]
        # m = M_n[n]
        n_asy = sym.get_asymmetric_unit()
        M_n = sym.get_asymmetric_unit_mapping()

        # buffer size = symmetry operations sum_n>=n_min^n_max counts_n
        counts_m = np.bincount(M_n, weights=counts_n.ravel(),
                               minlength=n_asy.size)

        counts_m = np.rint(counts_m).astype(int)

        # save
        with h5py.File(f'class_{cid}_buffersize.h5', 'w') as f:
            f['counts_n'] = counts_n
            f['counts_m'] = counts_m
            f['n_asym'] = n_asy
            f['M_n'] = M_n


if __name__ == "__main__":
    config_fnam = sys.argv[1]
    cid = int(sys.argv[2])
    d0, d1 = [int(c) for c in sys.argv[3].split('-')]
    dd = d1-d0
    device = int(sys.argv[4])

    config = pickle.load(open(config_fnam, 'rb'))

    c = config['classes'][cid]

    with h5py.File(c['probability_matrix_file']) as f:
        P_dr = f['P_dr'][d0: d1]

    cl_cpu = utils_cl.opencl_init_cpu()

    # Start main calculation
    # ----------------------
    with h5py.File(c['fluence_file']) as f:
        w_d = f['w_d'][d0: d1]

    with h5py.File(c['probability_matrix_file']) as f:
        P_dr = f['P_dr'][d0: d1]

    c['data'].load_from_file()

    C_i = np.ascontiguousarray(c['data'].C_i.astype(np.float32))

    N = c['model'].size

    Ps_dr, rs_d, Nr = get_sparse_P_matrix(P_dr)

    # check for empty Ps_dr
    if np.max([len(P) for P in Ps_dr]) == 0:
        counts_n = np.zeros((N,), dtype=int)

    else:
        # re-initialising this makes it much faster
        # probably large buffers slow it down
        # should look into it
        # could be cpu vs gpu
        mapper = c['mapper']
        mapper.load_coords(c['data'].mask)

        mapper_cpu = Mapper_cl_cpu_sparse(
                mapper,
                cl_cpu['context'],
                cl_cpu['queue'])

        # get buffer size per-n
        # number of pixels contributing to each model voxel
        # N_n = sum_dsr{i: for K_di>0} delta(n - M_sri)
        litpix = c['data'].data.total_row_counts[d0:d1]

        rmax = np.max([len(rs) for rs in rs_d])
        rmax = min(rmax, 32)
        max_buf_shape = (1,) + (rmax,) + (litpix.max(),)
        print(f'{cid=} {max_buf_shape=}')
        print(f'{cid=} {np.prod(max_buf_shape)=}')
        bincount = utils_cl.Bincount_cl(
                N,
                np.prod(max_buf_shape),
                cl_cpu['queue'],
                cl_cpu['context']
        )

        S = mapper_cpu.shape[0]
        #N_n = np.zeros(c['model'].size, dtype=int)
        for d in tqdm(range(dd), desc=f'calculating buffer sizes {cid}'):
            if len(rs_d[d]) > 0:
                pixels, Kd_i = c['data'].data.get_sparse(d0 + d)
                index = 0
                while index < len(rs_d[d]):
                    i = index
                    j = min(index+rmax, len(rs_d[d]))
                    rs = rs_d[d][i: j]
                    for s in range(S):
                        N_sri = mapper_cpu.calculate_mapping_rlist_pixlist(s, rs, pixels, ravel=True)
                        bincount.add(N_sri.copy())
                        #N_n += np.bincount(N_sri.ravel(), minlength=N_n.size)
                    index = j

        bincount.queue.finish()
        counts_n = bincount.out

        #assert(np.allclose(counts_n, N_n))

    print(f'{cid=} {np.sum(counts_n)=}')

    # save
    fnam = f'class_{c["class_id"]}_buffersize_{d0}_{d1}.h5'
    print(f'writing {fnam=}')
    with h5py.File(fnam, 'w') as f:
        f['counts_n'] = counts_n.reshape(c['model'].shape)
        f['d0'] = d0
        f['d1'] = d1
