"""
loop over d:
    calculate N_dsri
    calculate W_dsri
    save N_dsri
    save K_dsri
    save C_dsri
    save w_dsri
    save b_dsri
    save d_dsri
"""
import argparse
import sys
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import subprocess

from .. import utils_cl
from .update_I_buffer_size import get_sparse_P_matrix
from ..utils import solve_axbc
from .. import symmetry
from ..mapper import Mapper_cl_cpu_sparse
from ..update_models import apply_filter
from .update_w_d import w_update
from . import update_I_cn

import pyopencl as cl
import torch

def update_I(config, config_file, iters=1):
    for i in range(iters):
        # cheaper to not update PK on following iterations
        update_w_b = i > 0
        update_W = i > 0

        update_I_cn.calculate_c_n(config, config_file)

        # write PK, C, W, N, d to file
        if i == 0:
            write_raveled_data(config, config_file)

        # solve I
        solve_I(config, config_file, update_w_b=update_w_b)

        # solve w
        w_update(config, config_file)


def solve_I(config, config_file, update_w_b=False):
    cmds = []
    for cid in range(len(config['classes'])):
        cmd = f"""python -c "import emc3; emc3.background.update_I_test._solve_I('{config_file}',{cid},{update_w_b})" """
        cmds.append(cmd)

    fnam = 'solve_I_commands.txt'
    with open(fnam, 'w') as file:
        file.write('\n'.join(cmds))

    cmd = f'parallel --halt now,fail=1 --verbose --delay 2 < {fnam}'

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


def _solve_I(config_file, cid, update_w_b=False):
    config = pickle.load(open(config_file, 'rb'))
    c = config['classes'][cid]

    fnam = f'test_buffers_{cid}.h5'
    with h5py.File(fnam) as f:
        a = f['PK'][()]
        b = f['B'][()]
        d = f['d'][()]
        w = np.ascontiguousarray(f['w_d'][()][d])
        b /= f['C'][()] * w

        # sort a and b
        j = f['j'][()]
        a = a[j]
        b = b[j]

        m_inds = f['m_inds'][()]

    if update_w_b:
        # load fluence
        with h5py.File(config['fluence_file']) as f:
            w_new_d = f['w_d'][()]

        w_new = np.ascontiguousarray(w_new_d[d][j])
        b *= w / w_new
        del w_new
        del w

        # load fluence
        with h5py.File(config['fluence_file']) as f:
            b_new_d = f['b_d'][()]

        with h5py.File(fnam) as f:
            b_old_d = f['b_d'][()]

        b_new = np.ascontiguousarray(b_new_d[d][j])
        b_old = np.ascontiguousarray(b_old_d[d][j])
        b *= b_new / b_old

        del b_new
        del b_old

    del j
    del d

    fnam = f'class_{cid}_c_n.h5'
    with h5py.File(fnam, 'r') as f:
        c_n = f['c_n'][()].ravel()

    # apply symmetry (add symmetry partners) to c_n
    shape = c['model'].shape
    sym = symmetry.Symmetry(
        shape[0]//2,
        shape,
        symmetry=c['model'].symmetry
    )
    n_asy = sym.get_asymmetric_unit()

    M = len(m_inds)-1
    out_m = np.zeros((M,), dtype=float)
    for m in tqdm(range(M)):
        i = m_inds[m]
        j = m_inds[m+1]

        if j > i:
            a_m = a[i:j]
            b_m = b[i:j]
            n = n_asy[m]
            ns = list(sym.get_symmetry_partners(n))

            cc = np.sum(c_n[ns])
            out_m[m] = solve_axbc(
                a_m,
                b_m,
                cc,
                fill_value=0., ftol=1e-2, xtol=1e-3,
                maxiters=1000, algorithm='Halley', debug=False
            )

    out_n = np.zeros(np.prod(shape), dtype=float)
    out_n[n_asy] = out_m

    out_n = sym.apply_symmetry(out_n.reshape(shape))

    O_n = np.zeros(out_n.size, dtype=int)
    O_n[n_asy] = 1
    O_n = sym.apply_symmetry(O_n.reshape(shape))
    O_n[O_n == 0] = 1
    out_n /= O_n

    I_n = out_n

    if Path(c['model_file']).is_file:
        I0_n = None
        with h5py.File(c['model_file']) as f:
            data = f['data']
            dq = f['dq'][()]
            if data.shape == I_n.shape:
                if dq == c['model'].dq:
                    m = I_n == 0.
                    I0_n = data[()]

    if c['filter_model']:
        if I0_n is not None:
            I_n[m] = I0_n[m]
        I_n = apply_filter(c['model'].dq, I_n, c['filter_model'])

    if I0_n is not None:
        rms = np.mean((I0_n - I_n)**2)**0.5
        print(f'rms difference for model {cid}: {rms}')
        print(f'{cid}: {np.mean(I0_n)=} --> {np.mean(I_n)=}')

    # save
    with h5py.File(c['model_file'], 'w') as f:
        f['data'] = I_n
        f['dq'] = c['model'].dq


def write_raveled_data(config, config_file):
    cmds = []
    for cid in range(len(config['classes'])):
        cmd = f"""python -c "import emc3; emc3.background.update_I_test._write_raveled_data('{config_file}',{cid})" """
        cmds.append(cmd)

    fnam = 'write_raveled_data_commands.txt'
    with open(fnam, 'w') as file:
        file.write('\n'.join(cmds))

    cmd = f'parallel --halt now,fail=1 --verbose --delay 2 < {fnam}'

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


def _write_raveled_data(config_file, cid):
    config = pickle.load(open(config_file, 'rb'))

    cl_gpu = utils_cl.opencl_init()
    cl_cpu = utils_cl.opencl_init_cpu()

    # load fluence
    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]

    c = config['classes'][cid]

    # load prob
    with h5py.File(c['probability_matrix_file']) as f:
        P_dr = f['P_dr'][()]

    C_i = np.ascontiguousarray(c['data'].C_i.astype(np.float32))

    N = c['model'].size

    mapper = c['mapper']
    mapper.load_coords(c['data'].mask)

    mapper_cpu = Mapper_cl_cpu_sparse(
            mapper,
            cl_cpu['context'],
            cl_cpu['queue'])

    S = mapper_cpu.shape[0]
    D = c['data'].shape[0]

    Ps_dr, rs_d, Nr = get_sparse_P_matrix(P_dr)

    K_di = c['data']
    K_di.load_from_file()
    B_di = K_di.B_di

    rmax = np.max([len(rs) for rs in rs_d])
    max_buf_shape = (S,) + (rmax,) + (K_di.data.litpix.max(),)

    model = c['model']

    sym = symmetry.Symmetry(
        model.shape[0]//2,
        model.shape,
        symmetry=model.symmetry
    )

    # n: raveled model voxel location
    # m: raveled asymmetric unit voxel location
    # n = n_asy[m]
    # m = M_n[n]
    n_asy = sym.get_asymmetric_unit()
    M_n = sym.get_asymmetric_unit_mapping()

    # prepare h5 file
    # total size = S x sum_d rs_d[d] * K.pixels
    Ntot = S * np.sum([len(rs) * I for rs, I in zip(rs_d, K_di.data.litpix)])
    fnam = f'test_buffers_{cid}.h5'
    with h5py.File(fnam, 'w') as f:
        f.create_dataset('B', shape=(Ntot,), dtype=np.float32)
        f.create_dataset('C', shape=(Ntot,), dtype=np.float32)
        f.create_dataset('PK', shape=(Ntot,), dtype=np.float32)
        f.create_dataset('d', shape=(Ntot,), dtype=np.uint32)
        f.create_dataset('M', shape=(Ntot,), dtype=np.uint32)

        f['w_d'] = w_d
        f['b_d'] = B_di.b_d

        #N_n = np.zeros(c['model'].size, dtype=int)
        index = 0
        for d in tqdm(range(D), desc='calculating buffer sizes'):
            if len(rs_d[d]) > 0:
                rs = rs_d[d]
                pixels, Kd_i = c['data'].data.get_sparse(d)
                Bd_i = B_di[d][0][pixels]
                Cd_i = C_i[pixels]
                Pd_r = Ps_dr[d]

                PK_ri = Pd_r[:, None] * Kd_i[None, :]
                for s in range(S):
                    N_sri = mapper_cpu.calculate_mapping_rlist_pixlist(s, rs, pixels, ravel=True).ravel()

                    # this is an ri chunk
                    i0, i1 = index, index + len(N_sri)
                    f['M'][i0: i1]  = M_n[N_sri]
                    f['PK'][i0: i1] = PK_ri.ravel()
                    f['B'][i0: i1]  = np.broadcast_to(Bd_i, (len(rs), len(pixels))).ravel()
                    f['C'][i0: i1]  = np.broadcast_to(Cd_i, (len(rs), len(pixels))).ravel()
                    f['d'][i0: i1]  = d

                    index = i1

    # sort M
    with h5py.File(fnam, 'r') as f:
        M = f['M'][()]

    M = torch.from_numpy(M.astype(np.int32))
    j = torch.argsort(M)
    M_j = M[j]

    with h5py.File(fnam, 'a') as f:
        f['j'] = np.array(j, dtype=np.uint32)
        del j

    m_inds = np.searchsorted(np.array(M_j), np.arange(n_asy.size))
    # add last index
    m_inds = np.concatenate([m_inds, np.array([n_asy.size])])
    with h5py.File(fnam, 'a') as f:
        f['m_inds'] = m_inds
    return fnam


def main2(fnam, config_file, cid):
    """
    N (model voxels) --> M (model asymmetric unit voxel)
    argsort M
    """
    config = pickle.load(open(config_file, 'rb'))
    c = config['classes'][cid]

    with h5py.File(fnam) as f:
        N = f['N'][()]

    model = c['model']

    sym = symmetry.Symmetry(
        model.shape[0]//2,
        model.shape,
        symmetry=model.symmetry
    )

    # n: raveled model voxel location
    # m: raveled asymmetric unit voxel location
    # n = n_asy[m]
    # m = M_n[n]
    n_asy = sym.get_asymmetric_unit()
    M_n = sym.get_asymmetric_unit_mapping()

    M = M_n[N]
    del N

    with h5py.File(fnam, 'a') as f:
        f['M'] = M.astype(np.uint32)

    M_t = torch.from_numpy(M.astype(np.int32))

    del M

    # fast compared to numpy
    j = torch.argsort(M_t)

    del M_t

    # save
    with h5py.File(fnam, 'a') as f:
        f['j'] = j.astype(np.uint32)

    # save the starting index for each m
    M = M[j]  # hopefully free's memory
    M_j = np.searchsorted(M, np.arange(n_asy.size))
    with h5py.File(fnam, 'a') as f:
        f['m_inds'] = M_j


def main3(fnam, config_file, cid):
    """
    testing: write a b
    a_j = P_j K_j
    b_j = B_j / (C_j * w_j);

    where j indexes the set of s, r, i values that map to a given
    model voxel in the assymetric unit
    """
    with h5py.File(fnam, 'r') as f:
        j = f['j'][()]

        P = f['P'][()]
        K = f['K'][()]
        a = (P * K)[j]

        B = f['B'][()]
        C = f['C'][()]
        w = f['w'][()]
        b = (B /(C * w))[j]

    # save
    with h5py.File(fnam, 'a') as f:
        f['a'] = a
        f['b'] = b


def main4(fnam, config_file, cid):
    with h5py.File(fnam, 'a') as f:
        a = f['a'][()]
        b = f['b'][()]
        m_inds = f['m_inds'][()]

    fnam = f'class_{class_id}_c_n.h5'
    with h5py.File(fnam, 'r') as f:
        c_n = f['c_n'][()]

    # apply symmetry (add symmetry partners) to c_n
    shape = c['model'].shape
    model_symmetry = c['model'].symmetry
    sym = symmetry.Symmetry(
        shape[0]//2,
        shape,
        symmetry=model_symmetry
    )

    # solve
    M = fill_buffers.index0_m.size
    out_m = np.zeros((M,), dtype=float)
    a_n = fill_buffers.a
    b_n = fill_buffers.b

    for m in tqdm(range(M), desc='solving'):
        i = fill_buffers.index0_m[m]
        j = fill_buffers.index0_m[m] + fill_buffers.counts_m[m]
        n = fill_buffers.n_asy[m+m_min]
        if j > i:
            ns = list(sym.get_symmetry_partners(n))
            cc = np.sum(c_n[ns])
            out_m[m] = solve_axbc(
                a_n[i: j],
                b_n[i: j],
                cc,
                fill_value=0., ftol=1e-2, xtol=1e-3,
                maxiters=1000, algorithm='Halley', debug=False
            )

if __name__ == '__main__':
    config_fnam = sys.argv[1]
    cid = int(sys.argv[2])

    main(config_fnam, cid)
