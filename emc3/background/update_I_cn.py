"""
w_d update:
-----------
    dQ / dw_d = sum_ri P_dr C_i W_ri (K_di / F_dri - 1)
              = sum_ri P_dr C_i W_ri K_di / (w_d C_i W_ri + B_di)
              - sum_ri P_dr C_i W_ri

              = sum_ri P_dr K_di / (w_d + B_di / (C_i W_ri))
              - sum_ri P_dr C_i W_ri
              = sum_j a_j / (x + b_j) - c = 0

    a_j = P_dr K_di for P_dr > 0

    wmax = sum_ri P_dr K_di / sum_ri P_dr C_i W_ri
         = sum_i K_di / sum_r P_dr Wsums_r

assume sparse P_dr
we need:
    a_j = {P_dr K_di}         for non-zero elements
    b_j = {B_di / (C_i W_ri}  for above elements
    c_d = sum_r P_dr wsums_r


I_n update:
-----------
    dQ / dI_n = sum_dr sum_(i in M_rn) P_dr K_di / (I_n + B_di / (w_d C_i))
              - sum_r sum_(i in M_rn) C_i sum_d w_d P_dr
              = sum_j a_j / (x + b_j) - c = 0

assume sparse P_dr and K_di
    a_j = P_dr K_di         for i in M_rn and for non-zero elements
    b_j = B_di / (w_d C_i)  for above elements
    c   = C_r wP_r          where wP_r = sum_d w_d P_dr
                            and C_r = sum_(i in M_rn) C_i

loop over r, get contributing frames and non-zero pixels, calculate M_rn

I need a way to get the pixel values where Kd_i > 0 and Nr_i == n

This is basically an exercise in mapping each r,i pair to an n
then removing zero elements which depends on d
the (r, i) to n mapping is the same for all classes with the same
mapping function so it's feasable to store this for 2D classes
on the order of 20 x 200,000 numbers
but in 3D with high symmetry this would be a very large dataset

a_nm = F_l[l_nm], l_nm = inv(n_l)

dense, for a given 'n' j --> d, r, m:
a_ndrm = F_dri[d, r, i_nrm] = P[d, r] K[d, i_nrm]

P[d, r] is sparse, so looping over d means there are only a sub-set of r's to
process

K[d, i] is sparse, but this is harder to make use of since we have to
find the common elemnts between non-zero K[d, i] and i_nrm values
The fix for this would be to store the i'th non-zero elements of K[d, i]
instead of absolute i index. But this leads to a much bigger index array:
    i_nrm --> i_ndrm
    this will not change unless the pixel mask or mapping matrix changes
    k_ndrm: Ks[d, k] = K[d, inds[d, k]] sparse frame d with all Ks>0
    a_ndrm = F[d, r, i[n, r, m]] = P[d, r] Ks[d, k[n, d, r, m]]

If every photon is directed to some voxel, then the total size of k
will be:
    [k] = R x number of non-zero elements in K_di
which will be a very large dataset (2D-EMC) ~ 10 Gb
"""
import numpy as np
from .. import utils_cl
from .. import utils
from ..mapper import Mapper_cl

import pyopencl as cl
import h5py
from tqdm import tqdm
import sys
import subprocess
import pickle

# this should be chunked over r, not d
def _calculate_c_n(w_d, P_dr, M_sri, C_i, N, queue, context):
    """
    c_n = C_i_n sum_d (w_d P_dr_n)
    """
    code = """
    // single worker single group
    __kernel void add (
        global int *N_sri,
        global float *wP_r,
        global float *C_i,
        global float *out,
        const  int S,
        const  int R,
        const  int I
    ) {
    int n, s, r, i;
    float wP, C;
    for (s=0; s<S; s++){
    for (r=0; r<R; r++){
        wP = wP_r[r];
        for (i=0; i<I; i++){
            n = N_sri[s * R * I + r * I + i];
            C = C_i[i];
            out[n] += wP * C;
    }}}
    }
    """
    S, R, I = M_sri.shape
    S = np.int32(S)
    I = np.int32(I)

    # np accelerates this over cpus
    for i in tqdm(range(1), desc='calculating wP_r = sum_d w_d P_dr'):
        wP_r = np.dot(w_d, P_dr)
        wP_r = np.ascontiguousarray(wP_r.astype(np.float32))

    # now we need to merge C_i wP_r into n-space (model-space)
    cc_n = np.zeros(N, dtype=np.float32)
    r_chunk_size = 32

    M_sri.load_buffers(
        r_chunk_size=r_chunk_size,
        ravel=True
    )

    r_iter = tqdm(
        utils.chunker(r_chunk_size, R),
        desc='merging C_i sum_d w_d P_dr for model'
    )

    cl_code = cl.Program(context, code).build()
    queue = queue
    events = None

    for r0, r1, dr in r_iter:
        ns = []
        for s in range(S):
            ns.append(M_sri.calculate_mapping(s, r0, r1, ravel=True, cpu=True).copy())
        N_sri = np.ascontiguousarray(np.array(ns).astype(np.int32))

        if events is not None:
            events[0].wait()

        N_sri_temp = N_sri.copy()
        wP_r_temp = wP_r[r0:r1].copy()

        event = cl.Kernel(cl_code, 'add')(
                queue, (1,), (1,),
                cl.SVM(N_sri_temp),
                cl.SVM(wP_r_temp),
                cl.SVM(C_i),
                cl.SVM(cc_n),
                S,
                np.int32(r1-r0),
                I
        )
        events = [event]

    event.wait()
    return cc_n


def calculate_c_n(config, config_file):
    # classes to process
    cids = []
    for ci, c in enumerate(config['classes']):
        if c['update_model']:
            cids.append(ci)

    cids_str = ' '.join([str(i) for i in cids])

    # number parallel processes (how to choose?)
    nproc = 8

    cmd = f"time parallel --halt now,fail=1 --jobs {nproc} python -m emc3.background.update_I_cn {config_file} ::: {cids_str}"
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


if __name__ == '__main__':
    config_fnam = sys.argv[1]
    cid = int(sys.argv[2])

    config = pickle.load(open(config_fnam, 'rb'))

    cl_gpu = utils_cl.opencl_init()
    cl_cpu = utils_cl.opencl_init_cpu()

    # load fluence
    with h5py.File(config['fluence_file']) as f:
        w_d = f['w_d'][()]

    c = config['classes'][cid]

    # load prob
    with h5py.File(c['probability_matrix_file']) as f:
        c['P_dr'] = f['P_dr'][()]

    C_i = np.ascontiguousarray(c['data'].C_i.astype(np.float32))

    N = c['model'].size

    c['mapper'].load_coords(c['data'].mask)

    mapper_cl = Mapper_cl(c['mapper'], cl_gpu['context'], cl_gpu['queue'])

    c_n = _calculate_c_n(
        w_d,
        c['P_dr'],
        mapper_cl,
        C_i,
        N,
        cl_cpu['queue'],
        cl_cpu['context']
    )

    # save
    class_id = c['class_id']
    fnam = f'class_{class_id}_c_n.h5'
    with h5py.File(fnam, 'w') as f:
        f['c_n'] = c_n.reshape(c['model'].shape)
