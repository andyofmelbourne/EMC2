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
"""

import numpy as np
from tqdm import tqdm

from . import utils
from . import symmetry

import pyopencl as cl
import logging

logger = logging.getLogger(__name__)

code = """

// a_j = P_dr K_di         for i in M_rn and for non-zero elements
// b_j = B_di / (w_d C_i)  for above elements
// c   = C_r wP_r          where wP_r = sum_d w_d P_dr
//                         and C_r = sum_(i in M_rn) C_i

// might be more efficient to loop over s and r
// to avoid redundant P * K and B / (w C) calcs
// (checked intelHD) doesn't seem to make a difference
// and the code is messier
__kernel void fill_buffer (
    global float *a_dsri,
    global float *b_dsri,
    global int   *n_dsri,
    global float *P_r,
    global uchar *K_i,
    global float *B_i,
    global float *C_i,
    const  float w,
    global int   *N_sri,
    const  int   offset,
    const  int   d
) {
    int i = get_global_id(0);
    int I = get_global_size(0);

    int r = get_global_id(1);
    int R = get_global_size(1);

    int s = get_global_id(2);
    int S = get_global_size(2);

    int j;

    j = offset + s * R * I + r * I + i;

    n_dsri[j] = N_sri[j-offset];
    a_dsri[j] = P_r[r] * (float)K_i[i];
    b_dsri[j] = B_i[i] / (C_i[i] * w);
}

"""


def I_update(w_d, I_n, P_dr, K_di, B_di, M_sri, C_i,
             queue, context, device,
             model_symmetry):
    # we shouldn't need this as P_thresh is already
    # used to zero P_dr values where P_dr[d] < P_thresh max_r(P_dr[d])
    # P_thresh = 0.01
    # P_thresh = 1e-3

    D, I = K_di.shape
    R = P_dr.shape[1]

    # np accelerates this over cpus
    for i in tqdm(range(1), desc='calculating wP_r = sum_d w_d P_dr'):
        wP_r = np.dot(w_d, P_dr)

    # now we need to merge C_i wP_r into n-space (model-space)
    cc_n = np.zeros(I_n.size, dtype=float)
    r_chunk_size = 128

    S = np.int32(M_sri.shape[0])

    t = np.zeros((S, r_chunk_size, I), dtype=np.float32)

    r_iter = tqdm(
        utils.chunker(r_chunk_size, R),
        desc='merging C_i sum_d w_d P_dr for model'
    )

    for r0, r1, dr in r_iter:
        # (sym * rs, pixels)
        for _ in tqdm(range(1),
                      desc='mapping pixels for r-chunk', leave=False):

            N_sri = M_sri[:, r0:r1, :]

        for _ in tqdm(range(1),
                      desc='bincount merging over r-chunk', leave=False):

            # this broadcasts over the symmetry axis t[:]
            np.outer(wP_r[r0:r1], C_i, out=t[:, :dr])

            cc_n += np.bincount(
                N_sri.ravel(), t[:, :dr].ravel(), minlength=cc_n.size
            )

    cl_code = cl.Program(context, code).build()

    # make sparse P_dr
    rs_d = []
    Ps_dr = []
    Nr = 0

    for d in tqdm(range(D), desc='generating sparse P-matrix'):
        rs = np.where(P_dr[d] > 0)[0]
        Ps_dr.append(P_dr[d][rs])
        rs_d.append(rs)
        Nr += len(rs)

    # it's possible that no frames contribute to a given class
    logger.debug(f'average number of tomograms per frame: {Nr/D:.3f}')
    if Nr == 0:
        logger.warning('This class has been abandoned by EMC! '
                       'Returning original model')
        return I_n

    logger.debug(f'minimum P value after threshold: '
                 f'{np.min([min(P) for P in Ps_dr if len(P) > 0])}')
    logger.debug(f'maximum P value after threshold: '
                 f'{np.max([max(P) for P in Ps_dr if len(P) > 0])}')

    for _ in tqdm(range(1), desc='calculating buffer size', disable=False):
        len_rs_d = np.array([len(r) for r in rs_d])
        len_pix_d = np.diff(K_di.frame_inds_indices)
        N = S * np.sum(len_rs_d * len_pix_d)

    # is there enough memory to store:
    #   a_dsri
    #   b_dsri
    #   n_dsri
    mem = 3 * 4 * N
    if mem > 0.8 * device.global_mem_size:
        raise ValueError(f'not enough memory to store a, b, n '
                         f'buffers on gpu {mem/1024**3:.2f} gb '
                         f'required {device.global_mem_size/1024**3:.2f} '
                         f'gb available')
    else:
        logger.debug(f'there is enough memory to store a, b, n buffers on gpu '
                     f'{mem/1024**3:.2f} gb required '
                     f'{device.global_mem_size/1024**3:.2f} gb available')

    # make a and b buffer, gpu or cpu? try gpu
    a_dsri_cl = cl.array.empty(queue, (N,), dtype=np.float32)
    b_dsri_cl = cl.array.empty(queue, (N,), dtype=np.float32)
    n_dsri_cl = cl.array.empty(queue, (N,), dtype=np.int32)

    P_cl = cl.array.empty(queue, (R,), dtype=np.float32)
    K_cl = cl.array.empty(queue, (I,), dtype=np.uint8)
    B_cl = cl.array.empty(queue, (I,), dtype=np.float32)
    C_cl = cl.array.empty(queue, (I,), dtype=np.float32)

    M_sri.cpu = False
    offset = np.int32(0)
    event = None
    for d in tqdm(range(D),
                  desc='filling buffers (looping over d)', disable=False):

        # get r inds with P_dr above threshold
        rs = rs_d[d]

        if len(rs) > 0:
            Kd_i, pixels = K_di.sparse(d)
            Bd_i = B_di.sparse(d, pixels)
            Cd_i = C_i[pixels]

            if event:
                event.wait()

            enq_events = []
            enq_events.append(cl.enqueue_copy(
                queue,
                K_cl.data,
                np.ascontiguousarray(Kd_i.astype(np.uint8)),
                is_blocking=False
            ))
            enq_events.append(cl.enqueue_copy(
                queue,
                B_cl.data,
                np.ascontiguousarray(Bd_i.astype(np.float32)),
                is_blocking=False
            ))
            enq_events.append(cl.enqueue_copy(
                queue,
                C_cl.data,
                np.ascontiguousarray(Cd_i.astype(np.float32)),
                is_blocking=False
            ))
            enq_events.append(cl.enqueue_copy(
                queue,
                P_cl.data,
                np.ascontiguousarray(Ps_dr[d].astype(np.float32)),
                is_blocking=False
            ))

            # (sym, rs, pixels)
            N_sri = M_sri[:, rs, pixels]

            event = cl_code.fill_buffer(
                queue,
                (len(pixels), len(rs), S),
                None,
                a_dsri_cl.data,
                b_dsri_cl.data,
                n_dsri_cl.data,
                P_cl.data,
                K_cl.data,
                B_cl.data,
                C_cl.data,
                np.float32(w_d[d]),
                N_sri,
                offset,
                np.int32(d),
                wait_for=[M_sri.event,] + enq_events
            )

            # offset += np.int32(N_sri.size)//4
            offset += np.int32(S * len(rs) * len(pixels))

    assert (offset == N)

    # now we need to fill the buffers
    # accounting for pixel perfect symmetry
    # problem is, if we do this all at once then
    # we use a lot of memory

    # hopefully memory is freed at each point
    a_dsri = np.empty(a_dsri_cl.size, dtype=np.float32)
    cl.enqueue_copy(queue, a_dsri, a_dsri_cl.data)
    del a_dsri_cl

    b_dsri = np.empty(b_dsri_cl.size, dtype=np.float32)
    cl.enqueue_copy(queue, b_dsri, b_dsri_cl.data)
    del b_dsri_cl

    n_dsri = np.empty(n_dsri_cl.size, dtype=np.int32)
    cl.enqueue_copy(queue, n_dsri, n_dsri_cl.data)
    del n_dsri_cl

    for _ in tqdm(range(1), desc='sorting buffers'):
        i = np.argsort(n_dsri)
        n_dsri = n_dsri[i].copy()
        a_dsri = a_dsri[i].copy()
        b_dsri = b_dsri[i].copy()

    # return (n_dsri, a_dsri, b_dsri)

    # why is this so slow? calling search sorted must have overhead
    # i_n[n] = starting index in a and b corresponding to voxel n
    # j_n[n] = ending   index in a and b corresponding to voxel n
    i_n = np.searchsorted(n_dsri, np.arange(I_n.size), side='left')
    j_n = np.searchsorted(n_dsri, np.arange(I_n.size), side='right')

    sym = symmetry.Symmetry(
        I_n.shape[0]//2,
        I_n.shape,
        symmetry=model_symmetry
    )

    n_asy = sym.get_asymmetric_unit()

    out_n = np.zeros(I_n.size, dtype=float)

    # storing the indexes for later slicing is
    # faster than using np.arange below
    inds0 = np.arange(n_dsri.size)

    for n in tqdm(n_asy, desc='solving for I'):
        ns = sym.get_symmetry_partners(n)
        inds = np.concatenate([inds0[i_n[ni]: j_n[ni]] for ni in ns])
        if len(inds) == 0:
            continue

        a_n = a_dsri[inds]
        b_n = b_dsri[inds]
        c_n = np.sum(cc_n[ns])

        out_n[n] = utils.solve_axbc(
            a_n, b_n, c_n, fill_value=0., ftol=1e-2, xtol=1e-3,
            maxiters=1000, algorithm='Halley', debug=False
        )

    out_n = symmetry.apply_symmetry(
        out_n.reshape(I_n.shape),
        model_symmetry,
        I_n.shape[0]//2
    )

    O_n = np.zeros(I_n.size, dtype=int)
    O_n[n_asy] = 1
    O_n = symmetry.apply_symmetry(
        O_n.reshape(I_n.shape),
        model_symmetry,
        I_n.shape[0]//2
    )

    O_n[O_n == 0] = 1
    out_n /= O_n

    return out_n


def w_update(P_cdr, K_di, B_di, W_cri, Wsums_cr, C_i):
    """
    assume sparse P_dr and K_di
    we need:
        a_j = {Pd_r Kd_i}         for non-zero elements
        b_j = {Bd_i / (C_i W_ri}  for above elements
        c_d = sum_r Pd_r Wsums_r

    calculate W_ri on cpu? give it a go

    wmax = sum_i K_di / sum_r P_dr Wsums_r
         = sum_i K_di / sum_r P_dr Wsums_r

    or calculate W_ri on gpu with r and i list
    """
    # we shouldn't need this as P_thresh is already
    # used to zero P_dr values where P_dr[d] < P_thresh max_r(P_dr[d])
    # P_thresh = 0.01
    D, I = K_di.shape
    C = len(P_cdr)

    assert (len(P_cdr) == len(Wsums_cr))

    w_d = np.zeros(D, dtype=float)

    for c in range(C):
        W_cri[c].cpu = True

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
    Ps_cdr = [[] for c in range(C)]
    Nr = 0
    for c in tqdm(range(C), desc='generating sparse P-matrix'):
        for d in range(D):
            rs = np.where(P_cdr[c][d] > 0)[0]
            Ps_cdr[c].append(P_cdr[c][d][rs])
            rs_cd[c].append(rs)
            Nr += len(rs)

    logger.debug(f'average number of tomograms per frame: {Nr/D:.3f}')

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

    # dominated by W_ri evaluation
    # could use threading to calculate W on different gpus simultaneously
    # or multiprocessing to call this funtion
    for d in tqdm(range(D), desc='updating w_d', disable=False):
        Kd_i, pixels = K_di.sparse(d)
        Bd_i = B_di.sparse(d, pixels)
        Cs_i = C_i[pixels]

        Psd_r = np.concatenate(
            [Ps_cdr[c][d] for c in range(C)]
        )

        Ws_ri = np.concatenate(
            [W_cri[c][rs_cd[c][d], pixels] for c in range(C)
             if len(rs_cd[c][d]) > 0]
        )
        Ws_ri = np.clip(Ws_ri, 1e-10, None)

        a = np.outer(Psd_r, Kd_i)
        b = Bd_i.astype(float) / (Cs_i * Ws_ri)

        a = np.ascontiguousarray(a.ravel().astype(np.float64))
        b = np.ascontiguousarray(b.ravel().astype(np.float64))

        w_d[d] = utils.solve_axbc(
            a, b, c_d[d], fill_value=0., ftol=1e-2, xtol=1e-3,
            maxiters=1000, algorithm='Newton', debug=False
        )

    return w_d
