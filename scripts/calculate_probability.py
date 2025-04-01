"""
normalse logR to calculate probability matrix
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
import pyopencl as cl
import pyopencl.array
import os

import context
from emc2 import utils
from emc2 import utils_cl
from emc2 import input_output
from emc2 import data_getter
from emc2 import get_script_logger


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""calculate probability matrix"""
    )
    parser.add_argument(
        'config',
        type=str,
        help='config file name'
    )
    parser.add_argument(
        'iteration',
        type=int,
        help='iteration number'
    )
    parser.add_argument(
        '--frame_chunk_size',
        type=int,
        default=4096,
        help='loop over frame chunks to reduce memory consumption'
    )
    args = parser.parse_args()
    return args


def get_code():
    return """
    // optimised for cpu with one worker per d
    __kernel void normalise_P_dr (
        global double *logR_dr,
        global double *P_dr,
        global int    *class_r,
        global long   *rmax_d,
        global double *Pmax_d,
        global double *occupancy_dc,
        global double *Q_d,
        const double beta,
        const double P_thresh,
        const int d_offset,
        const int C,
        const int R
    ) {{
        int d = d_offset + get_global_id(0);

        double t, thresh;
        int r, rmax;

        double logR_max = -DBL_MAX;

        // find argmax and max of logR_dr
        for (r=0; r<R; r++) {{
            t = logR_dr[d * R + r];
            if (t > logR_max){{
                rmax = r;
                logR_max = t;
            }}
            P_dr[d * R + r] = t;
        }}

        rmax_d[d] = (long)rmax;

        // calculate
        // P_dr = exp( beta * (logR - logRmax))
        for (r=0; r<R; r++) {{
            P_dr[d * R + r] = exp(beta * (P_dr[d * R + r] - logR_max));
        }}

        // threshold
        if (P_thresh > 0.) {{
            thresh = P_thresh * P_dr[d * R + rmax] ;
            for (r=0; r<R; r++) {{
                if (P_dr[d * R + r] < thresh)
                    P_dr[d * R + r] = 0.;
            }}
        }}

        // normalise sum_r P_dr to 1
        t = 0.;
        for (r=0; r<R; r++) {{
            t += P_dr[d * R + r];
        }}
        for (r=0; r<R; r++) {{
            P_dr[d * R + r] /= t;
        }}

        Pmax_d[d] = P_dr[d * R + rmax];

        // calculate occupancy_dc and Q
        // Q = sum_r P_dr logR_dr
        for (r=0; r<R; r++) {{
            occupancy_dc[d * C + class_r[r]] += P_dr[d * R + r];
            Q_d[d] += P_dr[d * R + r] * logR_dr[d * R + r];
        }}
    }}
    """


def normalise_P_dr(
    logR_dr,
    class_r,
    beta,
    P_thresh,
    cl_cpu_code,
    queue
):
    D, R = logR_dr.shape

    models = np.max(class_r)+1

    # normalise
    rmax_d = np.zeros(D, dtype=int)
    Pmax_d = np.zeros(D, dtype=float)
    occupancy_dc = np.zeros((D, models), dtype=float)
    Q_d = np.zeros(D, dtype=float)
    beta = np.float64(beta)
    P_thresh = np.float64(P_thresh)
    class_r = np.ascontiguousarray(class_r.astype(np.int32))

    d_chunk_size = max(1, int(os.cpu_count()/2))
    d_iter = tqdm(
        utils.chunker(d_chunk_size, D),
        desc='calculating P_dr from logR',
        disable=True
    )

    P_dr = np.empty_like(logR_dr)

    assert (class_r.shape == (R,))
    assert (logR_dr.dtype == np.float64)

    for d0, d1, dd in d_iter:
        cl_cpu_code.normalise_P_dr(
            queue,
            (dd,),
            None,
            cl.SVM(logR_dr),
            cl.SVM(P_dr),
            cl.SVM(class_r),
            cl.SVM(rmax_d),
            cl.SVM(Pmax_d),
            cl.SVM(occupancy_dc),
            cl.SVM(Q_d),
            beta,
            P_thresh,
            np.int32(d0),
            np.int32(models),
            np.int32(R),
        )

    queue.finish()

    return P_dr, rmax_d, Pmax_d, occupancy_dc, Q_d


if __name__ == '__main__':
    """
    Load P_dr chunked over frames from each class file
    """
    args = get_args()

    working_directory = Path(args.config).parent

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('calculate_probability (start)')

    # load config file
    config = input_output.load_config(args.config)

    models = len(config['classes'])

    class_files = [
        Path(working_directory).joinpath(f'class_{c}.h5')
        for c in range(models)
    ]

    # get beta
    config['iteration'] = args.iteration
    beta = utils.get_beta(**config)

    # get number of frames
    # get number of r's
    Rs = []
    Ds = []

    # global_r --> class_id, local_r
    for fnam in class_files:
        with h5py.File(fnam, 'r') as f:
            D, R = f['probability_matrix'].shape
            P_thresh = f['P_thresh'][()]
            Rs.append(R)
            Ds.append(D)

    assert (np.allclose(Ds, D))
    R = np.sum(Rs)

    args.frame_chunk_size = min(args.frame_chunk_size, D)

    d_iter = tqdm(
        utils.chunker(args.frame_chunk_size, D),
        desc='calculating P_dr over frame chunks'
    )

    logR_dr = np.empty((args.frame_chunk_size, R), dtype=np.float64)
    class_r = np.empty((R,), dtype=np.int32)
    local_r = np.empty((R,), dtype=np.int32)
    local_rmax_d = np.empty((D,), dtype=np.int32)
    class_max_d = np.empty((D,), dtype=np.int32)
    Pmax_d = np.empty((D,), dtype=np.float32)
    occupancy_dc = np.empty((D, models), dtype=np.float32)
    occupancy_r = np.zeros((R,), dtype=np.float32)
    Q_d = np.empty((D,), dtype=np.float32)

    index = 0
    for c, fnam in enumerate(class_files):
        with h5py.File(fnam) as f:
            r0, r1 = index, index + Rs[c]
            class_r[r0: r1] = f['class_id'][...]
            local_r[r0: r1] = np.arange(Rs[c])
            index = r1

    # load opencl on cpu then compile
    cl_cpu = utils_cl.opencl_init_cpu()
    code = get_code()
    cl_cpu_code = cl.Program(cl_cpu['context'], code).build()

    for d0, d1, dd in d_iter:
        # Load P_dr for frame selection
        index = 0
        for c, fnam in enumerate(class_files):
            with h5py.File(fnam) as f:
                r0, r1 = index, index + Rs[c]
                logR_dr[: dd, r0: r1] = f['probability_matrix'][d0: d1]
                index = r1

        # normalise and calculate
        P_dr, rmax_d_chunk, Pmax_d_chunk, occupancy_dc_chunk, Q_d_chunk \
            = normalise_P_dr(
                logR_dr[:dd],
                class_r,
                beta,
                P_thresh,
                cl_cpu_code,
                cl_cpu['queue']
            )

        assert (np.all(np.isfinite(P_dr[:dd])))

        class_max_d[d0:d1] = class_r[rmax_d_chunk]
        local_rmax_d[d0:d1] = local_r[rmax_d_chunk]
        Pmax_d[d0:d1] = Pmax_d_chunk
        occupancy_dc[d0:d1] = occupancy_dc_chunk
        Q_d[d0:d1] = Q_d_chunk
        occupancy_r += np.sum(P_dr, axis=0)

        # save P_dr to class files
        index = 0
        for c, fnam in enumerate(class_files):
            r0, r1 = index, index + Rs[c]
            with h5py.File(fnam, 'r+') as f:
                update_probability = f['update_probability'][()]
                if update_probability:
                    f['probability_matrix'][d0:d1] = P_dr[:dd, r0:r1]
                    f['beta'][...] = beta
                else:
                    logger.info('update_probability is False '
                                f'skipping update for class {c}')
            index = r1

    # get sparse fnam (must be an easier way to do this...)
    with h5py.File(class_files[0]) as f:
        sparse_fnam = data_getter.get_sparse_fnam(
            f['cxi_file'][()].decode(),
            working_directory,
            f['mask'][()]
        )

    # write to iteration info
    utils.save_iteration_info(
        Pmax_d,
        Q_d,
        class_max_d,
        local_rmax_d,
        occupancy_dc,
        occupancy_r,
        working_directory,
        beta,
        sparse_fnam
    )
    logger.info('calculate_probability (stop)')
