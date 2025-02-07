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
import time
import pickle
import sys

import context
from emc2 import utils
from emc2 import utils_cl


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""calculate probability matrix"""
    )
    parser.add_argument(
        'class_files',
        type=str,
        nargs='+',
        help='file name class file(s)'
    )
    parser.add_argument(
        '--data_chunk',
        type=int,
        default=0,
        help='calculate for a subset of frames'
    )
    parser.add_argument(
        '--data_chunks',
        type=int,
        default=1,
        help='number of blocks to split frames over'
    )
    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('wb'),
        default=sys.stdout.buffer,
        help="Python pickle output file. \
            The result is written as a dictionary"
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
            //printf("       %e %e       ", t, logR_max);
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
            //t = exp(beta * (P_dr[d * R + r] - logR_max));
            //printf("  %d  ", r);
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

        //printf("            %d %d %d        ", d, R, rmax);

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
    P_thresh
):

    # load opencl on cpu then compile
    cl_cpu = utils_cl.opencl_init_cpu()
    code = get_code()
    cl_cpu_code = cl.Program(cl_cpu['context'], code).build()

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

    d_chunk_size = 8
    d_iter = tqdm(
        utils.chunker(d_chunk_size, D),
        desc='calculating P_dr from logR'
    )

    P_dr = np.empty_like(logR_dr)

    assert (class_r.shape == (R,))
    assert (logR_dr.dtype == np.float64)

    for d0, d1, dd in d_iter:
        cl_cpu_code.normalise_P_dr(
            cl_cpu['queue'],
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

    cl_cpu['queue'].finish()

    return P_dr, rmax_d, Pmax_d, occupancy_dc, Q_d


if __name__ == '__main__':
    """
    Load P_dr chunked over frames from each class file
    """
    args = get_args()

    working_directory = Path(args.class_files[0]).parent

    # get number of frames
    # get number of r's
    Rs = []
    Ds = []
    for fnam in args.class_files:
        with h5py.File(fnam, 'r') as f:
            D, R = f['probability_matrix'].shape
            beta = f['beta'][()]
            P_thresh = f['P_thresh'][()]
            Rs.append(R)
            Ds.append(D)

    assert (np.allclose(Ds, D))
    R = np.sum(Rs)

    # get frames to process
    d0, d1, dd = utils.chunker_mpi(args.data_chunks, D)
    d0, d1, dd = d0[args.data_chunk], d1[args.data_chunk], dd[args.data_chunk]

    # Load P_dr for frame selection
    logR_dr = np.empty((dd, R), dtype=np.float64)
    class_r = np.empty((R,), dtype=np.int32)

    index = 0
    for c, fnam in enumerate(args.class_files):
        with h5py.File(fnam) as f:
            r0, r1 = index, index + Rs[c]
            logR_dr[: dd, r0: r1] = f['probability_matrix'][d0: d1]
            class_r[r0: r1] = f['class_id'][...]
            index = r1

    # normalise and calculate
    P_dr, rmax_d, Pmax_d, occupancy_dc, Q_d = normalise_P_dr(
        logR_dr,
        class_r,
        beta,
        P_thresh
    )

    # pipe to std out
    index = 0
    file = sys.stdout.buffer
    for c, fnam in enumerate(args.class_files):
        r0, r1 = index, index + Rs[c]
        msg = {
            'file': fnam,
            'mode': 'r+',
            'probability_matrix': {
                'slice': slice(d0, d1),
                'data': P_dr[:dd, r0: r1]
            }
        }
        pickle.dump(msg, file)
