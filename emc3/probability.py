import numpy as np
import pyopencl as cl
import os
from pathlib import Path
import h5py

from tqdm import tqdm
from .utils_cl import opencl_init_cpu
from .utils import chunker
from . import input_output

import warnings
# warnings.filterwarnings("ignore", category=cl.CompilerWarning)
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

cl_code = """
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

class Probability():
    """
    Normalise the log-likelihood array over all r's to produce a
    probability distribution:
        P_dr = self.calculate(logR_dr)

        P_dr = R_dr / sum_r R_dr
        where
        R_dr = exp^{beta logR_dr}

    logR_dr can be chunked by frame but must span the entire r domaine.
    """
    def __init__(self, D, R, beta=1, class_r=None, P_thresh=0):
        # this could be per class
        self.beta = np.float64(beta)
        self.P_thresh = np.float64(P_thresh)

        if class_r is None:
            self.class_r = np.zeros(R, dtype=np.int32)
        else:
            self.class_r = np.ascontiguousarray(class_r.astype(np.int32))

        self.Rs = np.bincount(self.class_r)

        self.models = np.int32(np.max(self.class_r)+1)
        self.occupancy_r = np.zeros((R,), dtype=np.float32)

        self.rmax_d = np.zeros(D, dtype=int)
        self.Pmax_d = np.zeros(D, dtype=float)
        self.occupancy_dc = np.zeros((D, self.models), dtype=float)
        self.Q_d = np.zeros(D, dtype=float)
        self.class_max_d = np.empty((D,), dtype=np.int32)
        self.local_r = np.empty((R,), dtype=np.int32)
        self.local_rmax_d = np.empty((D,), dtype=np.int32)

        index = 0
        for c in range(len(self.Rs)):
            r0, r1 = index, index + self.Rs[c]
            self.local_r[r0: r1] = np.arange(self.Rs[c])
            index = r1

        # load opencl on cpu then compile
        cl_cpu = opencl_init_cpu()
        self.queue = cl_cpu['queue']
        self.context = cl_cpu['context']
        self.cl_cpu_code = cl.Program(self.context, cl_code).build()
        self.normalise_P_dr = cl.Kernel(self.cl_cpu_code, "normalise_P_dr")

    def calculate(self, logR_dr, d00, d11):
        D, R = logR_dr.shape

        # per chunk arrays
        # I would love to move the chunking logic elsewhere
        rmax_d = np.zeros(D, dtype=int)
        Pmax_d = np.zeros(D, dtype=float)
        occupancy_dc = np.zeros((D, self.models), dtype=float)
        Q_d = np.zeros(D, dtype=float)

        # d_chunk_size = max(1, int(os.cpu_count()/2))
        d_chunk_size = max(1, int(os.cpu_count()/2))
        d_chunk_size = min(d_chunk_size, D)

        d_iter = tqdm(
            chunker(d_chunk_size, D),
            desc='calculating P_dr from logR',
            disable=True
        )

        P_dr = np.empty_like(logR_dr)

        assert (self.class_r.shape == (R,))
        assert (logR_dr.dtype == np.float64)
        assert (logR_dr.flags['C_CONTIGUOUS'])

        for d0, d1, dd in d_iter:
            self.normalise_P_dr(
                self.queue,
                (dd,),
                None,
                cl.SVM(logR_dr),
                cl.SVM(P_dr),
                cl.SVM(self.class_r),
                cl.SVM(rmax_d),
                cl.SVM(Pmax_d),
                cl.SVM(occupancy_dc),
                cl.SVM(Q_d),
                self.beta,
                self.P_thresh,
                np.int32(d0),
                self.models,
                np.int32(R),
            )

        self.queue.finish()

        dd = d11-d00
        assert (np.all(np.isfinite(P_dr[:dd])))
        assert (np.allclose(np.sum(P_dr[:dd], axis=1), 1.))

        self.occupancy_r += np.sum(P_dr[:dd], axis=0)
        self.P_dr = P_dr

        self.class_max_d[d00:d11] = self.class_r[rmax_d]
        self.local_rmax_d[d00:d11] = self.local_r[rmax_d]
        self.Pmax_d[d00:d11] = Pmax_d
        self.occupancy_dc[d00:d11] = occupancy_dc
        self.Q_d[d00:d11] = Q_d

        return P_dr

    def save_P_dr(self, fnams, update_probability_c, d0, d1):
        index = 0
        for c, fnam in enumerate(fnams):
            r0, r1 = index, index + self.Rs[c]
            if update_probability_c[c]:
                with h5py.File(fnam, 'r+') as f:
                    f['P_dr'][d0:d1, :] = self.P_dr[:d1-d0, r0:r1]
                    if 'beta' not in f:
                        f['beta'] = self.beta
            else:
                logger.info('update_probability is False '
                            f'skipping update for class {c}')
            index = r1

    def save_iteration(self, working_directory):
        input_output.save_iteration_info(
            self.Pmax_d,
            self.Q_d,
            self.class_max_d,
            self.local_rmax_d,
            self.occupancy_dc,
            self.occupancy_r,
            working_directory,
            self.beta
        )


def calculate_P(config, beta):
    """
    basic all in memory cpu process
    """
    R = config['R']
    D = config['classes'][0]['P_data'].shape[0]

    class_r = np.empty(R)
    logR_dr = np.empty((D, R))
    update_probability_c = np.empty(len(config['classes']))

    P_thresh = config['classes'][0]['P_thresh']

    Rs = []
    for ci, c in enumerate(config['classes']):
        R_c = c['mapper'].shape[1]
        Rs.append(R_c)
        r0 = c['r_offset']
        class_r[r0:r0+R_c] = ci

        # load logR for class ci
        with h5py.File(c['logR_file']) as f:
            logR_dr[:, r0:r0+R_c] = f['logR_dr'][()]
            D, R = f['logR_dr'].shape

        update_probability_c[ci] = c['update_probability']

        # initialise probability files
        if c['update_probability']:
            with h5py.File(c['probability_matrix_file'], 'w') as f:
                f.create_dataset('P_dr', shape=(D, R), dtype=float)

    fnams = [c['probability_matrix_file'] for c in config['classes']]

    # calculate all probabilities (normalise logR)
    prob = Probability(D, np.sum(Rs), beta, class_r, P_thresh)

    # calculate frame chunksize ~2G
    D, R = logR_dr.shape
    mem = 2 * 1024**3
    d_chunk_size = max(1, int(mem / (8 * R)))
    d_chunk_size = min(d_chunk_size, D)

    for d0, d1, dd in chunker(d_chunk_size, D):
        P_dr = prob.calculate(logR_dr[d0:d1], d0, d1)

        # save in class files
        prob.save_P_dr(fnams, update_probability_c, d0, d1)

    # save extra data in iteration file
    prob.save_iteration(config['working_directory'])

    config['most_likely_model_d'] = prob.class_max_d
