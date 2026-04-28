import numpy as np
import pyopencl as cl
import os
from pathlib import Path
import h5py

from tqdm import tqdm
from .utils_cl import opencl_init_cpu
from .utils import chunker
from . import input_output

from scipy.ndimage import gaussian_filter1d


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
        global double *Q_old_d,
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

        // calculate P . logR using old P-values
        // Q = sum_r P_dr logR_dr
        for (r=0; r<R; r++) {{
            Q_old_d[d] += P_dr[d * R + r] * logR_dr[d * R + r];
        }}

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
        self.Q_old_d = np.zeros(D, dtype=float)
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

    def calculate(self, P_dr, logR_dr, d00, d11):
        """
        d00 and d11 are global indices
        P_dr and logR_dr are chunked arrays

        e.g. P_dr[:d11-d00] will be set
        """
        D = d11-d00
        R = logR_dr.shape[1]

        # per chunk arrays
        # I would love to move the chunking logic elsewhere
        rmax_d = np.zeros(D, dtype=int)
        Pmax_d = np.zeros(D, dtype=float)
        occupancy_dc = np.zeros((D, self.models), dtype=float)
        Q_d = np.zeros(D, dtype=float)
        Q_old_d = np.zeros(D, dtype=float)

        # d_chunk_size = max(1, int(os.cpu_count()/2))
        d_chunk_size = max(1, int(os.cpu_count()/2))
        d_chunk_size = min(d_chunk_size, D)

        d_iter = tqdm(
            chunker(d_chunk_size, D),
            desc='calculating P_dr from logR',
            disable=True
        )

        if P_dr is None:
            P_dr = np.zeros_like(logR_dr)

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
                cl.SVM(Q_old_d),
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
        self.Q_old_d[d00:d11] = Q_old_d

        return P_dr

    def save_P_dr(self, fnams, update_probability_c, d0, d1, P_dr):
        index = 0
        for c, fnam in enumerate(fnams):
            r0, r1 = index, index + self.Rs[c]
            if update_probability_c[c]:
                with h5py.File(fnam, 'r+') as f:
                    f['P_dr'][d0:d1, :] = P_dr[:d1-d0, r0:r1]
                    if 'beta' not in f:
                        f['beta'] = self.beta
            else:
                print('update_probability is False '
                     f'skipping update for class {c}')
            index = r1

def continuity(P_dr, config):
    """test: smooth P_dr across classes

    only works if each class has the same number of rotations

    preserves normalisation along r for each d

    config['continuity_groups'] = [
        {
            'classes': [1,2,3],
            'sigma': 0.7,
        },
        ...
    ]
    """
    if 'continuity_groups' not in config:
        return

    if not config['continuity_groups']:
        return

    # get P_dcr for each continuity group
    rs_c = np.array([c['r_offset'] for c in config['classes']] + [config['R'],])
    Rs_c = np.diff(rs_c)

    for group in config['continuity_groups']:
        cs = group['classes']
        sigma = group['sigma']
        Rg = Rs_c[cs][0]

        # make sure rotation sampling is the same for all classes
        assert(np.all(Rs_c[cs] == Rg))

        # fill P_dcr matrix
        P_dcr = np.empty((P_dr.shape[0], len(cs), Rs_c[0]), dtype=float)
        for i, ci in enumerate(cs):
            r0 = rs_c[ci]
            r1 = r0 + Rg
            P_dcr[:, i, :] = P_dr[:, r0: r1]

        # smooth along class axis P_dcr
        P_dcr = gaussian_filter1d(P_dcr, sigma, mode='reflect', axis=1)

        # fill P_dr
        for i, ci in enumerate(cs):
            r0 = rs_c[ci]
            r1 = r0 + Rg
            P_dr[:, r0: r1] = P_dcr[:, i, :]

    # P_dcr = P_dr.reshape((dd, len(config['classes']), -1))
    # P_dcr = gaussian_filter1d(P_dcr, 0.7, mode='reflect', axis=1)
    # P_dr[:] = P_dcr.reshape((dd, -1))

def load_logR_dr_chunks(d0, d1, config, logR_dr):
    """
    load logR_dr d-chunk from each class into a single array

    data is written to logR_dr
    """
    for ci, c in enumerate(config['classes']):
        r0 = c['r_offset']  # global offset for this class
        R_c = c['mapper'].shape[1]  # number of r's for this class
        with h5py.File(c['logR_file']) as f:
            logR_dr[:d1-d0, r0:r0+R_c] = f['logR_dr'][d0:d1]

def load_P_dr_chunks(d0, d1, config, P_dr):
    """
    load P_dr d-chunk from each class into a single array

    only load P_dr from a class if it is
    not going to be updated (for continuity)
    """
    for ci, c in enumerate(config['classes']):
        r0 = c['r_offset']          # global offset for this class
        R_c = c['mapper'].shape[1]  # number of r's for this class
        fnam = c['probability_matrix_file']
        if c['update_probability']:
            with h5py.File(fnam) as f:
                P_dr[:d1-d0, r0:r0+R_c] = f['P_dr'][d0:d1]


def initialise_P_dr_files_if_needed(config):
    """
    If P_dr matrix file does not exist, or has the wrong shape
    then initialise the file for writing in chunks
    """
    for ci, c in enumerate(config['classes']):
        fnam = c['probability_matrix_file']

        # class (not global) D, R
        D = c['data'].shape[0]
        R = c['mapper'].shape[1]

        p_init = True
        if Path(fnam).is_file():
            with h5py.File(fnam) as f:
                if f['P_dr'].shape == (D, R):
                    p_init = False

        if not c['update_probability']:
            p_init = False

        # initialise probability files if needed
        if p_init:
            with h5py.File(fnam, 'w') as f:
                f.create_dataset('P_dr', shape=(D, R), dtype=float)


def get_class_index_r(config):
    # calculate class index for each global r-index
    # this is needed only for occupancy_dc
    # ---------------------------------------------
    class_r = np.empty(config['R'])
    for ci, c in enumerate(config['classes']):
        R_c = c['mapper'].shape[1]
        r0 = c['r_offset']
        class_r[r0:r0+R_c] = ci
    return class_r

def keep_n_frames(P_dr, config):
    """
    keep the highest 'f' terms in P_dr for each frame (d)
    remaining terms are set to zero

    where:
        f = config['P_thresh_frames']
    """
    # test threshold
    key = 'P_thresh_frames'
    R = P_dr.shape[1]
    if key in config and config[key]:
        # at most f r's per frame
        f = config[key]
        p = (1-f/R)*100
        thresh_d = np.percentile(P_dr, p, axis=1)
        m = P_dr > thresh_d[:, None]
        P_dr *= m

        # renormalise
        P_dr /= np.sum(P_dr, axis=1)[:, None]


def calculate_P(config, beta):
    """
    calculate P_dr from logR_dr in d-chunks

    load logR_dr[d0:d1] for each class
    calculate P_dr[d0:d1]
    save
    continue
    """
    R = config['R']
    D = config['classes'][0]['P_data'].shape[0]

    # Hack: this should be set for each class independently
    # but for now is gloabal
    P_thresh = config['classes'][0]['P_thresh']

    # calculate chunksize ~2Gb
    # ------------------------
    mem = 2 * 1024**3
    d_chunk_size = max(1, int(mem / (8 * R)))
    d_chunk_size = min(d_chunk_size, D)

    logR_dr_chunk = np.empty((d_chunk_size, R))
    P_dr_chunk    = np.zeros((d_chunk_size, R))
    class_r       = get_class_index_r(config)
    fnams_c       = [c['probability_matrix_file'] for c in config['classes']]
    update_probability_c = [c['update_probability'] for c in config['classes']]

    initialise_P_dr_files_if_needed(config)

    # calculate all probabilities (normalise logR)
    prob = Probability(D, R, beta, class_r, P_thresh)

    # loop over d-chunks
    # ------------------
    for d0, d1, dd in chunker(d_chunk_size, D):
        load_logR_dr_chunks(d0, d1, config, logR_dr_chunk)
        load_P_dr_chunks(d0, d1, config, P_dr_chunk)

        prob.calculate(P_dr_chunk, logR_dr_chunk, d0, d1)

        # post processing
        continuity(P_dr_chunk[:dd], config)
        keep_n_frames(P_dr_chunk[:dd], config)

        # save P_dr_chunk in class files
        prob.save_P_dr(fnams_c, update_probability_c, d0, d1, P_dr_chunk)

    config['most_likely_model_d'] = prob.class_max_d

    return {
        'P_max_d':       prob.Pmax_d,
        'Q_d':           prob.Q_d,
        'Q_old_d':       prob.Q_old_d,
        'class_max_d':   prob.class_max_d,
        'local_rmax_d':  prob.local_rmax_d,
        'occupancy_dc':  prob.occupancy_dc,
        'occupancy_r':   prob.occupancy_r,
        'beta':          beta,
    }
