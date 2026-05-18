"""
Common-lines analysis for 2D EMC class images.

Pipeline
--------
1. calculate_lines  – extract sinograms (line profiles + masks) from class
                      model files and store them back into those files.
2. compare_lines    – load sinograms, compute the weighted Pearson correlation
                      surface C[i,j] for every class pair, and write the
                      condensed (N_pairs, N_angles, N_angles) array to HDF5.
3. build_consistency_matrix – reduce the condensed C array to an
                              N_classes × N_classes score matrix.

Angles sweep [0, 2π) — no Friedel / centrosymmetry assumption.
"""

import numpy as np
import pyopencl as cl
import pyopencl.array
import h5py
from pathlib import Path

from .utils_cl import opencl_init, to_gpu_image
from . import profiling
from . import input_output


# ---------------------------------------------------------------------------
# Sinogram extraction (OpenCL)
# ---------------------------------------------------------------------------

class Mapper_lines:
    """
    Extract radial line profiles from a 2-D image using OpenCL texture sampling.

    Parameters
    ----------
    shape   : (H, W) image shape (must be square, H == W)
    i0      : centre pixel coordinate (float)
    Nrot    : number of azimuthal angles, uniformly spanning [0, 2π)
    Nline   : number of samples along each line
    queue, context : OpenCL objects
    interp  : 'linear' or 'nearest'
    """

    def __init__(self, shape, i0, Nrot, Nline, queue, context,
                 interp='linear'):
        self.i0      = np.float32(i0)
        self.Nrot    = np.int32(Nrot)
        self.Nline   = np.int32(Nline)
        self.queue   = queue
        self.context = context
        self.shape   = (int(Nrot), int(Nline))   # output sinogram shape

        # sample positions along each line  (-half … +half)
        half = shape[0] / 2.0
        self.t_i = np.linspace(-half, half, Nline,
                               endpoint=True, dtype=np.float32)

        self.W_ri = np.empty((Nrot, Nline), dtype=np.float32)
        self.M_ri = np.empty((Nrot, Nline), dtype=np.float32)

        mf = cl.mem_flags
        self.t_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=self.t_i)
        self.W_cl = cl.Buffer(context, mf.READ_WRITE, self.W_ri.nbytes)
        self.M_cl = cl.Buffer(context, mf.READ_WRITE, self.M_ri.nbytes)

        interpolation = interp.upper()

        # Angles span [0, 2π) — no Friedel assumption
        code = r"""
        constant sampler_t interpolation =
            CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP |
            CLK_FILTER_{interpolation};

        __kernel void tomo(
            global float *W_ri,
            global float *M_ri,
            __read_only image2d_t I_im,
            __read_only image2d_t M_im,
            global float *t_i,
            const float i0,
            const int I
        )
        {{
            long r = get_global_id(0);
            long R = get_global_size(0);

            /* angle in [0, 2π) */
            float theta = r * 2.0f * M_PI / (float)R;
            float c = cos(theta);
            float s = sin(theta);

            for (int i = 0; i < I; i++) {{
                float2 coord;
                coord.x = t_i[i] * c + i0 + 0.5f;
                coord.y = t_i[i] * s + i0 + 0.5f;

                float4 W = read_imagef(I_im, interpolation, coord);
                float4 M = read_imagef(M_im, interpolation, coord);

                W_ri[r * I + i] = W.x;
                M_ri[r * I + i] = M.x;
            }}
        }}
        """

        self.prg = cl.Program(
            context,
            code.format(interpolation=interpolation)
        ).build()

    def load_image(self, data):
        self.I_im = to_gpu_image(data.astype(np.float32), self.queue,
                                 self.context)
        self.M_im = to_gpu_image((data > 0.).astype(np.float32), self.queue,
                                 self.context)

    def _run(self):
        cl.Kernel(self.prg, 'tomo')(
            self.queue,
            (int(self.Nrot),), None,
            self.W_cl, self.M_cl,
            self.I_im, self.M_im,
            self.t_cl,
            self.i0,
            self.Nline,
        )

    def __call__(self, data=None):
        """Return (W_ri, M_ri) sinograms, shape (Nrot, Nline)."""
        if data is not None:
            self.load_image(data)
        self._run()
        cl.enqueue_copy(self.queue, self.W_ri, self.W_cl)
        cl.enqueue_copy(self.queue, self.M_ri, self.M_cl)
        self.queue.finish()
        return self.W_ri.copy(), self.M_ri.copy()


# ---------------------------------------------------------------------------
# Step 1 – extract sinograms and store in model files
# ---------------------------------------------------------------------------

def calculate_lines(config, Nrot=360, Nline=None, device=0):
    """
    Extract (Nrot, Nline) sinograms for every 2D class and save them into
    the class model file under datasets 'lines' and 'lines_mask'.

    Parameters
    ----------
    config  : dict  (the EMC config dict, not a filename)
    Nrot    : number of azimuthal angles spanning [0, 2π)
    Nline   : samples per line; defaults to model image width
    device  : OpenCL device index
    """
    cl_stuff = opencl_init(device_no=device)

    mapper = None
    last_key = None

    for c in config['classes']:
        if c['model'].ndim != 2:
            continue

        with h5py.File(c['model_file']) as f:
            data = f['data'][()]

        H, W = data.shape
        nline = Nline if Nline is not None else W
        key   = (H, W, c['model'].i0, Nrot, nline)

        if key != last_key:
            mapper   = Mapper_lines(data.shape, c['model'].i0,
                                    Nrot, nline,
                                    cl_stuff['queue'], cl_stuff['context'])
            last_key = key

        lines, lines_mask = mapper(data)

        with h5py.File(c['model_file'], 'r+') as f:
            input_output.write_h5(f, 'lines',       lines,      compression=True)
            input_output.write_h5(f, 'lines_mask',  lines_mask, compression=True)


# ---------------------------------------------------------------------------
# Step 2 – compute the full correlation surface for every class pair
# ---------------------------------------------------------------------------

def pair_idx(i, j, N):
    """Index into condensed upper-triangle storage (i < j required)."""
    return i * N - i * (i + 1) // 2 + j - i - 1


def common_line_correlations(A, B, mA, mB):
    """
    Weighted Pearson correlation for all (N_angles_A × N_angles_B) line pairs.

    Parameters
    ----------
    A, B   : (N_angles, N_samples) – line profiles
    mA, mB : (N_angles, N_samples) – soft masks in [0, 1]

    Returns
    -------
    C : (N_angles_A, N_angles_B) – weighted Pearson correlations.
        NOT symmetric when A != B.
    """
    mA_A = mA * A
    mB_B = mB * B

    W   = mA   @ mB.T        # Σ w
    WAB = mA_A @ mB_B.T      # Σ w·x·y
    WA  = mA_A @ mB.T        # Σ w·x
    WB  = mA   @ mB_B.T      # Σ w·y
    WA2 = (mA * A**2) @ mB.T # Σ w·x²
    WB2 = mA @ (mB * B**2).T # Σ w·y²

    num   = W * WAB - WA * WB
    varA  = W * WA2 - WA**2
    varB  = W * WB2 - WB**2
    denom = np.sqrt(np.maximum(varA * varB, 0.0))
    return np.where(denom > 0, num / denom, 0.0)


def compare_lines(config, power=1., output_file=None):
    """
    Load sinograms from model files and compute weighted Pearson correlation
    surfaces C[i,j] for all (class_a, class_b) pairs with class_a < class_b.

    Results are written to *output_file* (default: working_directory/common_lines.h5)
    as a condensed (N_pairs, N_angles_a, N_angles_b) dataset.

    Also writes:
    - 'scores'    : (N_classes, N_classes) – max correlation per pair
    - 'best_phi'  : (N_classes, N_classes, 2) – argmax (angle_a, angle_b)

    Parameters
    ----------
    config      : dict
    output_file : str or Path, optional
    """
    classes_2d = [
        (ci, c)
        for ci, c in enumerate(config['classes'])
        if c['model'].ndim == 2
    ]

    if len(classes_2d) < 2:
        raise ValueError('Need at least two 2D classes for common-lines comparison')

    # load sinograms
    lines_c      = []
    lines_mask_c = []
    valid_ids    = []

    for ci, c in classes_2d:
        with h5py.File(c['model_file'], 'r') as f:
            if 'lines' not in f:
                print(f'  class {ci}: no sinogram found, run calculate_lines first')
                continue
            lines_c.append(f['lines'][()])
            lines_mask_c.append(f['lines_mask'][()])
            valid_ids.append(ci)

    N = len(valid_ids)
    if N < 2:
        raise ValueError('Fewer than 2 classes have sinograms')

    Nrot_a, Nline = lines_c[0].shape

    if output_file is None:
        output_file = Path(config['working_directory']) / 'common_lines.h5'
    output_file = Path(output_file)

    N_pairs = N * (N - 1) // 2
    Nrot_b  = lines_c[0].shape[0]   # same for all classes (same Mapper_lines)

    scores      = np.ones((N, N), dtype=np.float32)
    best_phi    = np.zeros((N, N, 2), dtype=np.int32)

    with h5py.File(output_file, 'w') as f:
        f['class_ids'] = np.array(valid_ids, dtype=np.int32)
        f['N_classes'] = N
        f['Nrot']      = np.int32(Nrot_a)

        # condensed (N_pairs, Nrot_a, Nrot_b) correlation surfaces
        C_ds = f.create_dataset(
            'C',
            shape=(N_pairs, Nrot_a, Nrot_b),
            dtype=np.float32,
            chunks=(1, Nrot_a, Nrot_b),
            compression='gzip',
        )
        # best-fit line profiles: shape (N_pairs, Nline) for each class
        best_lines_a_ds = f.create_dataset(
            'best_lines_a',
            shape=(N_pairs, Nline),
            dtype=np.float32,
            chunks=(1, Nline),
            compression='gzip',
        )
        best_lines_b_ds = f.create_dataset(
            'best_lines_b',
            shape=(N_pairs, Nline),
            dtype=np.float32,
            chunks=(1, Nline),
            compression='gzip',
        )

        for i in range(N):
            A  = lines_c[i].astype(np.float32)
            mA = lines_mask_c[i].astype(np.float32)

            for j in range(i + 1, N):
                B  = lines_c[j].astype(np.float32)
                mB = lines_mask_c[j].astype(np.float32)

                if power != 1.:
                    C = common_line_correlations(A**power, B**power, mA, mB)
                else:
                    C = common_line_correlations(A, B, mA, mB)

                k = pair_idx(i, j, N)
                C_ds[k] = C.astype(np.float32)

                best_k, best_l = np.unravel_index(np.argmax(C), C.shape)
                score = float(C[best_k, best_l])

                scores[i, j] = score
                scores[j, i] = score
                best_phi[i, j] = [best_k, best_l]
                best_phi[j, i] = [best_l, best_k]

                best_lines_a_ds[k] = A[best_k]
                best_lines_b_ds[k] = B[best_l]

                print(f'  pair ({valid_ids[i]}, {valid_ids[j]}): '
                      f'max_corr={score:.4f}  angles=({best_k}, {best_l})')

        f['scores']   = scores
        f['best_phi'] = best_phi

    print(f'Written to {output_file}')
    return scores, best_phi, output_file


# ---------------------------------------------------------------------------
# Step 3 – build N×N consistency matrix from saved C
# ---------------------------------------------------------------------------

def build_consistency_matrix(common_lines_file):
    """
    Read a common_lines.h5 file (produced by compare_lines) and return the
    N_classes × N_classes score matrix (max Pearson per pair).

    The diagonal is 1 by definition.
    """
    with h5py.File(common_lines_file, 'r') as f:
        scores   = f['scores'][()]
        class_ids = f['class_ids'][()]
    return scores, class_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import pickle

    if len(sys.argv) < 2:
        print('Usage: python -m emc3.common_lines <config.pickle> '
              '[--Nrot 360] [--Nline 128] [--pow 1] [--device 0] [--out common_lines.h5]')
        sys.exit(1)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('config_file')
    ap.add_argument('--Nrot',   type=int,  default=360)
    ap.add_argument('--Nline',  type=int,  default=None)
    ap.add_argument('--pow',    type=float,default=1.)
    ap.add_argument('--device', type=int,  default=0)
    ap.add_argument('--out',    type=str,  default=None)
    args = ap.parse_args()

    config = pickle.load(open(args.config_file, 'rb'))

    print('Step 1: extracting sinograms …')
    calculate_lines(config, Nrot=args.Nrot, Nline=args.Nline,
                    device=args.device)

    print('Step 2: comparing lines …')
    scores, best_phi, out = compare_lines(config, power=args.pow, output_file=args.out)

    print('\nConsistency score matrix:')
    print(np.array2string(scores, precision=3, suppress_small=True))
