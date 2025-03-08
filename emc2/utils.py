import numpy as np
import math
import h5py
import os
from tqdm import tqdm
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED
import logging

logger = logging.getLogger(__name__)


class MyFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter
):
    pass


def clip_scalar(val, vmin, vmax):
    """ convenience function to avoid using np.clip for scalar values
    cannot take None as an argument"""
    return vmin if val < vmin else vmax if val > vmax else val


class ThreadQueue():
    """
    useful for memory intensive opencl code utilisizing SVM
    as it keeps a reference to input arrays (preventing seg fault)
    and limits queue depth (preventing memory saturation)
    """

    def __init__(self, max_queue_depth=None):
        self.executor = ThreadPoolExecutor()
        self.futures = []

        if not max_queue_depth:
            self.max_queue_depth = len(os.sched_getaffinity(0))
        else:
            self.max_queue_depth = max_queue_depth

    def _run(self, fn, *args, **kwargs):
        event = fn(*args, **kwargs)

        if hasattr(event, 'result'):
            event.result()

        elif hasattr(event, 'wait'):
            event.wait()

    def submit(self, fn, *args, **kwargs):
        self.futures.append(
            self.executor.submit(self._run, fn, *args, **kwargs)
        )

        # block if the queue is full to prevent large memory usage
        if len(self.futures) >= self.max_queue_depth:
            fs_done, fs_not_done = wait(
                self.futures,
                return_when=FIRST_COMPLETED
            )
            for future in fs_done:
                e = future.exception()
                if e is not None:
                    logger.critical(e)
                self.futures.remove(future)

    def shutdown(self):
        self.executor.shutdown()


def solve_axbc(a, b, c, fill_value=0.,
               ftol=1e-2, xtol=1e-3, maxiters=1000,
               algorithm='Halley', debug=False):
    """
    find the root of sum_i a_i / (x + b_i) - c = 0

    for a_i, b_i, c_i, x >= 0 (this is not checked)

    return fill_value when x is undetermined

    maximum value of x given by all b_i = 0:
        sum_i a_i / x = c
        xmax          = sum_i / c

    minimum value of x :
        if any b_i == 0 where a_i >= 0 :
            xmin = a_i / c
        else :
            xmin = 0

    solution for I = 1
        x = a_0 / c - b_0

    solution for I = 2
        f  = a0 / (x + b0) + a1 / (x + b1) - c = 0
        x  = [-b' - (b'^2 - 4 a' c')^1/2] / 2 a'
        a' = -c
        b' = a[0] + a[1] - c * (b[0] + b[1])
        c' = a[0] * b[1] + a[1] * b[0] - c * b[0] * b[1]

    if all b > 0 where a > 0 then:
        x = 0 if f(0) <= 0
        c > sum_i a_i / b_i
    """
    # could be expensive but probably good to check
    # ---------------------------------------------
    assert (np.all(b >= 0.))
    assert (np.all(a >= 0.))

    m = a > 0
    a = np.ascontiguousarray(a[m])
    b = np.ascontiguousarray(b[m])
    # ---------------------------------------------

    I = len(a)

    if I == 0 or c <= 0.:
        if debug:
            if I == 0:
                logger.debug(f'zero length input returning {fill_value}')
            if c <= 0:
                logger.debug(f'bad c-value {c} returning {fill_value}')
        return fill_value

    # confirmed all a >  0
    # confirmed all b >= 0
    # confirmed c > 0

    if I == 1:
        x = a[0] / c - b[0]
        x = clip_scalar(x, 0, np.inf)
        return x

    elif I == 2:
        ap = -c
        bp = a[0] + a[1] - c * (b[0] + b[1])
        cp = a[0] * b[1] + a[1] * b[0] - c * b[0] * b[1]
        x = (-bp - (bp**2 - 4 * ap * cp)**0.5) / (2 * ap)
        x = clip_scalar(x, 0, np.inf)
        return x

    xmin = 0.
    xmax = np.sum(a)/c
    fmax = None
    fmin = np.sum(a / (xmax + b)) - c

    # if all b == 0 then return xmax
    i = np.where(b == 0.)[0]
    if len(i) == I:
        return xmax

    # if some b == 0 where a > 0 then xmin >= amax / c
    elif len(i) > 0:
        amax = np.max(a[i])
        if amax > 0.:
            xmin = amax / c

    # if all b > 0 where a > 0 then
    # there may be a negative solution
    elif len(i) == 0:
        fmax = np.sum(a / b) - c
        if fmax <= 0:
            if debug:
                logger.debug(f'fmax {fmax} is less than 0 returning 0')
            return 0.

    fmax = fmax or (np.sum(a / (xmin + b)) - c)

    # now that the edge cases are dealt with
    # we begin line search
    x = xmax/2

    def converged(x, step, fn):
        if abs(step) < (xtol * x) and abs(fn) < (ftol * (fmax-fmin)):
            return True
        else:
            return False

    if algorithm == 'Newton':
        def f(xn):
            xb = xn + b
            fn = np.sum(a / xb) - c
            fpn = -np.sum(a / xb**2)
            return fn, fpn

        for i in range(maxiters):
            # Newton's method
            # x_n+1 = x_n - f(x_n) / f'(x_n)
            fn, fpn = f(x)
            step = - fn / fpn
            x = clip_scalar(x + step, xmin, xmax)

            if converged(x, step, fn):
                return x

    elif algorithm == 'Halley':
        def f(xn):
            xb = xn + b
            fn = np.sum(a / xb) - c
            fpn = -np.sum(a / xb**2)
            fppn = np.sum(a / xb**3)
            return fn, fpn, fppn

        for i in range(maxiters):
            # Halley's method
            # better by about 10% than Newton's
            # x_n+1 = x_n - 2 f fp / (2 fp^2 - f fpp)
            fn, fpn, fppn = f(x)
            step = - 2 * fn * fpn / (2*fpn**2 - fn * fppn)
            x = clip_scalar(x + step, xmin, xmax)

            if converged(x, step, fn):
                return x
    else:
        raise ValueError(f'could not parse algorithm {algorithm=}')

    logger.warning(
        f'Warning maximum iterations exceeded! \
        {x=} {xmin=} {xmax=} {fmax=} {fn=} {fpn=}'
    )
    return x


def get_transpose(K_di, B_di, working_directory=None):
    from emc2 import data_getter
    # check if transpose is already done
    fnam_dataT = os.path.join(working_directory, 'cachdir/dataT.h5')

    write = False
    if not os.path.exists(fnam_dataT):
        write = True
    else:
        with h5py.File(fnam_dataT) as f:
            if f['entry_1/data_1/data'].shape != K_di.shape[::-1]:
                logger.debug('transposed data found but shape is wrong, rewriting')
                write = True

    if write:
        data_getter.transpose_data(
            K_di,
            dataset='/entry_1/data_1/data',
            fnam='dataT.h5',
            working_directory=working_directory
        )

    # load
    K_id = data_getter.Data_getter(
        mask=np.ones(K_di.shape[0], dtype=bool),
        cxi_file=fnam_dataT,
        working_directory=working_directory
    )

    fnam_backgroundT = os.path.join(
        working_directory,
        'cachdir/backgroundT.h5'
    )

    write = False
    if not os.path.exists(fnam_backgroundT):
        write = True
    else:
        with h5py.File(fnam_backgroundT) as f:
            if f['entry_1/data_1/data'].shape != K_di.shape[::-1]:
                logger.debug('transposed background found but shape is wrong, '
                             'rewriting')
                write = True

    if write:
        data_getter.transpose_data(
            B_di,
            dataset='/entry_1/data_1/data',
            fnam='backgroundT.h5',
            dtype=np.float16,
            working_directory=working_directory,
        )

    # load
    logger.info(f'loading {fnam_backgroundT}')
    with h5py.File(fnam_backgroundT) as f:
        B_id = f['entry_1/data_1/data'][()]
    return K_id, B_id


class Geom_corr():
    def __init__(
        self,
        geom_fnam='/home/andyofmelbourne/Documents'
                  '/git_repos/xfel7927/geom/r0600.geom',
        return_centre=False
    ):
        # can we do this inside __init__?
        import extra_geom
        self.geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fnam)
        self.return_centre = return_centre

    def apply(self, ar):
        out, centre = self.geom.position_modules(ar)
        if self.return_centre:
            return out, centre
        else:
            return out


class Geom_corr_masked():
    def __init__(
        self,
        mask,
        geom_fnam='/home/andyofmelbourne/Documents'
                  '/git_repos/xfel7927/geom/r0600.geom'
    ):
        # can we do this inside __init__?
        import extra_geom
        self.geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fnam)

        self.mask = mask
        self.frame = np.zeros(self.geom.expected_data_shape, dtype=np.float32)
        self.frame[:] = np.nan

    def apply(self, ar):
        if ar.ndim == 2:
            frames = []
            for a in tqdm(ar):
                self.frame[self.mask] = a
                frames.append(self.frame.copy())
        else:
            self.frame[self.mask] = ar
            frames = self.frame
        out = self.geom.position_modules(np.array(frames))[0]
        return out


def get_beta(
    iteration=None,
    iterations=None,
    beta=None,
    beta_start=None,
    beta_stop=None,
    beta_strategy=None,
    **kwargs
):
    i = iteration
    I = iterations
    if beta_strategy == 'exponential':
        beta = (beta_stop / beta_start)**(min(i, I-1)/(I-1)) * beta_start

    elif beta_strategy == 'linear':
        beta = np.linspace(beta_start, beta_stop, I, endpoint=True)[min(i, I)]

    elif beta:
        pass

    else:
        raise ValueError('could not resolve beta strategy from config')

    return beta


def calc_q(wav, xyz):
    r = np.sum(xyz**2, axis=0)**0.5
    q = xyz / r
    q[2] -= 1
    q /= wav
    return q


def get_chunks(i0, i1, changes):
    """
    say we have an index 'i' where things change at location 'changes'

    and we have a range of values starting at i0
    and ending at i1-1 that we wish to iterate over
    in chunks such that nothing changes

    e.g.
    changes: [2, 7]
    0+diff : [0 0 1 0 0 0 0 1 0]
    thing  : [0 0 1 1 1 1 1 2 2]
                  -------------
                .............
                    +++++++
    -- i0 = 2, i1 = 9
    output = [(2, 7), (7, 9)]

    .. i0 = 1, i1 = 8
    output = [(1, 2), (2, 7), (7, 9)]

    ++ i0 = 3, i1 = 7
    output = [(3, 7)]
    """
    # j0s = [i0,] + [ i0 > changes < i1]
    # j1s = [i0 > changes < i1] + [i1,]

    m = (changes > i0) * (changes < i1)
    changes_in_range = tuple(changes[m])
    return list(zip((i0,) + changes_in_range, changes_in_range + (i1,)))


def int_to_list(N, v, name='parameter'):
    """
    if v is a list of length N, do nothing
    if v is an int then broadcast to list
    """
    if isinstance(v, int):
        out = N * [v]
    elif isinstance(v, float):
        out = N * [v]
    elif isinstance(v, str):
        out = N * [v]
    elif hasattr(v, '__len__') and len(v) == N:
        out = v
    else:
        raise ValueError(f'could not parse {name} in configuration file: {v}')

    return out


# thanks to Gaëtan de Menten
# https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def chunker(chunksize, size, offset=0):
    assert (size > 0)
    D = math.ceil(size / chunksize)
    dstart = np.arange(D) * chunksize + offset
    dstop = np.clip(dstart + chunksize, 0, size + offset)
    dd = dstop - dstart
    return list(zip(np.int32(dstart), np.int32(dstop), np.int32(dd)))


def chunker_mpi(size, N):
    ds = np.linspace(0, N, size+1, endpoint=True).astype(int)
    dstart = ds[:-1]
    dstop = ds[1:]
    dd = dstop - dstart
    return dstart, dstop, dd


def chunker_mpi_local(chunksize, size, N):
    """
    split a list of length N into 'size' chunks of roughly equal size
    then split the chunks into 'p' sub-chunks with all sub-chunks
    =< chunksize in size

    size = number of mpi processors
    p    = number of local chunks
    N    = total number of elememts to iterate over
    chunksize = the size of the chunks for all but the last
                chunk (which is <= chunksize)

    e.g. chunksize = 3, size = 2, N = 9
    l = 0 1 2 3 4 5 6 7 8
        ------- +++++++++ <- 2 mpi chunks
        +++++ - ##### ... <- 2 mpi chunks x 2 local chunks
    """
    # return an iterator for each rank
    out = []
    dstart_mpi, dstop_mpi, dd_mpi = chunker_mpi(size, N)
    for d0_mpi, d1_mpi, dd_mpi in zip(dstart_mpi, dstop_mpi, dd_mpi):
        out.append(chunker(chunksize, d1_mpi - d0_mpi, d0_mpi))
    return out


def get_model_slices(Is):
    N = Is[0].shape[0]
    classes = []

    slices = []
    for i, I in enumerate(Is):
        if I.ndim == 2:
            slices.append(I)
            classes.append(i)
        elif I.ndim == 3:
            slices.append(I[N//2])
            classes.append(i)

            slices.append(I[:, N//2])
            classes.append(i)

            slices.append(I[:, :, N//2])
            classes.append(i)
    return np.array(slices), np.array(classes)


def make_models_2D_image(Is, classes = None):
    # 3D --> 3 x 2D slices

    if classes is None:
        classes = np.arange(len(Is))

    # calculate grid
    n = math.ceil(len(Is)**0.5)

    N = Is[0].shape[0]

    slices_im = np.zeros((n * N, n * N), dtype=np.float32)

    # output centre positions
    positions = []
    classes_im = []

    for i in range(n):
        for j in range(n):
            c = n*i + j
            if c < len(Is):
                slices_im[i*N: (i+1)*N, j*N: (j+1)*N] = Is[c]
                x = N * (c % n) + N/2 + 0.5
                y = N * (c // n) + N/2 + 0.5
                positions.append((x, y))
                classes_im.append(classes[c])

    return slices_im, {'positions': positions, 'classes': classes_im, 'N': N}


def save_model_slices(
    models_I,
    model_dq,
    working_directory
):
    fnam = Path(working_directory).joinpath('iteration_info.h5')

    # get iteration number
    if fnam.is_file():
        with h5py.File(fnam, 'r') as f:
            N = f['iterations'][()]-1
    else:
        raise ValueError(f'could not find {fnam}! Model slices not saved')

    logger.info(f'saving model slices to {fnam} for iteration {N}')

    slices, classes = get_model_slices(models_I)

    with h5py.File(fnam, 'r+') as f:
        k = f'iteration_{N}'
        if k not in f:
            g = f.create_group(k)
        else:
            g = f[k]

        for k, v in zip(['model_slices', 'slice_classes'], [slices, classes]):
            if k in g and g[k].shape == v.shape:
                g[k][:] = v

            elif k in g and g[k].shape != v.shape:
                del g[k]

            if k not in g:
                g.create_dataset(k, data=v, chunks=v.shape, compression='gzip')

        k = 'model_dq'
        if k in g:
            del g[k]
        g[k] = model_dq


def write_h5(f, k, v, compression=True, chunks=None):
    if not hasattr(v, 'shape') or isinstance(v, str):
        f[k] = v
    else:
        if k in f:
            if f[k].shape == v.shape and f[k].dtype == v.dtype:
                f[k][:] = v
            else:
                del f[k]

        if k not in f:
            if not chunks:
                chunks = v.shape
            f.create_dataset(k, data=v, chunks=chunks, compression='gzip')


def save_iteration_info(
    P_max_d,
    Q_d,
    class_max_d,
    local_rmax_d,
    occupancy_dc,
    occupancy_r,
    working_directory,
    beta,
    sparse_file=''
):
    fnam = Path(working_directory).joinpath('iteration_info.h5')

    # get iteration number
    if fnam.is_file():
        with h5py.File(fnam, 'r') as f:
            N = f['iterations'][()]
    else:
        N = 0

    logger.info(f'saving iteration info to {fnam} '
                f'for iteration {N}')

    # initialise or resize datasets
    if N == 0:
        with h5py.File(fnam, 'w') as f:
            f['iterations'] = N+1

            f.create_dataset(
                'beta',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'Q',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'P_gini',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'orientation_changes',
                shape=(1,),
                maxshape=(None,),
                dtype=np.int32
            )

            f.create_dataset(
                'class_changes',
                shape=(1,),
                maxshape=(None,),
                dtype=np.int32
            )
    else:
        # resize
        keys = ['beta', 'Q', 'P_gini', 'orientation_changes', 'class_changes']
        with h5py.File(fnam, 'r+') as f:
            for key in keys:
                f[key].resize(N+1, axis=0)

    # write other results to iteration_{N}
    with h5py.File(fnam, 'r+') as f:
        k = f'iteration_{N}'
        if k in f:
            g = f[k]
        else:
            g = f.create_group(k)

        write_h5(g, 'occupancy_r', occupancy_r)
        write_h5(g, 'P_gini_d', P_max_d)
        write_h5(g, 'Q_d', Q_d)
        write_h5(g, 'most_likely_model_d', class_max_d)
        write_h5(g, 'occupancy_dc', occupancy_dc)
        write_h5(g, 'most_likely_orientation_d', local_rmax_d)
        write_h5(g, 'sparse_file', sparse_file)
        f['iterations'][...] = N+1
        f['beta'][N] = beta
        f['Q'][N] = np.mean(Q_d)
        f['P_gini'][N] = np.mean(P_max_d)

        # write differences
        if N > 0:
            k0 = f'iteration_{N-1}'
            g0 = f[k0]
            mlm_d0 = g0['most_likely_model_d'][()]
            mlm_d1 = class_max_d
            dc = np.sum(mlm_d0 != mlm_d1)
            f['class_changes'][N] = dc

            mlo_d0 = g0['most_likely_orientation_d'][()]
            mlo_d1 = local_rmax_d
            do = np.sum(mlo_d0 != mlo_d1)
            f['orientation_changes'][N] = do
        else:
            f['class_changes'][N] = Q_d.shape[0]
            f['orientation_changes'][N] = Q_d.shape[0]

    return True


if __name__ == "__main__":
    print('hello I am emc2.utils.py')
