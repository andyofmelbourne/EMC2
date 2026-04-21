import h5py
from pathlib import Path
import numpy as np
import math
import argparse

class MyFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter
):
    pass

def chunker(chunksize, size, offset=0):
    assert (size > 0)
    chunksize = min(chunksize, size)
    D = math.ceil(size / chunksize)
    dstart = np.arange(D) * chunksize + offset
    dstop = np.clip(dstart + chunksize, 0, size + offset)
    dd = dstop - dstart
    return list(zip(np.int32(dstart), np.int32(dstop), np.int32(dd)))


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
        with h5py.File(fnam, 'w') as f:
            f['iterations'] = 1
            N = 0
        #raise ValueError(f'could not find {fnam}! Model slices not saved')

    # logger.info(f'saving model slices to {fnam} for iteration {N}')

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
                g.create_dataset(k, data=v, chunks=v.shape, compression='gzip',
                                 compression_opts=1)

        k = 'model_dq'
        if k in g:
            del g[k]
        g[k] = model_dq


def save_data_info(
    frames_d,
    frame_labels_j,
    DOS,
    D,
    working_directory
):
    fnam = Path(working_directory).joinpath('iteration_info.h5')

    # get iteration number
    if fnam.is_file():
        with h5py.File(fnam, 'r') as f:
            N = f['iterations'][()]-1
    else:
        raise ValueError(f'could not find {fnam}! Data info not saved')

    with h5py.File(fnam, 'r+') as f:
        k = f'iteration_{N}'
        if k not in f:
            g = f.create_group(k)
        else:
            g = f[k]

        out = {'frames': frames_d}

        if DOS is not None:
            out['DOS'] = DOS

        if frame_labels_j is not None:
            for k, v in frame_labels_j.items():
                out[f'frame_labels/{k}'] = v[frames_d]

        # save class labels if possible
        k = f'most_likely_model_d'
        if k in g:
            # d: subset of frames index
            m_d = g[k][()]
            # j: global frame index
            labels_j = -np.ones(D, dtype=int)
            labels_j[frames_d] = m_d
            out['class_labels'] = labels_j

        for k, v in out.items():
            if k in g and g[k].shape == v.shape:
                g[k][:] = v

            elif k in g and g[k].shape != v.shape:
                del g[k]

            if k not in g:
                g.create_dataset(k, data=v)


class Geom_corr_xyz():
    """
    xy_map = (2+,) + frame_shape shaped array of floating point pixel locations
    x_coords = xyz_map[0]
    y_coords = xyz_map[1]

    display array of shape (N, M):
        n[i] = round((x[i] - x_min) / pixel_size)
        m[i] = round((y[i] - y_min) / pixel_size)
        l[i] = M * n[i] + m[i]

        where i is a flattened (frame) pixel index
        where j is a flattened (image) pixel index

        image[l[i]] = frame[i]

    N = round((x_max - x_min) / pixel_size)+1
    M = round((y_max - y_min) / pixel_size)+1

    centre coordinates in image frame:
        n_0 = round(-x_min / pixel_size)
        m_0 = round(-y_min / pixel_size)
    """

    def __init__(
        self,
        xyz_map,
        pixel_size,
        return_centre=False
    ):
        x, y = xyz_map[:2]
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()

        self.centre = [round(-x_min / pixel_size), round(-y_min / pixel_size)]

        n = np.round((x - x_min) / pixel_size).astype(np.uint32)
        m = np.round((y - y_min) / pixel_size).astype(np.uint32)
        N = round((x_max - x_min) / pixel_size)+1
        M = round((y_max - y_min) / pixel_size)+1

        self.l = M * n.ravel() + m.ravel()
        self.shape = (N, M)
        self.size = N * M
        self.image = np.zeros((M*N,), dtype=np.float32)
        self.image.fill(np.nan) # shows as background
        self.dtype = self.image.dtype
        self.frame_shape = x.shape
        self.return_centre = return_centre

        # make inverse of l
        t = np.arange(x.size).reshape(x.shape)
        im = self.apply(t).ravel().copy()
        im[np.isnan(im)] = -1
        self.linv = im.astype(int)

        """
        maps raveled data coordinate i to
        raveled image coordinate n
        """
        self.data_to_display_coords_ravel = self.l

        """
        maps raveled image coordinate n to
        raveled data coordinate i

        return -1 for n values that fall in panel gaps
        """
        self.display_to_data_coords_ravel = self.linv

    def apply(self, ar):
        if len(ar.shape) == (len(self.frame_shape) + 1):
            N = ar.shape[0]

            out = np.empty(
                    (N,) + self.shape,
                    dtype=self.image.dtype)

            for n in range(N):
                self.image[self.l] = ar[n].ravel()
                out[n][:] = self.image.reshape(self.shape)
        else:
            self.image[self.l] = ar.ravel()
            out = self.image.reshape(self.shape)

        if self.return_centre:
            out = (out, self.centre)

        return out

    # for compatibility with extra_geom
    def position_modules(self, ar, out=None):
        self.return_centre = False
        t = self.apply(ar)
        if out is not None:
            out[:] = t
        else:
            out = t.copy()
        return out, self.centre

class Geom_corr_xyz_masked():

    def __init__(
        self,
        xyz_map,
        mask,
        pixel_size,
        return_centre=False
    ):
        self.mask = mask
        self.geom = Geom_corr_xyz(xyz_map, pixel_size, return_centre=return_centre)

    def apply(self, ar):
        if len(ar.shape) == 1:
            ar2 = np.zeros(self.geom.frame_shape, dtype=self.geom.dtype)
            ar2[self.mask] = ar
        elif len(ar.shape) == 2:
            ar2 = np.zeros((ar.shape[0],) + self.geom.frame_shape, dtype=self.geom.dtype)
            ar2[:, self.mask] = ar
        else:
            raise ValueError(f'could not parse array shape {ar.shape}')
        return self.geom.apply(ar2)


def clip_scalar(val, vmin, vmax):
    """ convenience function to avoid using np.clip for scalar values
    cannot take None as an argument"""
    return vmin if val < vmin else vmax if val > vmax else val


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
    if not np.all(b >= 0.):
        print(f'{b[b<0.]=}')
        print(f'{b=}')
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
                print(f'zero length input returning {fill_value}')
            if c <= 0:
                print(f'bad c-value {c} returning {fill_value}')
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
                print(f'fmax {fmax} is less than 0 returning 0')
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

    print(
        f'Warning maximum iterations exceeded! \
        {x=} {xmin=} {xmax=} {fmax=} {fn=} {fpn=}'
    )
    return x
