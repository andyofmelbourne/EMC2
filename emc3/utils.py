import h5py
from pathlib import Path
import numpy as np
import math


def chunker(chunksize, size, offset=0):
    assert (size > 0)
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
        raise ValueError(f'could not find {fnam}! Model slices not saved')

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
                g.create_dataset(k, data=v, chunks=v.shape, compression='gzip')

        k = 'model_dq'
        if k in g:
            del g[k]
        g[k] = model_dq


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
