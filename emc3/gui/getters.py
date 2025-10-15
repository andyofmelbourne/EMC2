"""
Provide getters that take a h5file name and dataset and present
this data for more simple display widgets

data = Getter(fnam, dataset)
    data.dtype
    data.shape
    data.ndim
    data.size
    len(data)
    data[0]
    data[()]
"""
import numpy as np
import h5py

from ..utils import Geom_corr_xyz
from ..data import DataSparseCXI_full_frame

class DataGetter_h5:
    """
    A basic getter that serves h5 datasets without leaving the
    file open, in case it is being written to by another process
    """
    def __init__(self, fnam, dataset, size_limit_gb=1):
        self.fnam = fnam
        self.dataset = dataset
        self.size_limit_gb = size_limit_gb
        self.numpy = False

        self.refresh()

    def refresh(self):
        fnam, dataset = self.fnam, self.dataset

        with h5py.File(fnam) as f:
            self.ndim = f[dataset].ndim
            self.shape = f[dataset].shape
            self.size = np.prod(self.shape)
            self.dtype = f[dataset].dtype
            self.nbytes = f[dataset].nbytes

        if (self.nbytes / 1024**3) < self.size_limit_gb:
            self.data = self[()]
            self.numpy = True
        else:
            self.data = None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if self.numpy:
            return self.data[key]
        else:
            with h5py.File(self.fnam) as f:
                return f[self.dataset][key]


def find_in_h5(h5obj, find_key):
    result = None
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            result = find_in_h5(item, find_key)
        elif isinstance(item, h5py.Dataset):
            if key == find_key:
                result = item
        if result is not None:
            return result
    return result


class GeomGetterCXI:
    """
    A thin wrapper around Geom_corr_xyz() with extras for gui.

    file must contain:
        xyz_map
        x_pixel_size or pixel_size
    and the dataset shape must be compatible with the xyz_map

    This object will act like a numpy array with geometry corrected shape.
    Extra functionality is provided for mask making.
    """
    def __init__(self, data, xyz_map, pixel_size):
        """
        check if the dataset is suitable for geometry correction
        """
        # check if the shape matches
        s1 = data.shape
        s2 = xyz_map.shape[1:]

        if (len(s2) != len(s1) and len(s2) != (len(data.shape)-1)):
            raise ValueError(f'incompatible shapes {s1} {s2}')

        if s2 != s1[-len(s2):]:
            raise ValueError(f'incompatible shapes {s1} {s2}')

        self.xyz_map = xyz_map
        self.pixel_size = pixel_size
        self.data = data

        self.geom = Geom_corr_xyz(self.xyz_map, self.pixel_size)
        self.ndim = 2 + len(s1) - len(s2)
        self.shape = s1[:-len(s2)] + self.geom.shape
        self.dtype = self.geom.dtype

        # for mask making
        self.data_to_display_coords_ravel = \
                self.geom.data_to_display_coords_ravel

        self.display_to_data_coords_ravel = \
                self.geom.display_to_data_coords_ravel

        self.image_shape = self.geom.shape
        self.frame_shape = self.geom.frame_shape
        self.frame_data = None

    def update_data(self):
        pass

    def apply(self, ar):
        self.frame_data = ar
        out = self.geom.apply(ar)
        return out

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        return self.apply(self.data[key])


class GeomGetterCXI_h5(GeomGetterCXI):
    def __init__(self, fnam, dataset):
        data = DataGetter_h5(fnam, dataset)

        # find xyz_map and pixel_size
        xyz_map, pixel_size = self.get_from_h5(fnam)

        self.fnam = fnam
        self.dataset = dataset

        super().__init__(data, xyz_map, pixel_size)

    def get_from_h5(self, fnam):
        """
        check if the file and dataset are suitable for geometry correction
        """
        with h5py.File(fnam) as f:
            # see if 'xyz_map' is in file
            xyz_map = find_in_h5(f, 'xyz_map')
            if xyz_map is None:
                raise ValueError(f'could not find xyz_map in {fnam}')

            # see if 'pixel_size' is in file
            pixel_size = find_in_h5(f, 'pixel_size')
            if pixel_size is None:
                pixel_size = find_in_h5(f, 'x_pixel_size')

            if pixel_size is None:
                raise ValueError(f'could not find pixel_size in file \
                                 {pixel_size}')

            pixel_size = pixel_size[()]
            xyz_map = xyz_map[()]

        return xyz_map, pixel_size

    def refresh(self):
        """
        This can be called when the h5 file is updated
        """
        self.__init__(self.fnam, self.dataset)


class GeomGetterSparseCXI_h5(GeomGetterCXI):
    def __init__(self, fnam, dataset):
        if 'data' not in dataset:
            raise ValueError(f'{dataset=} must be called "/data"')

        data = DataSparseCXI_full_frame(fnam)

        self.fnam = fnam
        self.dataset = dataset

        super().__init__(data, data.xyz_map, data.pixel_size)

    def refresh(self):
        """
        This can be called when the h5 file is updated
        """
        self.__init__(self.fnam, self.dataset)
