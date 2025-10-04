import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
import h5py
import numpy as np
import signal
from pathlib import Path

from .h5_viewer import H5_viewer
from ..utils import Geom_corr_xyz
from ..data import DataSparseCXI_full_frame

# add automatic geometry correction for multipanel datasets
class Geom_filter():
    def __init__(self, geom, ar=None):
        self.geom = geom
        self.l = len(geom.frame_shape)
        self.ndim = 2
        self.shape = geom.shape
        self.dtype = np.float32

        if ar is not None:
            self.check(ar)

        self.data_to_display_coords_ravel = \
                self.geom.data_to_display_coords_ravel

        self.display_to_data_coords_ravel = \
                self.geom.display_to_data_coords_ravel

        self.image_shape = self.geom.shape
        self.frame_shape = self.geom.frame_shape
        self.frame_data = None

    def check(self, ar):
        self.ar = ar
        if ar.shape[-self.l:] == self.geom.frame_shape:

            if len(ar.shape) == self.l + 1:
                self.ndim = 3
                self.shape = (ar.shape[0],)+self.geom.shape
            elif len(ar.shape) == self.l:
                self.ndim = 2
                self.shape = self.geom.shape
            else:
                raise ValueError(f'cannot handle panel geometry {ar.shape}')

            return True
        else:
            return False

    def apply(self, ar):
        self.frame_data = ar
        out = self.geom.apply(ar)
        return out

    def __len__(self):
        return self.shape[0]

    def __call__(self, ar):
        if self.check(ar):
            return Geom_filter(self.geom, ar)
        else:
            return ar

    def __getitem__(self, key):
        return self.apply(self.ar[key])


def get_geometry_filter(fnam):
    with h5py.File(fnam) as f:
        key = 'entry_1/instrument_1/detector_1/xyz_map'
        if key in f:
            xyz = f[key][()]

            key = 'entry_1/instrument_1/detector_1/x_pixel_size'
            if key in f:
                pixel_size = f[key][()]

                geom = Geom_corr_xyz(xyz, pixel_size)
                geom_filter = Geom_filter(geom)
        else:
            geom_filter = None
            print('could not find geometry information in {fnam}')
    return geom_filter


class CXI_viewer(H5_viewer):
    """
    Right now this just adds a function for applying geometry to multipanel
    datasets when:

        entry_1/instrument_1/detector_1/xyz_map
        and
        entry_1/instrument_1/detector_1/x_pixel_size

    can be found in a cxi file.

    Perhaps I should add display widget that does this at a lower level later.
    """
    def __init__(self, directory, parent=None):
        fnams = [str(f) for f in Path(directory).glob('*.cxi')]

        geom_filter = {}
        for f in fnams:
            geom_filter[f] = get_geometry_filter(f)

        super().__init__(directory, filters=geom_filter, parent=parent)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C

    pg.setConfigOption('background', pg.mkColor(0.1))

    app = QApplication(sys.argv)

    # fnam = '/home/andyofmelbourne/Documents/2025/LCLS-CXI-1008449/data/r0087_radial_profiles.h5'
    if len(sys.argv) > 1:
        fnam = sys.argv[1]
    else:
        # fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits_no_mask.cxi'
        # fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'
        fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/'

    fnams = [str(f) for f in Path(fnam).glob('*.cxi')]

    geom_filter = {}
    for f in fnams:
        geom_filter[f] = get_geometry_filter(f)

    window = H5_viewer(fnam, filters=geom_filter)
    window.setWindowTitle("CXI viewer")
    window.resize(800, 400)
    window.show()

    sys.exit(app.exec_())
