from .h5_viewer import H5_viewer
from .getters import DataGetter_h5, GeomGetterCXI_h5, GeomGetterSparseCXI_h5

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
        getters = [GeomGetterSparseCXI_h5, GeomGetterCXI_h5, DataGetter_h5]

        super().__init__(directory, getters=getters, parent=parent)

