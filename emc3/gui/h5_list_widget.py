import h5py
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from .nested_list_widget import NestedListWidget

class DataGetter_h5:
    """
    A basic getter that serves h5 datasets without leaving the
    file open, in case it is being written to by another process

    can treat me like an open h5py dataset
    """
    def __init__(self, fnam, dataset, size_limit_gb=.1):
        self.fnam = fnam
        self.dataset = dataset
        self.name = dataset
        self.size_limit_gb = size_limit_gb
        self.numpy = False
        self.value = '' # for storing scalar values
        self.dataset_name = dataset.split('/')[-1]
        self.is_loaded = False
        self.data = None
        self.callback = None

        self.update()

    def update(self):
        fnam, dataset = self.fnam, self.dataset

        with h5py.File(fnam) as f:
            self.ndim = f[dataset].ndim
            self.shape = f[dataset].shape
            self.size = np.prod(self.shape)
            self.dtype = f[dataset].dtype
            self.nbytes = f[dataset].nbytes

        # do not delay loading if scalar type
        if self.shape == ():
            self.value = self[()]
            self.data = self.value
            self.is_loaded = True

        if (self.nbytes / 1024**3) < self.size_limit_gb:
            self.numpy = True

        if self.callback is not None:
            self.callback()

    def __len__(self):
        return self.shape[0]

    def load_from_file(self, key):
        with h5py.File(self.fnam) as f:
            return f[self.dataset][key]

    def __getitem__(self, key):
        if self.numpy:
            if not self.is_loaded:
                self.data = self.load_from_file(())

            # special case for consistency b/w h5py and numpy
            # h5py  dataset[()] -> b'asdfasdf' fine
            # numpy dataset[()] -> error
            if type(self.data) is bytes and key == ():
                return self.data[:]

            return self.data[key]
        else:
            return self.load_from_file(key)

    def __repr__(self):
        return f'{self.dataset_name}: {self.dtype} {self.shape} {self.value}'


class H5_info():
    """
    store information about h5 datasets in a directory or file

    this prevents having to repeatedly read files and therefore block writing

    Examples:
        $ h5ls -r test.cxi
        /                        Group
        /entry_1                 Group
        /entry_1/data_1          Group
        /entry_1/data_1/data     Soft Link {/entry_1/instrument_1/detector_1/data}
        /entry_1/instrument_1    Group
        /entry_1/instrument_1/detector_1 Group
        /entry_1/instrument_1/detector_1/background Dataset {1, 16, 512, 128}
        /entry_1/instrument_1/detector_1/data Dataset {16384, 16, 512, 128}
        /entry_1/instrument_1/detector_1/good_pixels Dataset {16, 512, 128}
        /entry_1/instrument_1/detector_1/lit_pixels Dataset {16384}
        /entry_1/instrument_1/detector_1/photon_counts Dataset {16384}
        /entry_1/instrument_1/detector_1/pixel_size Dataset {SCALAR}
        /entry_1/instrument_1/detector_1/saturated_pixels Dataset {16384}
        /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
        /entry_1/instrument_1/detector_1/xyz_map Dataset {3, 16, 512, 128}
        /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
        /entry_1/instrument_1/source_1 Group
        /entry_1/instrument_1/source_1/fluence Dataset {16384}
        /entry_1/instrument_1/source_1/photon_energy Dataset {16384}
        /entry_1/instrument_1/source_1/photon_wavelength Dataset {16384}
        /entry_1/instrument_1/source_1/pulse_energy Dataset {16384}


    self = H5_info('test.cxi')

    self[key] gives you a nested dictionary if key references a file or group
    self['entry_1'] = {
        'data_1' = {...}
        'instrument_1' = {...}
    }

    self[key] gives you a getter object for a dataset if key references a h5py dataset
    self['entry_1/data_1/data'][0] = ndarray (16, 512, 128)
    print(self['entry_1/data_1/data'][0]) = 'data: float32 (12, 13, 14)'

    self.update(): updates information after change

    self.data = {
        'entry_1': {
            'type': 'group',
            'data': {
                'type': dataset,
                'dtype': dtype('<f8'),
                'shape': (12, 13, 14),
                'print': 'data: float32 (12, 13, 14)'
            }
        }
    }

    self.data_flat = {
        'file.h5/entry_1/data': {
                    'type': dataset,
                    'dtype': dtype('<f8'),
                    'shape': (12, 13, 14),
                    'print': 'data: float32 (12, 13, 14)'
        }
    }
    """
    def __init__(
        self,
        fnam,
        extensions=['*.cxi', '*.h5']
    ):
        self.fnam = fnam
        self.fnams = None
        self.tree = {}
        self.flat_tree = {}

        self.extensions = extensions
        self.update(fnam)

    def update(self, fnam=None):
        if fnam is None:
            fnam = self.fnam

        # determine if 'obj' is a file or directory
        # or a list of files
        self.is_file = False
        self.is_dir = False
        self.is_files = False

        if np.all([Path(f).is_file() for f in str(fnam)]):
            self.is_files = True
        elif Path(fnam).is_file():
            self.is_file = True
        elif Path(fnam).is_dir():
            self.is_dir = True
        else:
            raise ValueError(f'fnam of type {type(fnam)} is not a file or directory')

        # get file names
        if self.is_files:
            self.fnams = fnam
        elif self.is_dir:
            self.fnams = []
            for extension in self.extensions:
                self.fnams += [str(f) for f in Path(fnam).glob(extension)]
        else:
            self.fnams = [fnam]

        self.fnams = np.sort(self.fnams)

        # ------ populate dataset tree

        # so we can remove deleted datasets later
        old_keys = list(self.flat_tree.keys())
        self.added_keys = []

        tree = {}
        for fnam in self.fnams:
            with h5py.File(fnam) as f:
                tree[fnam] = self.hdf5_to_dict(f)

        for k in old_keys:
            if k not in self.added_keys:
                del self.flat_tree[k]

        self.tree = tree

    def hdf5_to_dict(self, h5obj):
        result = {}
        for key, item in h5obj.items():
            if isinstance(item, h5py.Group):
                result[key] = self.hdf5_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                # would like to update this rather than
                # overwrite if possible
                flat_key = f'{item.file.filename}{item.name}'
                if flat_key in self.flat_tree:
                    self.flat_tree[flat_key].update()
                else:
                    self.flat_tree[flat_key] = DataGetter_h5(item.file.filename, item.name)
                    self.added_keys.append(flat_key)

                result[key] = self.flat_tree[flat_key]
        return result

    def __getitem__(self, key):
        if key in self.tree:
            return self.tree[key]

        if key in self.flat_tree:
            return self.flat_tree[key]

        raise KeyError(key)


class H5_list_widget(NestedListWidget):
    """
    this joins NestedListWidget with H5_info
    """

    def __init__(self, fnams, parent=None):
        self.fnams = fnams
        self.h5_info = H5_info(fnams)

        super().__init__(data=self.h5_info.tree, parent=parent)

    def update(self):
        self.h5_info.update()
        self.setData(self.h5_info.tree)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        fnam = sys.argv[1]
    else:
        fnam = '/home/andyofmelbourne/Documents/2025/LCLS-CXI-1008449/data/r0087_radial_profiles.h5'

    window = H5_list_widget(fnam)
    window.setWindowTitle("Hdf5 datasets with Collapsible Sublists")
    window.resize(400, 300)

    # Connect a signal (optional)
    # window.tree.itemClicked.connect(window.on_item_clicked)

    window.show()

    sys.exit(app.exec_())
    #fnam = '/home/andyofmelbourne/Documents/git_repos/EMC2/emc3/fsim/temp'
    #h5_info = H5_info(fnam)

