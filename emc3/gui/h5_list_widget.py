import sys
import h5py
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from .nested_list_widget import NestedListWidget


class H5_list_widget(NestedListWidget):

    def __init__(self, fnams, parent=None):
        self.fnams = fnams
        self.update_data(fnams)

        super().__init__(data=self.data, parent=parent)

    def refresh(self, fnams=None):
        if fnams is None:
            fnams = self.fnams

        self.update_data(fnams)
        self.setData(self.data)

    def update_data(self, fnams):
        # get contents of h5 file
        data = {}
        for fnam in fnams:
            try:
                with h5py.File(fnam) as f:
                    data[fnam] = hdf5_to_dict_flatter(f)
            except Exception as e:
                print(f'could not open {fnam=}: {e}')

        self.data = data

    def path_to_file_dataset(self, path_list):
        """
        convert
        ['data/r0087_radial_profiles.h5', 'ebeam', 'L3_energy: dataset float64 shape=(78168)']
        to:
        {
            fnam: 'data/r0087_radial_profiles.h5',
            dataset: 'ebeam/L3_energy',
            dtype: float64,
            shape: (78168,)'
        }
        """
        if 'dataset' not in path_list[-1]:
            return None

        file = path_list[0]
        dataset = '/'.join(path_list[1:-1])
        dataset += '/' + path_list[-1].split(':')[0]
        return {'fnam': file, 'dataset': dataset}


def hdf5_to_dict(h5obj):
    result = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            result[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            if item.shape == ():  # scalar dataset
                result[key] = {
                    "type": "scalar",
                    "dtype": str(item.dtype),
                    "value": item[()]  # actual scalar value
                }
            else:
                result[key] = {
                    "type": "dataset",
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                }
    return result


def hdf5_to_dict_flatter(h5obj):
    """Convert an HDF5 file/group into nested dicts with dataset descriptions as strings."""
    result = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            result[key] = hdf5_to_dict_flatter(item)
        elif isinstance(item, h5py.Dataset):
            dtype = str(item.dtype)
            if item.shape == ():  # scalar dataset
                value = item[()]
                result[key] = f"{key}: scalar {dtype} {value}"
            else:
                shape_str = "x".join(map(str, item.shape))
                result[key] = f"{key}: dataset {dtype} shape=({shape_str})"
    return result


if __name__ == "__main__":
    app = QApplication(sys.argv)

    fnam = '/home/andyofmelbourne/Documents/2025/LCLS-CXI-1008449/data/r0087_radial_profiles.h5'

    window = H5_list_widget(fnam)
    window.setWindowTitle("Hdf5 datasets with Collapsible Sublists")
    window.resize(400, 300)

    # Connect a signal (optional)
    # window.tree.itemClicked.connect(window.on_item_clicked)

    window.show()

    sys.exit(app.exec_())

