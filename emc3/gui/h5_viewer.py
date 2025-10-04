import sys
import h5py
import numpy as np
from pathlib import Path

from .h5_list_widget import H5_list_widget
from .plot_widget import PlotWidget

from PyQt5.QtWidgets import (
        QApplication, QWidget, QMainWindow, QHBoxLayout, QSplitter,
        QVBoxLayout, QPushButton
        )
from PyQt5.QtCore import Qt


class MyMainWindow(QMainWindow):
    def closeEvent(self, event):
        if self.centralWidget():
            self.centralWidget().on_parent_close(event)  # custom method
        super().closeEvent(event)


def open_in_window(widget, title="New Window", size=(400, 300)):
    """
    Wraps any QWidget in a QMainWindow and shows it.
    Keeps a reference to avoid garbage collection.
    """
    window = MyMainWindow()
    window.setWindowTitle(title)
    window.setCentralWidget(widget)
    widget.setParent(window)
    window.resize(*size)
    window.show()
    return window


class DataGetter_h5:
    def __init__(self, fnam, dataset, size_limit_gb=1):
        self.fnam = fnam
        self.dataset = dataset

        self.refresh()
        """
        if (self.nbytes / 1024**3) < size_limit_gb:
            self.data = f[dataset][()]
        else:
            self.data = None
        """

    def refresh(self):
        fnam, dataset = self.fnam, self.dataset

        with h5py.File(fnam) as f:
            self.ndim = f[dataset].ndim
            self.shape = f[dataset].shape
            self.size = np.prod(self.shape)
            self.dtype = f[dataset].dtype
            self.nbytes = f[dataset].nbytes

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        with h5py.File(self.fnam) as f:
            return f[self.dataset][key]


class H5_viewer(QWidget):
    def __init__(self, fnam, filters=[], parent=None):
        super().__init__(parent)
        self.filters = filters
        self.fnam = fnam
        self.fnams = None

        # update file list
        self.update_file_list()

        # Create widgets
        self.h5_list_widget = H5_list_widget(self.fnams)

        self.open_plots = []
        self.wins = []

        # apply a data filter and tell the plot widget
        # about the other open windows
        self.plot_widget = PlotWidget(
                parent=self,
                filter=None,
                open_plots=self.open_plots)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        # splitter that separates plots stacked vertically
        self.vsplitter = QSplitter(Qt.Vertical)
        self.vsplitter.addWidget(self.plot_widget)

        # make refresh button
        leftPanel = QWidget()
        l = QVBoxLayout(leftPanel)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(0)
        l.addWidget(self.h5_list_widget)
        refresh_button = QPushButton('refresh')
        refresh_button.clicked.connect(self.refresh)
        l.addWidget(refresh_button)

        splitter = QSplitter()
        splitter.addWidget(leftPanel)
        splitter.addWidget(self.vsplitter)
        splitter.setSizes([200, 600])
        splitter.setStretchFactor(0, 0)       # tree widget does NOT stretch
        splitter.setStretchFactor(1, 1)       # plot widget expands/contracts
        self.splitter = splitter
        layout.addWidget(splitter)

        # Connect h5_list_widget signal directly to a handler
        self.h5_list_widget.itemClickedPath.connect(self.on_item_path_clicked)

    def update_file_list(self):
        fnam = self.fnam

        if Path(fnam).is_dir():
            self.fnams = [str(f) for f in Path(fnam).glob('*.cxi')]
            self.fnams += [str(f) for f in Path(fnam).glob('*.h5')]

        elif Path(fnam).is_file():
            self.fnams = [fnam]

        else:
            err = f'could not find file or files with \
                    *.h5 / *.cxi extension: {fnam}'
            raise ValueError(err)

    def on_item_path_clicked(self, path_list):
        # path = "/" + "/".join(path_list)
        data = self.h5_list_widget.path_to_file_dataset(path_list)

        if data is None:
            return

        fnam = data['fnam']
        name = data['dataset']

        modifiers = QApplication.keyboardModifiers()

        if modifiers == Qt.ControlModifier:
            where = 'window'

        elif modifiers == Qt.ShiftModifier:
            where = 'splitter'

        else:
            where = 'main'

        self.open(fnam, name, where=where)

    def open(self, fnam, name, where='main'):
        if fnam in self.filters:
            filter = self.filters[fnam]
        else:
            filter = lambda x: x

        if where == 'window' or where == 'splitter':
            widget = PlotWidget(filter=filter, open_plots=self.open_plots)
        else:
            widget = self.plot_widget

        data = DataGetter_h5(fnam, name)

        if where == 'window':
            self.wins.append(
                    open_in_window(
                        widget,
                        title=name,
                        size=(400, 200)
                        )
                    )

        elif where == 'splitter':
            self.vsplitter.addWidget(widget)

        else:
            widget.filter = filter

        widget.plot(data, name=name)

    def refresh(self):
        for plot in self.open_plots:
            # only I know about this method
            if plot.data is not None:
                plot.data.refresh()
            plot.refresh()

        self.update_file_list()
        self.h5_list_widget.refresh(self.fnams)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # fnam = '/home/andyofmelbourne/Documents/2025/LCLS-CXI-1008449/data/r0087_radial_profiles.h5'
    fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits_no_mask.cxi'

    window = H5_viewer(fnam)
    window.setWindowTitle("HDF5 Tree + Plot (Signal Version)")
    window.resize(800, 400)
    window.show()

    sys.exit(app.exec_())
