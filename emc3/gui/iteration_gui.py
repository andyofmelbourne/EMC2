from pathlib import Path

from .model_slices_widget import Model_slice_widget

from PyQt5.QtWidgets import (
        QApplication, QWidget, QMainWindow, QHBoxLayout, QSplitter
        )

from .h5_viewer import H5_viewer
from .emc_scat_widget import EMC_scatter_widget
from .getters import DataGetter_h5, GeomGetterCXI_h5, GeomGetterSparseCXI_h5

class Iteration_gui(Model_slice_widget):
    def __init__(self, directory, parent=None):
        super().__init__(directory)

        # launch other widgets in separate windows
        self.fnam = Path(directory) / 'iteration_info.h5'
        self.directory = directory

        # line plots in separate window
        # -----------------------------
        self.getters = [GeomGetterSparseCXI_h5, GeomGetterCXI_h5, DataGetter_h5]
        self.plots_widget = H5_viewer(self.fnam, getters=self.getters)
        self.plots_window = QMainWindow(parent=self)
        self.plots_window.setWindowTitle('plots')
        self.plots_window.setCentralWidget(self.plots_widget)
        self.plots_widget.setParent(self.plots_window)
        self.plots_window.resize(800, 800)
        self.plots_window.show()

        self.plots_widget.open(self.fnam, 'Q', where='main')
        self.plots_widget.open(self.fnam, 'beta', where='splitter')
        self.plots_widget.open(self.fnam, 'class_changes', where='splitter')
        self.plots_widget.open(self.fnam, 'orientation_changes', where='splitter')

        # scatter plot in separate window
        # -------------------------------
        self.scat_window = QMainWindow(parent=self)
        self.scat_widget = EMC_scatter_widget(directory,
                                              parent=self.scat_window)
        self.scat_window.setWindowTitle('plots')
        self.scat_window.setCentralWidget(self.scat_widget)
        self.scat_window.resize(800, 800)
        self.scat_window.show()

        # connect selected classes between model slices and scatter
        self.selectedLabelsSignal.connect(self.scat_widget.update_selection)

        self.iterationChanged.connect(
                lambda x: self.scat_widget._increment(gkey=x)
                )

        self.scat_widget.iterationChanged.connect(
                lambda x: self.update_plots(key=x)
                )

        # show selected frames on signal
        self.showFramesSignal.connect(self.show_frames)

    def show_frames(self, frames):
        # hack: should include reference to cxi file in iteration info
        cxi = list(Path(self.directory).glob('*.cxi'))[0]
        pw = self.plots_widget.open(cxi, 'entry_1/data_1/data', where='window')
        pw.current_plot.update_xmap(frames)
