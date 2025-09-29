from pathlib import Path

from .model_slices_widget import Model_slice_widget

from PyQt5.QtWidgets import (
        QApplication, QWidget, QMainWindow, QHBoxLayout, QSplitter
        )

from .h5_viewer import H5_viewer
from .emc_scat_widget import EMC_scatter_widget

class Iteration_gui(Model_slice_widget):
    def __init__(self, directory, parent=None):
        # launch other widgets in separate windows
        fnam = Path(directory) / 'iteration_info.h5'

        # line plots in separate window
        # -----------------------------
        self.plots_widget = H5_viewer(fnam)
        self.plots_window = QMainWindow()
        self.plots_window.setWindowTitle('plots')
        self.plots_window.setCentralWidget(self.plots_widget)
        self.plots_widget.setParent(self.plots_window)
        self.plots_window.resize(800, 800)
        self.plots_window.show()

        self.plots_widget.open(fnam, 'Q', where='splitter')
        self.plots_widget.open(fnam, 'beta', where='splitter')
        self.plots_widget.open(fnam, 'class_changes', where='splitter')
        self.plots_widget.open(fnam, 'orientation_changes', where='splitter')

        # scatter plot in separate window
        # -------------------------------
        self.scat_window = QMainWindow()
        self.scat_widget = EMC_scatter_widget(directory,
                                              parent=self.scat_window)
        self.scat_window.setWindowTitle('plots')
        self.scat_window.setCentralWidget(self.scat_widget)
        self.scat_window.resize(800, 800)
        self.scat_window.show()

        super().__init__(directory)

        # connect selected classes between model slices and scatter
        self.selectedLabelsSignal.connect(self.scat_widget.update_selection)
        self.selectedLabelsSignal.connect(self.scat_widget.update_selection)

        self.iterationChanged.connect(
                lambda x: self.scat_widget._increment(gkey=x)
                )

        self.scat_widget.iterationChanged.connect(
                lambda x: self.update_plots(key=x)
                )



