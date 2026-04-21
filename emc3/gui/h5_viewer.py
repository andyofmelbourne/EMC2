import sys
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


class H5_viewer(QWidget):
    def __init__(self, fnam, getters=[], parent=None):
        super().__init__(parent)
        self.getters = getters
        self.fnam = fnam

        # Create widgets
        self.h5_list_widget = H5_list_widget(fnam)

        self.open_plots = []
        self.wins = []

        # tell the plot widget about the other open windows
        self.plot_widget = PlotWidget(
                parent=self,
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
        self.h5_list_widget.itemClicked.connect(self.on_item_path_clicked)

    def on_item_path_clicked(self, data):
        if data is None:
            return

        fnam = data.fnam
        name = data.dataset

        modifiers = QApplication.keyboardModifiers()

        if modifiers == Qt.ControlModifier:
            where = 'window'

        elif modifiers == Qt.ShiftModifier:
            where = 'splitter'

        else:
            where = 'main'

        self.open(fnam, name, data, where=where)

    def open(self, fnam, name, data=None, where='main', plugin=None):
        if where == 'window' or where == 'splitter':
            widget = PlotWidget(open_plots=self.open_plots)
        else:
            widget = self.plot_widget

        for g in self.getters:
            try:
                data2 = g(fnam, name)
                data = data2
                break
            except Exception as e:
                print(e)
                pass

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

        widget.plot(data, name=name, plugin=plugin)
        return widget

    def refresh(self):
        self.h5_list_widget.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        fnam = sys.argv[1]
    else:
        # fnam = '/home/andyofmelbourne/Documents/2025/LCLS-CXI-1008449/data/r0087_radial_profiles.h5'
        fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits_no_mask.cxi'

    window = H5_viewer(fnam)
    window.setWindowTitle("HDF5 Tree + Plot (Signal Version)")
    window.resize(800, 400)
    window.show()

    sys.exit(app.exec_())
