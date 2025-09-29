"""
Plot the projection of the relative occupancy for each frame to each class in a
2D plane:
    occupancy: Q_dc = sum_r P^c_dr

    Qxy_d = sum_c Q_dc [x_c, y_c]

    theta_c = 2 pi c / C
    x_c     = cos(theta_c)
    y_c     = sin(theta_c)

where P^c_dr is the probability matrix for class_id = c
Q_dc is already written to iteration_info.h5

The display widget is a glorified scatter plot widget, it doesn't add much to
the pyqtgraph's basic functionality.

Display widget:
    plot x_c, y_c as white circles
    plot Qxy_d as small green dots
    if there are labels, then show them as different colours
    right arrow: next frame
    left arrow: previous frame

    Input:
        title
        Sequence of length 3 tuples:
            label, pen, list of vectors
"""
import numpy as np
import pyqtgraph as pg
import argparse
from pathlib import Path
import h5py
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from collections import defaultdict
import signal
import math

from .model_slices_widget import BaseIterationInfo_getter


# Get key mappings from Qt namespace
qt_keys = (
    (getattr(Qt, attr), attr[4:])
    for attr in dir(Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)


import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout


class ScatterPlotWidget(QWidget):
    leftarrow = pyqtSignal()
    rightarrow = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        # Plot area
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Store items
        self.scatter_items = []
        self.label_items = []

    def set_spots(self, groups):
        """Clear plot and add new spots.

        groups: list of dicts with keys:
            - "style": dict with brush, pen, size, symbol
            - "points": list of (x, y)
        """
        self.clear()

        for group in groups:
            style = group.get("style", {})
            points = group.get("points", None)

            if points is None:
                continue

            xs, ys = zip(*points)

            sp = pg.ScatterPlotItem(x=xs, y=ys, **style)
            self.plot_widget.addItem(sp)
            self.scatter_items.append(sp)

    def clear(self):
        """Remove all scatter and label items from the plot."""
        for sp in self.scatter_items:
            self.plot_widget.removeItem(sp)

        for lbl in self.label_items:
            self.plot_widget.removeItem(lbl)

        self.scatter_items.clear()
        self.label_items.clear()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        key = keys_mapping[event.key()]


        if key == 'Right' :
            self.rightarrow.emit()

        elif key == 'Left' :
            self.leftarrow.emit()


class EMC_scatter_widget(QWidget):
    """
    Calculate unit vectors for each class uxy_c
        theta_c = 2 pi c / C
        ux_c     = cos(theta_c)
        uy_c     = sin(theta_c)

    Get Q_dc (BaseIterationInfo_getter)
        occupancy: Q_dc = sum_r P^c_dr

    Calculate scatter plot:
        Qxy_d = sum_c Q_dc [x_c, y_c]

    Update plot (ScatterPlotWidget)
    """

    def __init__(self, directory, parent=None):
        super().__init__(parent=parent)

        self.fnam = Path(directory) / 'iteration_info.h5'

        self.title = 'iteration {}'

        self.updating = False

        # add occupancy getter
        self.keys = ['occupancy_dc']
        self.itget = BaseIterationInfo_getter(self.fnam, self.keys)
        self.iterationChanged = self.itget.iterationChanged

        # add scatter plot
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self.scatter_widget = ScatterPlotWidget()
        layout.addWidget(self.scatter_widget)

        self.scatter_widget.leftarrow.connect(self.previous)
        self.scatter_widget.rightarrow.connect(self.next)

        self.selected_classes = []

        # set spot styles
        self.unit_style = {
                    'pen': 'w',
                    'size': 10,
                    'brush': None,
                    'hoverable': True,
                    'hoverBrush': pg.mkBrush(255, 255, 255, 100),
                    'data': None
                    }

        self.frame_style = {
                "brush": (100, 100, 255, 50),
                "size": 5,
                "pen": None
                }

        self.uxy_c = None
        self.Qxy_d = None

        self.next()

    def update_selection(self, selection):
        self.selected_classes = selection

        if self.uxy_c is not None:
            C = self.uxy_c.shape[0]
            pen = []
            for c in range(C):
                pen.append('g' if c in self.selected_classes else 'w')
            self.unit_style['pen'] = pen

        self._increment(0)

    def unit_vectors(self, C):
        if self.uxy_c is None or self.uxy_c.shape[0] != C:
            theta = 2*np.pi*np.arange(C) / C
            unit_vectors = np.zeros((C, 2))
            unit_vectors[:, 0] = np.cos(theta)
            unit_vectors[:, 1] = np.sin(theta)
            self.uxy_c = unit_vectors

            # set points
            self.unit_style['data'] = np.arange(C)
            self.u_data = {
                    'style': self.unit_style,
                    'points': unit_vectors
                    }

    def next(self):
        self._increment(+1)

    def previous(self):
        self._increment(-1)

    def _increment(self, inc=1, gkey=None):
        if self.updating:
            return

        self.updating = True

        # get occupancy
        title, Q_dc = self.itget._increment(inc=inc, gkey=gkey)

        self.title = title
        self.parent().setWindowTitle(self.title)

        # calculate vectors
        self.unit_vectors(Q_dc.shape[1])
        self.Qxy_d = np.dot(Q_dc, self.uxy_c)

        data = [
            {
                "style": self.frame_style,
                "points": self.Qxy_d,
            }
        ]

        data.append(self.u_data)

        # plot
        self.scatter_widget.set_spots(data)

        self.updating = False

# Example usage
if __name__ == "__main__":
    """
    app = QApplication(sys.argv)

    data = [
        {
            "style": {"brush": "r", "size": 10, "pen": None},
            "points": [(0, 1), (1, 2)],
            "labels": ["A", "B"]
        },
        {
            "style": {"brush": "b", "size": 8, "pen": "w"},
            "points": [(2, 2), (3, 3)],
            "labels": ["C", "D"]
        }
    ]

    win = ScatterPlotWidget("Custom Scatter")

    win.leftarrow.connect(lambda : print('left'))

    win.set_spots(data)
    win.show()

    sys.exit(app.exec_())
    """

