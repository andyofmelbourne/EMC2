"""
PlotWidget - top level plotter widget
             decides which widget to call given data

PlotItem - derived from pg.PlotItem for adding context menus

PlotItem1D - has:
             QVBox
                pg.GraphicsLayoutWidget
                    PlotItem

PlotItem2D - has:
             QVBox
                pg.GraphicsLayoutWidget
                    PlotItem
                        pg.ImageItem
                    HistogramLUTItem

PlotItem3D - has:
             QVBox
                pg.GraphicsLayoutWidget
                    PlotItem
                        pg.ImageItem
                    HistogramLUTItem
                QSlider (for selecting slices)
"""

import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
import numpy as np

from PyQt5.QtWidgets import (
        QApplication, QWidget, QHBoxLayout, QVBoxLayout, QTreeWidget,
        QTreeWidgetItem, QLabel, QSlider, QAction, QMenu, QDialog, QPushButton
        )
from PyQt5.QtCore import Qt, pyqtSignal


class PlotWidget(QWidget):
    """
    I am a top level widget that dispatches display to other widgets
    based on the data dimensions or other things

    I have a layout and keep track of the list of open PlotWidgets (which
    may be in separate windows)
    """
    def __init__(self, parent=None, open_plots=[]):

        super().__init__(parent)

        self.open_plots = open_plots
        self.open_plots.append(self)
        self.data = None

        self.current_plot = PlotWidget1D(
                parent=self,
                title='Nothing',
                open_plots=open_plots)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # remove top, left, bottom, right margins
        layout.setSpacing(0)  # remove spacing between items
        layout.addWidget(self.current_plot, 1)
        self.setLayout(layout)

    def on_parent_close(self, event):
        """
        deregister myself from the open_plots list
        """
        if self in self.open_plots:
            self.open_plots.remove(self)
        self.remove_current_plot()
        super().closeEvent(event)

    def remove_current_plot(self):
        self.current_plot.close()
        self.layout().removeWidget(self.current_plot)
        self.current_plot.setParent(None)

    def choose_widget(self, data):
        if data.ndim == 1:
            Widget = PlotWidget1D
        elif data.ndim == 2:
            Widget = PlotWidget2D
        elif data.ndim == 3:
            Widget = PlotWidget3D
        else:
            raise ValueError(f'cannot display {data.ndim=} data')

        return Widget

    def plot(self, data, name=None):
        self.remove_current_plot()

        self.data = data

        Widget = self.choose_widget(data)

        self.current_plot = Widget(
                parent=self,
                data=data,
                title=name,
                open_plots=self.open_plots
                )

        self.current_plot.plot()

        self.layout().addWidget(self.current_plot, 1)

    def refresh(self, data=None):
        if data is None:
            data = self.current_plot.data

        ndim0 = self.current_plot.data.ndim
        ndim1 = data.ndim
        if ndim0 != ndim1:
            err = f'cannot update_data with different \
                    dimensions {ndim0} {ndim1}'
            raise ValueError(err)

        # update data source without destroying widget
        self.current_plot.data = data
        self.current_plot.xmap = np.arange(len(data))

        if hasattr(self.current_plot, 'vline') and \
                self.current_plot.vline is not None:

            self.current_plot.vline.setBounds((0, data.shape[0]))

        self.current_plot.plot()


class LinkedAxis:
    """
    Link multiple vlines and xmaps accros different objects
    """
    def __init__(self, obj):
        self.linked_lines = LinkedLines([obj.vline])
        self.linked_xmaps = LinkedXmaps([[obj.xmap, obj]])
        self.objs = self.linked_xmaps.objs
        self.update_xmap = self.linked_xmaps.update_xmap

    def add(self, obj):
        self.linked_lines.add(obj.vline)
        self.linked_xmaps.add(obj.xmap, obj)

    def resolve_refs(self):
        for obj in self.objs:
            obj.linked_axis = self

    def remove(self, obj):
        self.linked_lines.remove(obj.vline)
        self.linked_xmaps.remove(obj)


def link_axis(obj, linked_axis=None):
    """
    obj.linked_axis is None
    linked_axis = None indicates user right click on obj
    """
    if linked_axis is None:
        # add obj to new linked axis pool:
        # e.g. user right click on obj
        if obj.linked_axis is None:
            linked_axis = LinkedAxis(obj)

        # nothing
        # e.g. user right click on obj with existing link
        else:
            linked_axis = obj.linked_axis

    else:
        # add obj to existing linked axis pool:
        # e.g. user right click on other obj and passed linked axis
        # to obj
        if obj.linked_axis is None:
            linked_axis.add(obj)

        # merge pools
        # then make sure all references to linked_axis are the same object
        else:
            for o in obj.linked_axis.objs:
                linked_axis.add(o)

            linked_axis.resolve_refs()

    return linked_axis


class LinkedXmaps:
    """
    Link multiple arrays accros different objects

    linked_xmaps = LinkedXmaps([xmap1, a])
    linked_xmaps.add(xmap2, b) # leads to
        linked_xmaps.xmaps is xmap2 (True)
        and a call to: a.update_xmap(xmap2)

    linked_xmaps.remove(a)
    deregisters object a from xmap list

    if a new object is registered or the array
    changes, the "update_xmap" method is called
    for all linked objects.
    """
    def __init__(self, ar_obj_list):
        self.xmap = None
        self.objs = []

        for ar, obj in ar_obj_list:
            self.add(ar, obj)

    def add(self, xmap, obj):
        if obj not in self.objs:
            self.objs.append(obj)

        # the first array takes precedence
        if self.xmap is None:
            self.xmap = xmap.copy()
        else:
            self.update_xmap()

    def update_xmap(self, xmap=None):
        if xmap is not None:
            self.xmap[:] = xmap

        for obj in self.objs:
            obj.update_xmap(self.xmap)

    def remove(self, obj):
        if obj in self.objs:
            self.objs.remove(obj)


class LinkedLines:
    """
    Link multiple InfiniteLine objects across different plots.

    Moving any line updates all others.

    calling self.update_xmap(new_xmap) results in the call:
        plot_widget.update_xmap(new_xmap) for all attached widgets

    self.xmap stores the last updated xmap value
    """
    def __init__(self, lines):
        # [ [line1, plot_widget1], [line2, plot_widget2] ]
        self.lines = []
        self._updating = False  # prevent recursive updates

        for line in lines:
            self.add(line)

    def add(self, line):
        if line not in self.lines:
            self.lines.append(line)
            line.sigPositionChanged.connect(self._line_moved)

            # to make sure they start at the same point
            self._line_moved(self.lines[0])

    def _line_moved(self, moved_line):
        if self._updating:
            return
        self._updating = True
        pos = moved_line.value()
        for line in self.lines:
            if line is not moved_line:
                line.setPos(pos)
        self._updating = False

    def remove(self, line):
        if line in self.lines:
            self.lines.remove(line)


class PlotItem(pg.PlotItem):
    def __init__(self, *args, open_plots=[], parent=None, **kwargs):
        self.open_plots = open_plots
        self.parent = parent
        super().__init__(*args, **kwargs)

        self.menu = super().getContextMenus(None)
        self.menu.addSeparator()
        self.link_menu = QMenu("Link X-axis to other plots", self.menu)
        self.menu.addMenu(self.link_menu)

        self.map_menu = QMenu("Link values to X-axis of other plots", self.menu)
        self.menu.addMenu(self.map_menu)

        sort = QAction("sort (ascending)", self.menu)
        sort.triggered.connect(self.parent.sort)
        self.menu.addAction(sort)

        sortd = QAction("sort (descending)", self.menu)
        sortd.triggered.connect(lambda x: self.parent.sort(inverse=True))
        self.menu.addAction(sortd)

        aspect = QAction("lock / unlock aspect", self.menu)
        aspect.triggered.connect(self.parent.toggle_aspect)
        self.menu.addAction(aspect)

    def getContextMenus(self, event=None):
        self.link_menu.clear()

        for other in self.open_plots:
            if other.current_plot.plot_item is self:
                continue

            my_pw = self.parent
            pw = other.current_plot

            # X map
            title = pw.title
            act_xmap = QAction(f"Link values to {title}", self.map_menu)
            # act_x.triggered.connect(lambda _, o=plot_item: self.setXLink(o))
            act_xmap.triggered.connect(lambda _, o=pw: my_pw.setXMap(o))
            self.map_menu.addAction(act_xmap)


            shape = my_pw.xmap.shape
            other_shape = pw.xmap.shape
            if shape != other_shape:
                continue

            # X link
            title = pw.title
            act_x = QAction(f"Link X to {title}", self.link_menu)
            # act_x.triggered.connect(lambda _, o=plot_item: self.setXLink(o))
            act_x.triggered.connect(lambda _, o=pw: my_pw.setXLink(o))
            self.link_menu.addAction(act_x)

        return self.menu


class PlotWidgetBase(QWidget):
    def __init__(self, parent=None, title="Data Plot", xlabel=None,
                 ylabel=None, data=None, open_plots=[], **kwargs):
        super().__init__(parent)

        self.title = title

        if data is None:
            data = np.array([])

        self.data = data
        self.xmap = np.arange(len(data))

        self.lock_aspect = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # GraphicsLayoutWidget allows adding PlotItem + LUT correctly
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Add a PlotItem
        self.plot_item = PlotItem(
                title=title,
                open_plots=open_plots,
                parent=self)

        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setLabel('bottom', xlabel)
        self.plot_item.setLabel('left', ylabel)
        self.plot_item.curve = self.plot_item.plot([], [])

        self.glw.addItem(self.plot_item)

        # Automatically rescale both axes to fit the new data
        self.plot_item.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        self.linked_axis = None
        self.vline = None
        self.current_slice = 0

    def toggle_aspect(self):
        self.lock_aspect = not self.lock_aspect
        self.plot_item.getViewBox().setAspectLocked(self.lock_aspect)

    def plot(self, data=None):
        pass

    def setXLink(self, other, linked_axis=None):
        """
        add myself to a linked axis pool
        """
        # actually this is very weird across windows
        # self.plot_item.setXLink(other.plot_item)

        if self.vline is None:
            self.add_vline()

        # attach to shared linked_axis instance
        # -------------------------------------

        # must be called self.linked_axis
        # and I must have self.update_xmap(self, xmap)
        self.linked_axis = link_axis(self, linked_axis)

        print('\nLinked Axis:')
        for obj in self.linked_axis.objs:
            print(f'{obj.title=} {obj.linked_axis}')

        # add the other guy to same pool of linked axis
        if other.linked_axis is not self.linked_axis:
            other.setXLink(self, self.linked_axis)

    def setXMap(self, other):
        """
        Use my values to xmap the other plot

        show: other_plot[my_values][:]
        """
        i = self.data[()][self.xmap]
        if i.max() < other.data.shape[0]:
            other.update_xmap(i)
        else:
            print('Error: cannot use map on other data')

    def add_vline(self):
        vline = pg.InfiniteLine(
                pos=0,
                angle=90,
                movable=True,
                bounds = [0, len(self.data)-1]
                )
        self.vline = vline
        self.plot_item.addItem(vline)

    def sort(self, inverse=False):
        # change xmapping (upates plot)
        xmap = np.argsort(self.data)
        if inverse:
            xmap = xmap[::-1]

        # inform others
        if self.linked_axis is not None:
            self.linked_axis.update_xmap(xmap)
        else:
            self.update_xmap(xmap)

    def update_xmap(self, xmap):
        self.xmap = xmap
        if self.vline is not None:
            self.vline.setBounds([0, len(xmap)-1])

        self.current_slice = xmap[self.current_slice]
        self.plot()

    def close(self):
        # deregister from linked plots
        if self.linked_axis is not None:
            self.linked_axis.remove(self)


class PlotWidget1D(PlotWidgetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, data=None):
        if data is None:
            data = self.data[()]

        # in case h5
        data = data[()][self.xmap]

        x_data = np.arange(len(data))
        self.plot_item.curve.setData(x_data, data)


class PlotWidget2D(PlotWidgetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # square pixels
        self.plot_item.getViewBox().setAspectLocked(True)

        # Add the ImageItem
        self.img_item = pg.ImageItem()
        self.plot_item.addItem(self.img_item)

        # hide grid
        self.plot_item.showGrid(x=False, y=False)

        # set defaults
        self.autoRange = True
        self.autoLevels = True
        self.autoHistogramRange = True

        # Add HistogramLUTItem (colorbar)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)

        # Put the colorbar to the right of the plot
        self.glw.addItem(self.hist)

    def plot(self, data=None, xmap=None):
        if xmap is None:
            xmap = self.xmap

        if data is None:
            data = self.data

        # in case h5
        data = data[()][xmap]

        if data.dtype == bool:
            data = data.astype(np.uint8)

        self.img_item.setImage(
                data,
                autoRange=self.autoRange,
                autoLevels=self.autoLevels,
                autoHistogramRange=self.autoHistogramRange)


class InfiniteSlider(QSlider):
    """
    make a slider that looks like a pg.InfiniteLine as well
    """
    # Custom signal that matches InfiniteLine's "sigPositionChanged"
    sigPositionChanged = pyqtSignal(object)

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)

        # Connect QSlider's signal to our custom one
        self.valueChanged.connect(self._emit_position_changed)

    def _emit_position_changed(self, value):
        """Emit our custom signal with self (similar to InfiniteLine)."""
        self.sigPositionChanged.emit(self)

    # Provide a value() alias for InfiniteLine-like API
    def value(self):
        return super().value()

    # Provide setPos alias like InfiniteLine
    def setPos(self, val):
        self.setValue(int(val))

    def setBounds(self, bounds):
        """Set (min, max) bounds for the slider."""
        if len(bounds) != 2:
            raise ValueError("Bounds must be a tuple/list of length 2: (min, max)")
        self.setMinimum(bounds[0])
        self.setMaximum(bounds[1])



class PlotWidget3D(PlotWidget2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        l2 = QHBoxLayout()

        # Label for slice number
        self.slice_label = QLabel("Slice: 0")
        l2.addWidget(self.slice_label)

        # Slider to select slice
        self.vline = InfiniteSlider(Qt.Horizontal)
        self.vline.setMinimum(0)
        self.vline.setMaximum(self.data.shape[0]-1)
        self.vline.valueChanged.connect(self._on_slider_change)
        self.vline.setValue(0)
        self.current_slice = 0
        l2.addWidget(self.vline)

        self.layout().addLayout(l2)

        self.initial_slice = True

    def _on_slider_change(self, value):
        self.current_slice = self.xmap[value]
        self.plot()

    def add_vline(self):
        # self.vline = self.slider
        pass

    def sort(self, inverse=False):
        pass

    def plot(self, data=None):
        if data is None:
            slice_data = self.data[self.current_slice]
        else:
            slice_data = data

        if self.initial_slice is False:
            self.autoRange = False
            self.autoLevels = False
            self.autoHistogramRange = False
        else:
            self.initial_slice = False

        l = f"Slice: {self.current_slice:>0{len(str(self.data.shape[0]))}}"
        self.slice_label.setText(l)

        super().plot(slice_data, xmap=())


