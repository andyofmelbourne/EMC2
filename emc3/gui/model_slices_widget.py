"""
Show model slices as montage

They are selectable with click

Arrow keys change iteration number

Button or keypress saves class selection and frame list

---------------------------------------------------------------------
sequence of 2D arrays + labels
    --> Display Widget
        --> selected labels
---------------------------------------------------------------------
    - keeps track of scatter plot item vs class number
    - show model slices
    - hover outlines class
    - click adds / removes class to selection
    - button click / keyress calls external function
    - left right arrow loads previous next iteration
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
import fnmatch
import re


# Get key mappings from Qt namespace
qt_keys = (
    (getattr(Qt, attr), attr[4:])
    for attr in dir(Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)


class ImageView(pg.ImageView):
    """
    takes array_getter with call signiture:
        array_getter.next()     -> title, (arrays_i, labels_i, info_i)
        array_getter.previous() -> title, (arrays_i, labels_i, info_i)

    where:
        title is a string to display at the top of the plot
        arrays_i is a sequence of 2D arrays of the same shape
        labels_i is a sequence of array labels
        info_i is a sequence of dictionaries with information
            that will be displayed on hover or click
    """
    selectedLabelsSignal = pyqtSignal(list)

    def __init__(self, array_getter, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)

        self.pow = 0.2
        self.selected_labels = []
        self.scatter_hover = None
        self.array_getter = array_getter
        self.updating = False
        self.update_plots(next=True)

    def keyPressEvent(self, event):
        super(ImageView, self).keyPressEvent(event)
        key = keys_mapping[event.key()]

        if key == 'Right':
            self.update_plots(next=True)

        elif key == 'Left':
            self.update_plots(last=True)

        elif key == 'S':
            self.saveLabels(self.selected_labels)

        elif key == 'F':
            self.showFrames(self.selected_labels)

    def saveLabels(self, labels):
        print('selected labels:', labels)

    def showFrames(self, labels):
        print('selected labels:', labels)

    def update_plots(self, current=False, next=False, last=False, key=False):
        if self.updating:
            return

        self.updating = True

        if current:
            title, (arrays_i, labels_i, info_i) = self.array_getter._increment(inc=0)

        if next:
            title, (arrays_i, labels_i, info_i) = self.array_getter.next()

        if last:
            title, (arrays_i, labels_i, info_i) = self.array_getter.previous()

        if key:
            title, (arrays_i, labels_i, info_i) = \
                self.array_getter._increment(gkey=key)

        plot = self.getView()
        plot.setTitle(title)

        if (
                arrays_i is None
                or labels_i is None
                or info_i is None
                ):
            self.updating = False
            return None

        self.info_i = info_i
        self.labels_i = labels_i

        im, self.pos_i, self.N_i = make_2D_image(arrays_i)
        im[im == 0] = np.nan

        self.setImage(
            im**self.pow,
            autoRange=False,
            autoLevels=False,
            autoHistogramRange=False
        )

        self.update_scatter(
                self.pos_i,
                self.labels_i,
                self.N_i,
                self.info_i
                )

        self.updating = False

    def update_scatter(self, pos_i, labels_i, N_i, info_i):
        if self.scatter_hover is not None:
            self.removeItem(self.scatter_hover)

        # set hover: fill with grey
        self.scatter_hover = pg.ScatterPlotItem(
            pen=None,
            brush=None,
            symbol='o',
            pxMode=False,
            hoverBrush=pg.mkBrush(255, 255, 255, 100),
            hoverable=True
        )
        # size=info['N'],

        self.addItem(self.scatter_hover)
        self.scatter_hover.sigClicked.connect(self.clicked)

        spots = []
        for i, label in enumerate(labels_i):
            pen = pg.mkPen('g') if label in self.selected_labels else None
            data = {'class_id': label}
            data.update(info_i[i])
            spot = {
                'size': N_i[i],
                'pos': pos_i[i],
                'pen': pen,
                'data': data
            }
            spots.append(spot)

        self.scatter_hover.setData(spots)

    def clicked(self, points, ev):
        for p in ev:
            c = p.data()['class_id']
            if c in self.selected_labels:
                self.selected_labels.remove(c)
            else:
                self.selected_labels.append(c)

        self.selectedLabelsSignal.emit(self.selected_labels)

        self.update_scatter(
                self.pos_i,
                self.labels_i,
                self.N_i,
                self.info_i
                )


def make_2D_image(arrays_i):
    # calculate grid
    n = math.ceil(len(arrays_i)**0.5)

    N = arrays_i[0].shape[0]

    im = np.zeros((n * N, n * N), dtype=np.float32)

    # output centre positions
    positions_i = []

    for i in range(n):
        for j in range(n):
            c = n*i + j
            if c < len(arrays_i):
                im[i*N: (i+1)*N, j*N: (j+1)*N] = arrays_i[c]
                x = N * (c % n) + N/2 + 0.5
                y = N * (c // n) + N/2 + 0.5
                positions_i.append((x, y))

    # must be equal for now
    N_i = N * np.ones((len(arrays_i),), dtype=int)

    return im, positions_i, N_i


class BaseIterationInfo_getter(QObject):
    """
    Given an iteration_info.h5 file (like below) provide the following:
        a = BaseIterationInfo_getter(
            fnam='iteration_info.h5'
            keys=['Q_d', 'model_slices']
            )

        i, Q_d, model_slices = self.next() # i = 'iteration_0'
        i, Q_d, model_slices = self.next() # i = 'iteration_1', model_slices = None
        i, Q_d, model_slices = self.next() # i = 'iteration_3'
        i, Q_d, model_slices = self.next() # i = 'iteration_0'

    $ h5ls -r iteration_info.h5
    /                        Group
    /P_gini                  Dataset {20/Inf}
    /Q                       Dataset {20/Inf}
    /beta                    Dataset {20/Inf}
    /class_changes           Dataset {20/Inf}
    /iteration_0             Group
    /iteration_0/Q_d         Dataset {3000}
    /iteration_0/model_slices Dataset {16, 64, 64}
    /iteration_1             Group
    /iteration_1/Q_d         Dataset {3000}
    /iteration_3             Group
    /iteration_3/Q_d         Dataset {3000}
    /iteration_3/model_slices Dataset {16, 64, 64}
    """
    iterationChanged = pyqtSignal(object)

    def __init__(self, fnam, keys=[]):
        self.iteration = ''
        self.iteration_index = -1
        self.iteration_list = []
        self.fnam = fnam
        self.keys = keys
        self.updating = False
        super().__init__()

    def next(self):
        return self._increment(+1)

    def previous(self):
        return self._increment(-1)

    def _increment(self, inc=1, gkey=None):
        self._update_iterations()

        if gkey is None:
            i0 = (self.iteration_index+inc) % len(self.iteration_list)
            self.iteration_index = i0

            gkey = self.iteration_list[i0]
            self.iteration = gkey

        else:
            self.iteration_index = self.iteration_list.index(gkey)
            self.iteration = gkey

        d = self._get_data(gkey)

        self.iterationChanged.emit(gkey)
        return d

    def _get_data(self, gkey=None, keys=None):
        if gkey is None:
            gkey = self.iteration

        if keys is None:
            keys = self.keys

        data = [gkey,]

        with h5py.File(self.fnam) as f:
            if gkey not in f:
                raise ValueError(f'could not find {gkey} in {self.fnam}')
            else:
                g = f[gkey]

            for key in keys:
                data.append(self._get_from_h5(g, key))
        return data

    def _get_from_h5(self, h5obj, key):
        if key in h5obj:
            if isinstance(h5obj[key], h5py.Dataset):
                data = h5obj[key][()]
            elif isinstance(h5obj[key], h5py.Group):
                data = {}
                for k in h5obj[key].keys():
                    data[k] = self._get_from_h5(h5obj[key], k)
        else:
            data = None

        return data

    def _natural_key(self, s):
        # Split into parts: text stays as str, digits become int
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

    def _sort_select(self, strings, pattern):
        matches = [s for s in strings if fnmatch.fnmatch(s, pattern)]
        return sorted(matches, key=self._natural_key)

    def _update_iterations(self):
        with h5py.File(self.fnam) as f:
            keys = list(f.keys())

        self.iteration_list = self._sort_select(keys, 'iteration_*')


class SliceGetter(QObject):
    def __init__(self, fnam):
        self.fnam = fnam
        self.keys = ['model_slices',
                     'slice_classes',
                     'most_likely_model_d']

        self.itget = BaseIterationInfo_getter(fnam, self.keys)

        self.iterationChanged = self.itget.iterationChanged

        self.title = 'iteration {}'
        super().__init__()

    def _increment(self, inc=1, gkey=None):
        t, slices, slices_id, model_d = \
                self.itget._increment(inc=inc, gkey=gkey)

        # add number of frames per slice to info
        if model_d is not None and slices_id is not None:
            frames_c = np.bincount(model_d, minlength=len(slices_id))[slices_id]
            info = [{'frames': i} for i in frames_c]
        else:
            info = None
        return t, (slices, slices_id, info)

    def next(self):
        return self._increment(+1)

    def previous(self):
        return self._increment(-1)


class Model_slice_widget(ImageView):
    showFramesSignal = pyqtSignal(object)

    def __init__(self, directory, parent=None):
        fnam = Path(directory) / 'iteration_info.h5'

        if not fnam.is_file:
            raise ValueError(f'could not find {fnam}')

        self.slice_getter = SliceGetter(fnam)
        self.iterationChanged = self.slice_getter.iterationChanged

        super().__init__(self.slice_getter, view=pg.PlotItem())

        self.class_labels_key = 'class_labels' # class labels for full dataset
        self.class_labels_file = 'selected_classes_multi.h5' # class labels for full dataset

        # load previously saved selection
        if Path(self.class_labels_file).is_file():
            with h5py.File(self.class_labels_file) as f:
                self.selected_labels = list(f['selected_classes'][()])
                self.update_plots(current=True)


    def get_frames(self, labels):
        gkey, class_labels = self.slice_getter.itget._get_data(
            keys=[self.class_labels_key,]
        )

        selected_frames_full = []
        for c in labels:
            selected_frames_full.append(np.where(class_labels==c)[0])

        selected_frames_full = np.concatenate(selected_frames_full)

        print(selected_frames_full)
        print(labels)
        print('number of frames selected:', len(selected_frames_full))
        print('saving selection to:', self.class_labels_file)
        return gkey, selected_frames_full

    def showFrames(self, labels):
        gkey, selected_frames_full = self.get_frames(labels)

        self.showFramesSignal.emit(selected_frames_full)

    def saveLabels(self, labels):
        gkey, selected_frames_full = self.get_frames(labels)

        with h5py.File(self.class_labels_file, 'w') as f:
            f['selected_classes'] = np.array(labels)
            f['selected_frames'] = selected_frames_full
            f['source_file'] = str(self.fnam)
            f['source_key'] = str(gkey)


if __name__ == '__main__':
    fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/Ery/recon_2D_01/iteration_info.h5'

    app = pg.mkQApp()
    # Enable antialiasing for prettier plots
    # pg.setConfigOption('background', pg.mkColor(0.1))
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    pg.setConfigOptions(antialias=True)
    pg.setConfigOptions(imageAxisOrder='row-major')

    slice_getter = SliceGetter(fnam)

    plot = ImageView(slice_getter, view=pg.PlotItem())
    plot.show()

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow Control-C
    pg.exec()
