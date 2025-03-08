import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import argparse
from pathlib import Path
import h5py
from PyQt5 import QtGui, QtCore, QtWidgets
from collections import defaultdict
import signal
from tqdm import tqdm

from context import emc2
from emc2 import utils


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Show iteration information"""
    )
    parser.add_argument('fnam', type=str, help='iteration_info file name')
    parser.add_argument('--cxi', type=str, help='cxi file for frame viewing')
    args = parser.parse_args()

    return args


def get_sparse_fnam(cxi_file, iteration=0):
    with h5py.File(args.fnam) as f:
        key = f'iteration_{iteration}/sparse_file'
        fnam_sparse = f[key][()].decode()
    return fnam_sparse


# Get key mappings from Qt namespace
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)


# load config
args = get_args()
fnam = args.fnam

iteration = 0


def get_max_iteration():
    with h5py.File(fnam, 'r') as f:
        # find total number of iterations with slices
        max_iters = len(f.keys())
        max_iteration = 0
        for i in range(max_iters):
            k = f'iteration_{i}/model_slices'
            if k in f:
                max_iteration = i
    return max_iteration


def get_slices(fnam, iteration):
    with h5py.File(fnam, 'r') as f:
        k = f'iteration_{iteration}/model_slices'
        if k in f:
            Is = f[k][()]

            k = f'iteration_{iteration}/slice_classes'
            if k in f:
                classes = f[k][()]
            else:
                classes = None

            image, info = utils.make_models_2D_image(Is, classes)

        # what went wrong?
        else:
            print(f'{k} not in {fnam}')
            return None, None, None, None

        return image, info['positions'], info['classes'], info['N']


def get_plots(fnam, iteration):
    # get plots
    plots = []
    titles = []
    with h5py.File(fnam, 'r') as f:
        k = f'iteration_{iteration}'
        if k not in f:
            return None, None, None, None

        plots.append(f['Q'][()])
        titles.append('P logR')

        plots.append(f['beta'][()])
        titles.append('beta scaling')

        plots.append(f['P_gini'][()])
        titles.append('gini coefficient of P')

        plots.append(f['orientation_changes'][()])
        titles.append('orientation changes')

        plots.append(f['class_changes'][()])
        titles.append('class changes')

        g = f[k]
        # plots.append(np.sort(g['P_gini_d'][()])[::-1])
        # titles.append('P gini coefficient per frame')
        # plots.append(np.sort(g['Q_d'][()])[::-1])
        # titles.append('P logR per frame')

        # plots.append(np.sort(np.bincount(g['most_likely_model_d'][()])[::-1]))
        # plots.append(np.sort(g['occupancy_r'][()])[::-1])
        plots.append(g['occupancy_r'][()])
        titles.append('r-occupancy')

        R = g['occupancy_r'].shape[0]
        D = g['Q_d'].shape[0]

    return plots, titles, R, D


def get_most_likely(fnam, iteration):
    with h5py.File(fnam, 'r') as f:
        k = f'iteration_{iteration}/most_likely_model_d'
        if k not in f:
            return None

        m_d = f[k][()]
    return m_d


class GraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super(GraphicsLayoutWidget, self).__init__(*args, **kwargs)

        # get plots
        plots, titles, R, D = get_plots(fnam, iteration)

        pg_plots = []
        while True:
            for j in range(3):
                index = len(pg_plots)
                print(index, titles[index])
                pg_plots.append(self.addPlot(title=titles[index]))
                pg_plots[-1].plot(plots[index])
                if plots[index].shape[0] == R:
                    pg_plots[-1].setLabel('bottom', 'orientation')
                elif plots[index].shape[0] == D:
                    pg_plots[-1].setLabel('bottom', 'frame')

                if len(pg_plots) == len(titles):
                    break

            if len(pg_plots) == len(titles):
                break

            self.nextRow()

        self.pg_plots = pg_plots
        self.iteration = iteration

    def keyPressEvent(self, event):
        super(GraphicsLayoutWidget, self).keyPressEvent(event)
        key = keys_mapping[event.key()]
        # print("key press", key)

        if key == 'Right':
            self.update_plots(self.iteration + 1)

        elif key == 'Left':
            self.update_plots(self.iteration - 1)

    def update_plots(self, iteration):
        # get plots
        plots, self.titles, R, D = get_plots(fnam, iteration)

        if plots is None:
            return

        self.iteration = iteration

        for plot, pg_plot in zip(plots, self.pg_plots):
            pg_plot.plot(plot, clear=True)

        self.setWindowTitle(f'EMC summary: iteration {iteration}')


def show_frames(cxi_fnam, sparse_fnam, frames, iteration=0, max_frames=500):
    with h5py.File(sparse_fnam) as f:
        photon_sums = f['photon_sums'][frames]
        inds_s = np.argsort(photon_sums)[::-1]
        frame_index = f['frame_index'][frames][inds_s][:max_frames]

    with h5py.File(args.fnam) as f:
        key = f'/iteration_{iteration}/most_likely_orientation_d'
        most_likely_orientation_d = f[key][()]

        key = f'/iteration_{iteration}/most_likely_model_d'
        most_likely_model_d = f[key][()]

    for i, d in enumerate(inds_s[:max_frames]):
        print(f'{i:5} {d:5} {most_likely_model_d[d]:5}'
              f'{most_likely_orientation_d[d]:10}')

    with h5py.File(cxi_fnam) as f:
        data = f['entry_1/data_1/data']
        frames = np.zeros(
            (len(frame_index),) + data.shape[1:], dtype=np.float32
        )
        frames[:] = None

        for i, d in tqdm(enumerate(frame_index), total=frames.shape[0]):
            frames[i] = f['entry_1/data_1/data'][d]

        print('applying geometry to images')
        ims, centre = utils.Geom_corr(return_centre=True).apply(frames)
        im = pg.show(ims)
        centre[0] -= 400e-6/200e06
        centre[1] -= 4800e-6/200e06
        # transposed b/c of row-major setting
        c = pg.CircleROI(centre[::-1], [1, 1], pen=pg.mkPen('r', width=2))
        im.addItem(c)


def write_good_frames(
    iter_fnam, cxi_fnam, sparse_fnam,
    frames, good_classes, dset='/entry_1/2D_EMC/is_good'
):
    with h5py.File(sparse_fnam) as f:
        frame_index = f['frame_index'][frames]

    with h5py.File(cxi_fnam, 'r+') as f:
        data = f['entry_1/data_1/data']
        shape = data.shape
        is_good = np.zeros(shape[0], dtype=bool)
        is_good[frame_index] = True

        print(f'writing {np.sum(is_good)} is_good '
              f'labels to {cxi_fnam} in {dset}')
        if dset in f:
            f[dset][:] = is_good
        else:
            f[dset] = is_good

    # write class list to iteration info
    with h5py.File(iter_fnam, 'r+') as f:
        dset = 'good_classes'
        if dset in f:
            del f[dset]
        f[dset] = good_classes


class ImageView(pg.ImageView):
    def __init__(self, iteration, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)

        print('press "f" to display frames of class of last selection')
        print('press "s" to save selection to '
              'cxi file and good_classes.pickle')

        self.last_selected = None
        self.selection = None
        self.N = None
        self.classes = None
        self.positions = None
        self.pos = None
        self.scatter_hover = None
        self.occ_c = None
        self.most_likely_model_d = None
        self.iteration = 0
        self.pow = 0.2

        self.update_plots()

    def keyPressEvent(self, event):
        super(ImageView, self).keyPressEvent(event)
        key = keys_mapping[event.key()]
        # print("key press", key)

        if key == 'Right':
            self.update_plots(next=True)

        elif key == 'Left':
            self.update_plots(last=True)

        elif key == 'F':
            if (
                self.last_selected is not None
                and args.cxi is not None
                and self.most_likely_model_d is not None
            ):
                frames = np.where(
                    self.most_likely_model_d == self.last_selected
                )
                fnam_sparse = get_sparse_fnam(args.cxi, self.iteration)
                show_frames(args.cxi, fnam_sparse, frames, self.iteration)

        elif key == 'S':
            if (
                args.cxi is not None
                and self.most_likely_model_d is not None
                and np.any(self.selection)
            ):
                # good_classes = np.where(self.selection)[0]
                good_classes = np.unique(self.classes[self.selection])

                frames = np.where(
                    np.isin(self.most_likely_model_d, good_classes)
                )[0]

                fnam_sparse = get_sparse_fnam(args.cxi, self.iteration)
                write_good_frames(
                    args.fnam, args.cxi, fnam_sparse, frames, good_classes
                )

    def init_scatter(self):
        # set hover: fill with grey
        self.scatter_hover = pg.ScatterPlotItem(
            size=self.N,
            pen=None,
            brush=None,
            symbol='o',
            pxMode=False,
            hoverBrush=pg.mkBrush(255, 255, 255, 100),
            hoverable=True
        )
        self.addItem(self.scatter_hover)
        self.scatter_hover.sigClicked.connect(self.clicked)

        self.selection = np.zeros(len(self.classes), dtype=bool)

        # update selection if present in iteration file
        with h5py.File(args.fnam) as f:
            dset = 'good_classes'
            if dset in f:
                self.selection = np.isin(self.classes, f['good_classes'][()])

        self.update_selection()

    def clicked(self, points, ev):
        for p in ev:
            c = p.data()
            i = np.where(self.classes == c)[0]
            self.selection[i] = ~self.selection[i]
            if self.selection[c]:
                self.last_selected = c
            self.update_selection()
            self.print_number_of_events()

    def print_number_of_events(self):
        if self.occ_c is not None:
            number_of_frames = np.sum(
                self.occ_c[np.unique(self.classes[self.selection])]
            )
            print(f'number of frames selected: {number_of_frames}')

    def update_selection(self):
        spots = []
        for c in range(len(self.classes)):
            pen = pg.mkPen('g') if self.selection[c] else None
            spot = {
                'pos': self.positions[c],
                'pen': pen,
                'data': self.classes[c]
            }
            spots.append(spot)

        self.scatter_hover.setData(spots)

    def update_scatter(self):
        if self.scatter_hover is None:
            self.init_scatter()

    def update_plots(self, next=False, last=False):
        max_iteration = get_max_iteration()
        iteration = self.iteration
        im = None
        while True:
            if next:
                iteration += 1

            if last:
                iteration -= 1

            if iteration <= max_iteration and iteration >= 0:
                try:
                    im, pos, classes, N = get_slices(fnam, iteration)
                except Exception as e:
                    print(e)
                    im = None

                if im is not None:
                    break
            else:
                break

            if not (next or last):
                break

        if im is None:
            return None

        self.most_likely_model_d = get_most_likely(fnam, iteration)
        if self.most_likely_model_d is not None:
            self.occ_c = np.bincount(self.most_likely_model_d)

        self.iteration = iteration
        im[im == 0] = np.nan
        plot = self.getView()
        plot.setTitle(f'iteration: {iteration}')
        self.setImage(
            im**self.pow,
            autoRange=False,
            autoLevels=False,
            autoHistogramRange=False
        )
        # self.setImage(
        #   im, autoRange=False, autoLevels=False, autoHistogramRange=False
        # )

        self.positions = pos
        self.classes = np.array(classes)
        self.N = N

        self.update_scatter()


app = pg.mkQApp()

# Enable antialiasing for prettier plots
pg.setConfigOption('background', pg.mkColor(0.1))
pg.setConfigOption('foreground', 'w')
pg.setConfigOptions(antialias=True)
pg.setConfigOptions(imageAxisOrder='row-major')

win = GraphicsLayoutWidget(show=True)
win.resize(1000, 600)
win.setWindowTitle(f'EMC summary: iteration {iteration}')


plot = ImageView(iteration=iteration, view=pg.PlotItem())
plot.show()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow Control-C
    pg.exec()
