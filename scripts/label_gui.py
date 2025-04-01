# read in photons from each h5 file
# sort by photon count
# display

# global index is seqential list
# index events by global_index -> fnam, file_index pairs
import h5py
import numpy as np
import pyqtgraph as pg
import extra_geom
from PyQt5 import QtCore, QtWidgets
from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine
import signal
import pickle

# label = '/manual_selection/is_good'
label = '/manual_selection/is_crap'
dset_counts = '/entry_1/instrument_1/detector_1/photon_counts'
dset_frames = '/entry_1/data_1/data'
geom_fnam = '/home/andyofmelbourne/Documents/'\
            'git_repos/xfel7927/geom/r0600.geom'

DIR = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits'
# fnams = [f'{DIR}/Cube_all_hits.cxi']
fnams = [f'{DIR}/Ery_all_hits.cxi']

# geom = extra_geom.DSSC_1MGeometry.from_crystfel_geom(geom_fnam)
geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_fnam)
with h5py.File(fnams[0]) as f:
    frame = f[dset_frames][0]
    im, centre = geom.position_modules(frame)
frame_shape = frame.shape
frame_dtype = frame.dtype
image_shape = im.shape
image_dtype = np.float32

Nevents = 0
for fnam in fnams:
    with h5py.File(fnam) as f:
        data = f[dset_frames]
        Nevents += data.shape[0]

global_index = np.arange(Nevents)
photon_counts = np.empty((Nevents,), dtype=int)
file_index = np.empty((Nevents,), dtype=int)
frame_index = np.empty((Nevents,), dtype=int)
selection = np.zeros((Nevents,), dtype=bool)

index = 0
for i, fnam in enumerate(fnams):
    with h5py.File(fnam) as f:
        data = f[dset_counts]
        photon_counts[index: index + data.shape[0]] = f[dset_counts][()]
        file_index[index: index + data.shape[0]] = i
        frame_index[index: index + data.shape[0]] = np.arange(data.shape[0])

        # load previous selection
        if label in f:
            selection[index: index + data.shape[0]] = f[label][()]

        index += data.shape[0]

# sort file and frame indices by photon counts (most to least)
sorted_index = np.argsort(photon_counts)[::-1]
file_index = file_index[sorted_index]
frame_index = frame_index[sorted_index]
selection = selection[sorted_index]


# load the n'th most intense frame
def load_frame(i, frame):
    j = frame_index[i]
    fnam = fnams[file_index[i]]
    # print(f'loading frame {j} from file {fnam}')
    with h5py.File(fnam) as f:
        dset = f[dset_frames]
        print(dset.shape, frame.shape)
        # why np.s_[0]? segfault otherwise, might be bug
        dset.read_direct(frame, source_sel=np.s_[j], dest_sel=np.s_[:])


def clip_scalar(val, vmin, vmax):
    """ convenience function to avoid using np.clip for scalar values """
    return vmin if val < vmin else vmax if val > vmax else val


class Application(QtWidgets.QMainWindow):
    def __init__(self, selection):
        super().__init__()

        print("type 'r' to clear all frame selections "
              "(including those loaded from file)")
        print("type 'x' to toggle good/bad state of frame")
        print(f"type 's' to save True/False frames "
              f"list to h5 file under {label}")

        self.frame = np.zeros(frame_shape, dtype=frame_dtype)

        self.im = np.zeros(image_shape, dtype=image_dtype)

        self.selection = selection

        self.pow = 0.2
        # self.pow = 1

        self.initUI()

    def initUI(self):
        # 2D plot for the cspad and mask
        self.imageitem = pg.ImageItem()

        w = pg.GraphicsLayoutWidget()

        vb = w.addViewBox(lockAspect=True)
        vb.addItem(self.imageitem)

        cbar = pg.HistogramLUTItem(image=self.imageitem)
        cbar.gradient.loadPreset('thermal')
        w.addItem(cbar)

        self.bounds = [0, Nevents-1]

        self.timeline = InfiniteLine(0, movable=True)
        self.timeline.setPen((255, 255, 0, 200))
        self.timeline.setBounds(self.bounds)
        self.timeline.sigPositionChanged.connect(self.timeLineChanged)

        p2 = w.addPlot(row=1, col=0)
        p2.addItem(self.timeline)
        p2.hideAxis('left')
        p2.setXRange(self.bounds[0], self.bounds[1])
        w.ci.layout.setRowFixedHeight(1, 35)

        self.setCentralWidget(w)

        # display the image
        self.currentIndex = 0
        self.updateImage(init=True)

        # Display the widget as a new window
        self.resize(800, 480)
        # self.show()

    def updateImage(self, init=False):
        if init:
            autoLevels = True
        else:
            autoLevels = False

        # load frame
        load_frame(self.currentIndex, self.frame)

        # geometry correction
        geom.position_modules(self.frame, out=self.im)

        self.imageitem.setImage(
            self.im**self.pow,
            autoRange=True,
            autoLevels=autoLevels,
            autoHistogramRange=True
        )
        self.update_border()

    def show_frame(self):
        autoLevels = False

        self.imageitem.setImage(
            self.im**self.pow,
            autoRange=True,
            autoLevels=autoLevels,
            autoHistogramRange=True
        )

    def timeLineChanged(self):
        ind = int(self.timeline.value())
        if ind != self.currentIndex:
            self.currentIndex = ind
            self.updateImage()

        # self.sigTimeChanged.emit(ind)

    def keyPressEvent(self, event):
        super(Application, self).keyPressEvent(event)
        key = event.key()

        i = self.currentIndex

        if key == QtCore.Qt.Key_Left:
            ind = clip_scalar(i - 1, self.bounds[0], self.bounds[1]-1)
            self.timeline.setValue(ind)

        elif key == QtCore.Qt.Key_Right:
            ind = clip_scalar(i + 1, self.bounds[0], self.bounds[1]-1)
            self.timeline.setValue(ind)

        elif key == QtCore.Qt.Key_X:
            self.selection[i] = ~self.selection[i]
            print('total number of selected frames:', np.sum(self.selection),
                  'index', i)
            self.update_border()

        elif key == QtCore.Qt.Key_R:
            self.selection.fill(False)
            print(
                'total number of selected frames:',
                np.sum(self.selection),
                'index',
                i
            )
            self.update_border()

        elif key == QtCore.Qt.Key_S:
            self.save_selection()

    def update_border(self):
        if self.selection[self.currentIndex]:
            self.imageitem.setBorder('g')
        else:
            self.imageitem.setBorder('r')

    def save_selection(self):
        selections = {}
        for fnam in fnams:
            selections[fnam] = []

        # save a list of selected indices for each file
        for i in np.where(self.selection)[0]:
            selections[fnams[file_index[i]]].append(frame_index[i])

        for fnam in selections:
            with h5py.File(fnam, 'r+') as f:
                s = np.zeros((f[dset_counts].shape[0],), dtype=bool)
                if len(selections[fnam]) > 0:
                    s[selections[fnam]] = True
                if label in f:
                    f[label][:] = s
                else:
                    f[label] = s
                print(f'finished writing {len(selections[fnam])} '
                      f'selected labels to {fnam}')

                fnam_pickle = f'{label}_backup.pickle'

                pickle.dump({label: s}, open(fnam_pickle.split('/')[-1], 'wb'))
                print(f'finished writing {label} backup to {fnam_pickle}')
        print('done')


signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow Control-C

# Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])

a = Application(selection)
a.show()

# Start the Qt event loop
app.exec_()
