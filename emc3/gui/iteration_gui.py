from pathlib import Path
import numpy as np
import h5py
import pyqtgraph as pg

from .model_slices_widget import Model_slice_widget

from PyQt5.QtWidgets import (
        QApplication, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout,
        QSplitter, QDoubleSpinBox, QLabel,
        )

from .h5_viewer import H5_viewer
from .emc_scat_widget import EMC_scatter_widget
from .getters import DataGetter_h5, GeomGetterCXI_h5, GeomGetterSparseCXI_h5

from .show_fit_widget import plugin
from .orientation_widget import OrientationCoverageWidget


class CommonLinesWidget(QWidget):
    """
    Shows the best-matching line profiles for a pair of 2D EMC classes.

    Call update(class_a, class_b) to load and display the profiles from
    common_lines.h5 in the working directory.
    """

    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self._common_lines_file = self.directory / 'common_lines.h5'
        self._line_a = None
        self._line_b = None

        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('power:'))
        self._pow_spin = QDoubleSpinBox()
        self._pow_spin.setRange(0.05, 2.0)
        self._pow_spin.setSingleStep(0.05)
        self._pow_spin.setValue(0.3)
        self._pow_spin.valueChanged.connect(self._replot)
        ctrl.addWidget(self._pow_spin)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.pw = pg.PlotWidget()
        self.pw.setLabel('bottom', 'sample')
        self.pw.setLabel('left', 'intensity')
        self.pw.addLegend()
        layout.addWidget(self.pw)

        self._curve_a = self.pw.plot(pen=pg.mkPen('c', width=2), name='class A')
        self._curve_b = self.pw.plot(pen=pg.mkPen('y', width=2), name='class B')

    def _replot(self):
        if self._line_a is None:
            return
        p = self._pow_spin.value()
        x = np.arange(len(self._line_a))
        self._curve_a.setData(x, self._line_a ** p)
        self._curve_b.setData(x, self._line_b ** p)

    def update(self, class_a, class_b):
        """Load and display the best common line profiles for (class_a, class_b)."""
        fnam = self._common_lines_file
        if not fnam.is_file():
            self.pw.setTitle('common_lines.h5 not found')
            return

        with h5py.File(fnam, 'r') as f:
            class_ids = f['class_ids'][()]
            N = int(f['N_classes'][()])

            if class_a not in class_ids or class_b not in class_ids:
                self.pw.setTitle(
                    f'classes {class_a}/{class_b} not in common_lines.h5')
                return

            i = int(np.where(class_ids == class_a)[0][0])
            j = int(np.where(class_ids == class_b)[0][0])

            if i > j:
                i, j = j, i
                swapped = True
            else:
                swapped = False

            k = i * N - i * (i + 1) // 2 + j - i - 1
            line_a = f['best_lines_a'][k].astype(np.float32)
            line_b = f['best_lines_b'][k].astype(np.float32)
            score  = float(f['scores'][i, j])
            phi    = f['best_phi'][i, j]

        if swapped:
            line_a, line_b = line_b, line_a

        self._line_a = line_a
        self._line_b = line_b
        self.pw.setTitle(
            f'classes {class_a} & {class_b}  '
            f'corr={score:.4f}  angles=({phi[0]}, {phi[1]})'
        )
        self._replot()

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

        # orientation coverage in separate window
        # ----------------------------------------
        self.orientation_widget = OrientationCoverageWidget(directory,
                                                            parent=None)
        self.orientation_window = QMainWindow(parent=self)
        self.orientation_window.setWindowTitle('Orientation coverage')
        self.orientation_window.setCentralWidget(self.orientation_widget)
        self.orientation_widget.setParent(self.orientation_window)
        self.orientation_window.resize(700, 400)
        self.orientation_window.show()

        self.iterationChanged.connect(
                lambda x: self.orientation_widget.update(x)
                )
        self.scat_widget.iterationChanged.connect(
                lambda x: self.orientation_widget.update(x)
                )

        # common lines window — shown when exactly two classes are selected
        # -----------------------------------------------------------------
        self.common_lines_widget = CommonLinesWidget(directory, parent=None)
        self.common_lines_window = QMainWindow(parent=self)
        self.common_lines_window.setWindowTitle('Common lines')
        self.common_lines_window.setCentralWidget(self.common_lines_widget)
        self.common_lines_widget.setParent(self.common_lines_window)
        self.common_lines_window.resize(600, 300)

        self.selectedLabelsSignal.connect(self._on_class_selection)
        self._line_overlays  = []
        self._score_overlays = []

        # show selected frames on signal
        self.showFramesSignal.connect(self.show_frames)

    def _on_class_selection(self, labels):
        view = self.getView()
        for item in self._line_overlays + self._score_overlays:
            view.removeItem(item)
        self._line_overlays  = []
        self._score_overlays = []

        fnam = Path(self.directory) / 'common_lines.h5'

        if len(labels) >= 1 and fnam.is_file():
            self._draw_score_overlays(labels[0])

        if len(labels) == 2:
            self.common_lines_widget.update(labels[0], labels[1])
            self.common_lines_window.show()
            self.common_lines_window.raise_()
            self._draw_line_overlays(labels[0], labels[1])

    def _draw_score_overlays(self, class_id):
        fnam = Path(self.directory) / 'common_lines.h5'
        if not fnam.is_file():
            return
        if not hasattr(self, 'labels_i') or self.labels_i is None:
            return

        with h5py.File(fnam, 'r') as f:
            class_ids  = f['class_ids'][()]
            scores_mat = f['scores'][()]

        if class_id not in class_ids:
            return

        i      = int(np.where(class_ids == class_id)[0][0])
        scores = scores_mat[i]          # length N, diagonal entry = 1

        # print sorted scores (excluding self)
        other  = [(int(class_ids[j]), float(scores[j]))
                  for j in range(len(class_ids)) if j != i]
        other.sort(key=lambda x: -x[1])
        print(f'\nCommon-lines scores for class {class_id}:')
        for cid, s in other:
            print(f'  class {cid:3d}  score={s:.4f}')

        # percentile-based normalisation to show variation
        other_scores = np.array([s for _, s in other])
        vmin = float(np.percentile(other_scores, 5))
        vmax = float(np.percentile(other_scores, 95))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        view       = self.getView()
        labels_arr = np.asarray(self.labels_i)

        spots = []
        for j in range(len(class_ids)):
            if j == i:
                continue
            cls = int(class_ids[j])
            matches = np.where(labels_arr == cls)[0]
            if len(matches) == 0:
                continue
            idx    = matches[0]
            cx, cy = self.pos_i[idx]
            size   = self.N_i[idx]

            t = float(np.clip((scores[j] - vmin) / (vmax - vmin), 0.0, 1.0))
            v = int(255 * t)
            spots.append({
                'pos':   (cx, cy),
                'size':  size,
                'pen':   pg.mkPen(pg.mkColor(v, v, 0), width=3),
                'brush': None,
            })

        if spots:
            item = pg.ScatterPlotItem(pxMode=False, symbol='o')
            item.addPoints(spots)
            view.addItem(item)
            self._score_overlays.append(item)

    def _draw_line_overlays(self, class_a, class_b):
        fnam = Path(self.directory) / 'common_lines.h5'
        if not fnam.is_file():
            return
        if not hasattr(self, 'labels_i') or self.labels_i is None:
            return

        with h5py.File(fnam, 'r') as f:
            class_ids = f['class_ids'][()]
            N    = int(f['N_classes'][()])
            Nrot = int(f['Nrot'][()])
            if class_a not in class_ids or class_b not in class_ids:
                return
            i = int(np.where(class_ids == class_a)[0][0])
            j = int(np.where(class_ids == class_b)[0][0])
            swapped = i > j
            if swapped:
                i, j = j, i
            phi = f['best_phi'][i, j]   # [phi_a, phi_b]

        phi_a, phi_b = int(phi[0]), int(phi[1])
        if swapped:
            phi_a, phi_b = phi_b, phi_a

        view       = self.getView()
        labels_arr = np.asarray(self.labels_i)

        for cls, phi_idx, color in [(class_a, phi_a, 'c'), (class_b, phi_b, 'y')]:
            matches = np.where(labels_arr == cls)[0]
            if len(matches) == 0:
                continue
            idx  = matches[0]
            cx, cy = self.pos_i[idx]
            half = self.N_i[idx] / 2

            theta = phi_idx * 2 * np.pi / Nrot
            # to_gpu_2D_image uploads ar.T, so coord.x steps along axis-0
            # (rows) and coord.y along axis-1 (cols).  In plot space x=col,
            # y=row, so the line direction is (sin θ, cos θ) not (cos θ, sin θ).
            dx = half * np.sin(theta)
            dy = half * np.cos(theta)

            item = pg.PlotDataItem(
                [cx - dx, cx + dx],
                [cy - dy, cy + dy],
                pen=pg.mkPen(color, width=2),
            )
            view.addItem(item)
            self._line_overlays.append(item)

    def show_frames(self, l):
        frames, model_d, r_d = l

        # hack: should include reference to cxi file in iteration info
        cxi = list(Path(self.directory).glob('*.cxi'))[0]

        # hack: if hit_sigma is present, sort frames
        key = 'entry_1/instrument_1/detector_1/hit_sigma'
        with h5py.File(cxi) as f:
            if key in f:
                hit_sigma = f[key][()][frames]
                frames = frames[np.argsort(hit_sigma)[::-1]]

        # plugin to show model fit to frame
        # hack: need to get config file name somehow...
        config_file = Path('config.pickle')
        if config_file.is_file():
            p = lambda x: plugin(x, config_file, model_d, r_d)
        else:
            p = None

        pw = self.plots_widget.open(cxi, 'entry_1/data_1/data', where='window', plugin=p)
        pw.current_plot.update_xmap(frames)
