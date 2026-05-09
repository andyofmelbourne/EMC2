"""
Orientation coverage widget.

Displays a rectangular longitude × latitude histogram with three modes:
  • Most likely  — hard-assignment count from most_likely_orientation_d
  • Occupancy    — soft sum_d P_dr from occupancy_r, using r_offset per class
  • Sampling     — uniform count of symmetry-expanded orientation points per bin

local_rmax_d is a flat index into (J, K, L) with L varying fastest, so
the fundamental-zone index is recovered as  l = local_rmax_d % L.
occupancy_r is a global array spanning all classes; class c occupies
  occupancy_r[r_offset : r_offset + J*K*L].
"""
import pickle
from pathlib import Path

import h5py
import numpy as np

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QRadioButton,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QSplitter
)


def _voxel_sign_patterns(symmetry_str):
    I   = np.diag([+1, +1, +1]).astype(float)
    C2z = np.diag([-1, -1, +1]).astype(float)
    C2x = np.diag([+1, -1, -1]).astype(float)
    inv = np.diag([-1, -1, -1]).astype(float)

    if symmetry_str == 'D6':
        generators = [C2z, C2x, inv]
    elif symmetry_str == 'C6':
        generators = [C2z, inv]
    elif symmetry_str == 'inversion':
        generators = [inv]
    else:
        generators = []

    ops = [I]
    changed = True
    while changed:
        changed = False
        for g in list(ops):
            for h in generators:
                new = g @ h
                if not any(np.allclose(new, e) for e in ops):
                    ops.append(new)
                    changed = True

    return np.stack(ops)


class OrientationCoverageWidget(QWidget):

    def __init__(self, directory, parent=None, n_lon=180, n_lat=90):
        super().__init__(parent)
        self.directory = Path(directory)
        self.fnam = self.directory / 'iteration_info.h5'

        self.R = 0
        self.S = 1
        self.J = 1
        self.K = 1
        self.V = 1
        self.n_classes  = 1
        self.n_lon      = n_lon
        self.n_lat      = n_lat
        self.lon_idx    = None
        self.lat_idx    = None
        self._phi_all   = None
        self._lat_all   = None
        self._sampling_hist   = None
        self._r_offsets = []   # r_offset per class (global occupancy_r index)
        self._R_cs      = []   # J*K*L per class
        self.current_images   = None
        self._last_iter_key   = None
        self._levels_set      = False
        self._mlo_d_raw       = None
        self._mlm_d_raw       = None
        self._occ_r_global    = None
        self._has_hard        = False
        self._has_soft        = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- control row ---
        ctrl = QWidget()
        ctrl_row = QHBoxLayout(ctrl)
        ctrl_row.setContentsMargins(4, 2, 4, 2)

        ctrl_row.addWidget(QLabel('Class:'))
        self.class_spin = QSpinBox()
        self.class_spin.setRange(0, 0)
        self.class_spin.valueChanged.connect(self._show_class)
        ctrl_row.addWidget(self.class_spin)

        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QLabel('Bins (lon):'))
        self.lon_spin = QSpinBox()
        self.lon_spin.setRange(18, 1440)
        self.lon_spin.setSingleStep(18)
        self.lon_spin.setValue(self.n_lon)
        self.lon_spin.valueChanged.connect(self._on_bins_changed)
        ctrl_row.addWidget(self.lon_spin)

        ctrl_row.addWidget(QLabel('× lat:'))
        self.lat_spin = QSpinBox()
        self.lat_spin.setRange(9, 720)
        self.lat_spin.setSingleStep(9)
        self.lat_spin.setValue(self.n_lat)
        self.lat_spin.valueChanged.connect(self._on_bins_changed)
        ctrl_row.addWidget(self.lat_spin)

        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QLabel('Scale:'))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(0, 0)
        self.scale_spin.setEnabled(False)
        self.scale_spin.valueChanged.connect(self._on_scale_changed)
        ctrl_row.addWidget(self.scale_spin)
        self.scale_all_cb = QCheckBox('All')
        self.scale_all_cb.setChecked(True)
        self.scale_all_cb.toggled.connect(self._on_scale_all_toggled)
        ctrl_row.addWidget(self.scale_all_cb)

        ctrl_row.addSpacing(16)
        self.normalise_cb = QCheckBox('Normalise by sampling')
        self.normalise_cb.setChecked(False)
        self.normalise_cb.toggled.connect(lambda: self._show_class(self.class_spin.value()))
        ctrl_row.addWidget(self.normalise_cb)

        ctrl_row.addSpacing(16)
        self.mode_group = QButtonGroup(self)
        for label in ('Most likely', 'Occupancy', 'Sampling'):
            rb = QRadioButton(label)
            self.mode_group.addButton(rb)
            ctrl_row.addWidget(rb)
        self.mode_group.buttons()[0].setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        self.title_label = QLabel()
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(self.title_label)
        ctrl_row.addStretch()
        layout.addWidget(ctrl, 0)

        # --- plot + histogram LUT ---
        self.pw = pg.PlotWidget()
        self.pw.setLabel('bottom', 'Longitude (°)')
        self.pw.setLabel('left', 'Latitude (°)')
        self.pw.setXRange(-180, 180, padding=0.02)
        self.pw.setYRange(-90, 90, padding=0.02)

        self.img_item = pg.ImageItem()
        self.pw.addItem(self.img_item)

        self.hist = pg.HistogramLUTWidget(orientation='vertical')
        self.hist.setImageItem(self.img_item)
        self.hist.setMaximumWidth(120)
        cmap = pg.colormap.get('inferno', source='matplotlib')
        self.hist.gradient.setColorMap(cmap)

        plot_row = QSplitter(Qt.Horizontal)
        plot_row.addWidget(self.pw)
        plot_row.addWidget(self.hist)
        plot_row.setStretchFactor(0, 1)
        plot_row.setStretchFactor(1, 0)
        layout.addWidget(plot_row, 1)

        self._load_rotations()
        self._try_initial_update()

    # ------------------------------------------------------------------
    def _load_rotations(self):
        config_file = self.directory / 'config.pickle'
        if not config_file.is_file():
            return

        with open(config_file, 'rb') as f:
            config = pickle.load(f)

        mapper = config['classes'][0]['mapper']
        if mapper.dimensions != 3:
            return

        self.n_classes = len(config['classes'])
        symmetry_str   = config['classes'][0]['model'].symmetry

        S, J, K, L = mapper.M_sjkl.shape[:4]
        self.S = S
        self.J = J
        self.K = K
        self.R = L

        self.scale_spin.setRange(0, K - 1)
        self.scale_spin.setValue(0)
        self.scale_all_cb.setChecked(True)
        self.scale_spin.setEnabled(False)

        # per-class global r_offset and total orientations J*K*L
        self._r_offsets = [c.get('r_offset', 0)      for c in config['classes']]
        self._R_cs      = [c['mapper'].shape[1]       for c in config['classes']]

        z_unnorm = mapper.M_sjkl[:, 0, 0, :, :3, 2].reshape(S * L, 3)
        norms    = np.linalg.norm(z_unnorm, axis=-1, keepdims=True)
        z_unit   = z_unnorm / np.where(norms == 0, 1.0, norms)

        sign_mats  = _voxel_sign_patterns(symmetry_str)
        self.V     = len(sign_mats)
        z_all      = np.einsum('vij,nj->vni', sign_mats, z_unit).reshape(-1, 3)

        self._phi_all = np.arctan2(z_all[:, 1], z_all[:, 0])
        self._lat_all = np.arcsin(np.clip(z_all[:, 2], -1.0, 1.0))

        self.class_spin.setRange(0, self.n_classes - 1)
        self._rebin(self.n_lon, self.n_lat)

    # ------------------------------------------------------------------
    def _rebin(self, n_lon, n_lat):
        if self._phi_all is None:
            return

        self.n_lon = n_lon
        self.n_lat = n_lat

        lon_edges = np.linspace(-np.pi,   np.pi,   n_lon + 1)
        lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)

        self.lon_idx = np.clip(np.digitize(self._phi_all, lon_edges) - 1, 0, n_lon - 1)
        self.lat_idx = np.clip(np.digitize(self._lat_all, lat_edges) - 1, 0, n_lat - 1)

        s_hist = np.zeros((n_lat, n_lon), dtype=np.float32)
        np.add.at(s_hist, (self.lat_idx, self.lon_idx), 1)
        self._sampling_hist = s_hist

        tr = QTransform(360 / n_lon, 0, 0, 180 / n_lat, -180, -90)
        self.img_item.setTransform(tr)

    # ------------------------------------------------------------------
    def _mode(self):
        """Return 'most_likely', 'occupancy', or 'sampling'."""
        checked = self.mode_group.checkedButton()
        if checked is None:
            return 'most_likely'
        return checked.text().lower().replace(' ', '_')

    def _on_mode_changed(self):
        self._show_class(self.class_spin.value())

    def _on_bins_changed(self):
        self._rebin(self.lon_spin.value(), self.lat_spin.value())
        if self._last_iter_key is not None:
            self.update(self._last_iter_key)

    # ------------------------------------------------------------------
    def _occ_to_image(self, occ_r):
        """occ_r : (L,) → (n_lat, n_lon) histogram."""
        occ_VSL = np.tile(np.tile(occ_r, self.S), self.V)
        hist = np.zeros((self.n_lat, self.n_lon), dtype=np.float32)
        np.add.at(hist, (self.lat_idx, self.lon_idx), occ_VSL)
        return hist

    def _display_image(self, img):
        if self.normalise_cb.isChecked() and self._sampling_hist is not None:
            s = self._sampling_hist
            img = np.where(s > 0, img / s, 0.0)
        vmin = img[img > 0].min() if np.any(img > 0) else 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            img_log = np.where(img > 0, np.log10(img), np.log10(vmin) - 1)
        self.img_item.setImage(img_log, autoLevels=not self._levels_set)
        self._levels_set = True

    def _show_class(self, c):
        mode = self._mode()
        if mode == 'sampling':
            if self._sampling_hist is not None:
                self._display_image(self._sampling_hist)
            return
        if self.current_images is not None and c < len(self.current_images):
            self._display_image(self.current_images[c])

    # ------------------------------------------------------------------
    def _try_initial_update(self):
        if not self.fnam.is_file():
            return
        with h5py.File(self.fnam, 'r') as f:
            n = int(f.get('iterations', np.array(0))[()])
        if n > 0:
            self.update(f'iteration_{n - 1}')

    # ------------------------------------------------------------------
    def _on_scale_all_toggled(self, checked):
        self.scale_spin.setEnabled(not checked)
        self._on_scale_changed()

    def _on_scale_changed(self):
        self._rebuild_images()
        self._show_class(self.class_spin.value())

    def _selected_k(self):
        """Return scale index, or None if 'All' is checked."""
        if self.scale_all_cb.isChecked():
            return None
        return self.scale_spin.value()

    # ------------------------------------------------------------------
    def update(self, iteration_key):
        if self.lon_idx is None:
            return
        if not self.fnam.is_file():
            return

        with h5py.File(self.fnam, 'r') as f:
            if iteration_key not in f:
                return
            grp = f[iteration_key]

            self._has_hard = ('most_likely_orientation_d' in grp and
                              'most_likely_model_d'       in grp)
            self._has_soft = 'occupancy_r' in grp

            if self._has_hard:
                self._mlo_d_raw = grp['most_likely_orientation_d'][()]
                self._mlm_d_raw = grp['most_likely_model_d'][()]
            if self._has_soft:
                self._occ_r_global = grp['occupancy_r'][()]

        if not self._has_hard and not self._has_soft:
            return

        self._last_iter_key = iteration_key
        self.title_label.setText(iteration_key)
        self._rebuild_images()

    def _rebuild_images(self):
        if not self._has_hard and not self._has_soft:
            return

        k = self._selected_k()
        images_hard = []
        images_soft = []

        for c in range(self.n_classes):
            # --- most likely (hard assignment) ---
            if self._has_hard:
                mask  = self._mlm_d_raw == c
                r_idx = self._mlo_d_raw[mask]
                l_idx = r_idx % self.R
                if k is not None:
                    k_idx = (r_idx // self.R) % self.K
                    l_idx = l_idx[k_idx == k]
                occ_r = np.zeros(self.R, dtype=float)
                np.add.at(occ_r, l_idx, 1)
                images_hard.append(self._occ_to_image(occ_r))

            # --- occupancy (soft, global occupancy_r sliced by r_offset) ---
            if self._has_soft:
                r0  = self._r_offsets[c]
                R_c = self._R_cs[c]
                occ_jkl = self._occ_r_global[r0 : r0 + R_c].reshape(self.J, self.K, self.R)
                if k is None:
                    occ_r = occ_jkl.sum(axis=(0, 1))
                else:
                    occ_r = occ_jkl[:, k, :].sum(axis=0)
                images_soft.append(self._occ_to_image(occ_r))

        self._images_hard = images_hard if images_hard else None
        self._images_soft = images_soft if images_soft else None

        mode = self._mode()
        if mode == 'most_likely':
            self.current_images = self._images_hard
        else:
            self.current_images = self._images_soft

        self._show_class(self.class_spin.value())

    # ------------------------------------------------------------------
    def _on_mode_changed(self):
        mode = self._mode()
        if mode == 'most_likely':
            self.current_images = getattr(self, '_images_hard', None)
        elif mode == 'occupancy':
            self.current_images = getattr(self, '_images_soft', None)
        self._show_class(self.class_spin.value())
