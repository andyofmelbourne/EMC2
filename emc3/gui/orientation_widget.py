"""
Orientation coverage widget — per-class, per-iteration.

3-D classes: rectangular longitude × latitude histogram with three modes:
  • Most likely  — hard-assignment count from most_likely_orientation_d
  • Occupancy    — soft sum_d P_dr from occupancy_r
  • Sampling     — uniform count of symmetry-expanded orientation points

2-D classes: bar chart of occupancy vs in-plane angle (degrees).

Geometry is read from iteration_N/mapper/class_C/ in iteration_info.h5
(written by input_output.save_iteration_info when config is supplied).
Falls back to config.pickle for iterations saved without geometry.

local_rmax_d is a flat index into (J, K, L) with L varying fastest.
occupancy_r spans all classes; class c occupies
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

        self.n_lon           = n_lon
        self.n_lat           = n_lat
        self.n_classes       = 0
        self._class_info     = []   # per-class geometry, current iteration
        self._config_class_info = [] # fallback from config.pickle
        self._R_cs           = []   # J*K*L per class
        self.current_images  = None
        self._last_iter_key  = None
        self._levels_set     = False
        self._mlo_d_raw      = None
        self._mlm_d_raw      = None
        self._occ_r_global   = None
        self._has_hard       = False
        self._has_soft       = False

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

        # --- 3-D lon/lat map + histogram LUT ---
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

        self.plot_row = QSplitter(Qt.Horizontal)
        self.plot_row.addWidget(self.pw)
        self.plot_row.addWidget(self.hist)
        self.plot_row.setStretchFactor(0, 1)
        self.plot_row.setStretchFactor(1, 0)
        layout.addWidget(self.plot_row, 1)

        # --- 2-D angle bar chart ---
        self.line_pw = pg.PlotWidget()
        self.line_pw.setLabel('bottom', 'Angle (°)')
        self.line_pw.setLabel('left', 'Occupancy')
        self.line_pw.hide()
        layout.addWidget(self.line_pw, 1)

        self._load_config_rotations()
        self._try_initial_update()

    # ------------------------------------------------------------------
    # Geometry loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_3d_class_info(mapper, symmetry_str, r_offset):
        S, J, K, L = mapper.M_sjkl.shape[:4]
        z_unnorm = mapper.M_sjkl[:, 0, 0, :, :3, 2].reshape(S * L, 3)
        norms    = np.linalg.norm(z_unnorm, axis=-1, keepdims=True)
        z_unit   = z_unnorm / np.where(norms == 0, 1.0, norms)
        sign_mats = _voxel_sign_patterns(symmetry_str)
        V = len(sign_mats)
        z_all = np.einsum('vij,nj->vni', sign_mats, z_unit).reshape(-1, 3)
        return {
            'dimensions': 3, 'J': J, 'K': K, 'L': L, 'S': S, 'V': V,
            'r_offset': r_offset,
            'phi_all': np.arctan2(z_all[:, 1], z_all[:, 0]),
            'lat_all': np.arcsin(np.clip(z_all[:, 2], -1.0, 1.0)),
            'lon_idx': None, 'lat_idx': None, 'sampling_hist': None,
            'angles_2d': None,
        }

    @staticmethod
    def _make_2d_class_info(mapper, r_offset):
        S, J, K, L = mapper.M_sjkl.shape[:4]
        M2    = mapper.M_sjkl[0, 0, 0, :, :2, :2]
        norms = np.hypot(M2[:, 0, 0], M2[:, 1, 0])
        norms = np.where(norms == 0, 1.0, norms)
        angles_2d = np.degrees(np.arctan2(M2[:, 1, 0] / norms, M2[:, 0, 0] / norms))
        return {
            'dimensions': 2, 'J': J, 'K': K, 'L': L, 'S': S, 'V': 1,
            'r_offset': r_offset,
            'phi_all': None, 'lat_all': None,
            'lon_idx': None, 'lat_idx': None, 'sampling_hist': None,
            'angles_2d': angles_2d,
        }

    @staticmethod
    def _class_info_from_h5_group(c_grp):
        dims     = int(c_grp.attrs['dimensions'])
        J        = int(c_grp.attrs['J'])
        K        = int(c_grp.attrs['K'])
        L        = int(c_grp.attrs['L'])
        r_offset = int(c_grp.attrs['r_offset'])
        if dims == 3:
            z_unit       = c_grp['z_unit'][()]
            symmetry_str = str(c_grp.attrs['symmetry'])
            S            = z_unit.shape[0] // L
            sign_mats    = _voxel_sign_patterns(symmetry_str)
            V            = len(sign_mats)
            z_all        = np.einsum('vij,nj->vni', sign_mats, z_unit).reshape(-1, 3)
            return {
                'dimensions': 3, 'J': J, 'K': K, 'L': L, 'S': S, 'V': V,
                'r_offset': r_offset,
                'phi_all': np.arctan2(z_all[:, 1], z_all[:, 0]),
                'lat_all': np.arcsin(np.clip(z_all[:, 2], -1.0, 1.0)),
                'lon_idx': None, 'lat_idx': None, 'sampling_hist': None,
                'angles_2d': None,
            }
        else:
            angles_2d = c_grp['angles_deg'][()]
            return {
                'dimensions': 2, 'J': J, 'K': K, 'L': L, 'S': 1, 'V': 1,
                'r_offset': r_offset,
                'phi_all': None, 'lat_all': None,
                'lon_idx': None, 'lat_idx': None, 'sampling_hist': None,
                'angles_2d': angles_2d,
            }

    def _load_config_rotations(self):
        config_file = self.directory / 'config.pickle'
        if not config_file.is_file():
            return
        with open(config_file, 'rb') as f:
            config = pickle.load(f)

        class_info = []
        for c_cfg in config['classes']:
            mapper   = c_cfg['mapper']
            r_offset = c_cfg.get('r_offset', 0)
            if mapper.dimensions == 3:
                ci = self._make_3d_class_info(mapper, c_cfg['model'].symmetry, r_offset)
            elif mapper.dimensions == 2:
                ci = self._make_2d_class_info(mapper, r_offset)
            else:
                continue
            class_info.append(ci)

        self._config_class_info = class_info
        self._apply_class_info(class_info)

    # ------------------------------------------------------------------
    # Applying geometry
    # ------------------------------------------------------------------

    def _apply_class_info(self, class_info):
        self._class_info = class_info
        self.n_classes   = len(class_info)
        if not class_info:
            return

        self._R_cs = [ci['J'] * ci['K'] * ci['L'] for ci in class_info]

        max_K = max(ci['K'] for ci in class_info)
        self.class_spin.setRange(0, self.n_classes - 1)
        self.scale_spin.setRange(0, max_K - 1)

        any_3d = any(ci['dimensions'] == 3 for ci in class_info)
        self.lon_spin.setEnabled(any_3d)
        self.lat_spin.setEnabled(any_3d)
        self.normalise_cb.setEnabled(any_3d)
        for rb in self.mode_group.buttons():
            if rb.text() == 'Sampling':
                rb.setEnabled(any_3d)

        tr = QTransform(360 / self.n_lon, 0, 0, 180 / self.n_lat, -180, -90)
        self.img_item.setTransform(tr)

        for ci in class_info:
            if ci['dimensions'] == 3:
                self._rebin_class(ci, self.n_lon, self.n_lat)

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def _rebin(self, n_lon, n_lat):
        self.n_lon = n_lon
        self.n_lat = n_lat
        tr = QTransform(360 / n_lon, 0, 0, 180 / n_lat, -180, -90)
        self.img_item.setTransform(tr)
        for ci in self._class_info:
            if ci['dimensions'] == 3:
                self._rebin_class(ci, n_lon, n_lat)

    @staticmethod
    def _rebin_class(ci, n_lon, n_lat):
        lon_edges = np.linspace(-np.pi,   np.pi,   n_lon + 1)
        lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
        ci['lon_idx'] = np.clip(np.digitize(ci['phi_all'], lon_edges) - 1, 0, n_lon - 1)
        ci['lat_idx'] = np.clip(np.digitize(ci['lat_all'], lat_edges) - 1, 0, n_lat - 1)
        s_hist = np.zeros((n_lat, n_lon), dtype=np.float32)
        np.add.at(s_hist, (ci['lat_idx'], ci['lon_idx']), 1)
        ci['sampling_hist'] = s_hist

    def _on_bins_changed(self):
        self._rebin(self.lon_spin.value(), self.lat_spin.value())
        if self._last_iter_key is not None:
            self.update(self._last_iter_key)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _mode(self):
        checked = self.mode_group.checkedButton()
        if checked is None:
            return 'most_likely'
        return checked.text().lower().replace(' ', '_')

    def _occ_to_image(self, occ_l, ci):
        """(L,) occupancy → (n_lat, n_lon) histogram for a 3-D class."""
        occ_VSL = np.tile(np.tile(occ_l, ci['S']), ci['V'])
        hist = np.zeros((self.n_lat, self.n_lon), dtype=np.float32)
        np.add.at(hist, (ci['lat_idx'], ci['lon_idx']), occ_VSL)
        return hist

    def _display_image(self, img, ci):
        if self.normalise_cb.isChecked() and ci.get('sampling_hist') is not None:
            s   = ci['sampling_hist']
            img = np.where(s > 0, img / s, 0.0)
        vmin = img[img > 0].min() if np.any(img > 0) else 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            img_log = np.where(img > 0, np.log10(img), np.log10(vmin) - 1)
        self.img_item.setImage(img_log, autoLevels=not self._levels_set)
        self._levels_set = True

    def _display_line(self, occ_l, ci):
        self.line_pw.clear()
        angles = ci.get('angles_2d')
        if angles is None or len(occ_l) == 0:
            return
        idx = np.argsort(angles)
        x   = angles[idx]
        y   = occ_l[idx]
        dx  = float(x[1] - x[0]) * 0.8 if len(x) > 1 else 1.0
        bar = pg.BarGraphItem(x=x, height=y, width=dx, brush='steelblue')
        self.line_pw.addItem(bar)

    def _show_class(self, c):
        if not self._class_info or c >= len(self._class_info):
            return
        ci   = self._class_info[c]
        mode = self._mode()

        if ci['dimensions'] == 2:
            self.plot_row.hide()
            self.line_pw.show()
            if mode == 'sampling' and ci['angles_2d'] is not None:
                self._display_line(np.ones(ci['L']), ci)
            elif self.current_images is not None and c < len(self.current_images):
                self._display_line(self.current_images[c], ci)
        else:
            self.line_pw.hide()
            self.plot_row.show()
            if mode == 'sampling':
                if ci['sampling_hist'] is not None:
                    self._display_image(ci['sampling_hist'], ci)
            elif self.current_images is not None and c < len(self.current_images):
                self._display_image(self.current_images[c], ci)

    # ------------------------------------------------------------------
    # Scale controls
    # ------------------------------------------------------------------

    def _on_scale_all_toggled(self, checked):
        self.scale_spin.setEnabled(not checked)
        self._on_scale_changed()

    def _on_scale_changed(self):
        self._rebuild_images()

    def _selected_k(self):
        if self.scale_all_cb.isChecked():
            return None
        return self.scale_spin.value()

    # ------------------------------------------------------------------
    # Update from iteration file
    # ------------------------------------------------------------------

    def _try_initial_update(self):
        if not self.fnam.is_file():
            return
        with h5py.File(self.fnam, 'r') as f:
            n = int(f.get('iterations', np.array(0))[()])
        if n > 0:
            self.update(f'iteration_{n - 1}')

    def update(self, iteration_key):
        if not self._class_info and not self._config_class_info:
            return
        if not self.fnam.is_file():
            return

        with h5py.File(self.fnam, 'r') as f:
            if iteration_key not in f:
                return
            grp = f[iteration_key]

            # Per-iteration geometry (preferred)
            if 'mapper' in grp:
                new_info = []
                mgrp = grp['mapper']
                for c_idx in range(len(mgrp)):
                    c_key = f'class_{c_idx}'
                    if c_key in mgrp:
                        new_info.append(self._class_info_from_h5_group(mgrp[c_key]))
                if new_info:
                    self._apply_class_info(new_info)
            elif not self._class_info:
                self._apply_class_info(self._config_class_info)

            self._has_hard = ('most_likely_orientation_d' in grp and
                              'most_likely_model_d'       in grp)
            self._has_soft = 'occupancy_r' in grp

            if self._has_hard:
                self._mlo_d_raw = grp['most_likely_orientation_d'][()]
                self._mlm_d_raw = grp['most_likely_model_d'][()]
            if self._has_soft:
                occ     = grp['occupancy_r'][()]
                total_R = sum(self._R_cs) if self._R_cs else 0
                if total_R > 0 and len(occ) != total_R:
                    self._has_soft = False   # config / iteration mismatch
                else:
                    self._occ_r_global = occ

        if not self._has_hard and not self._has_soft:
            return

        self._last_iter_key = iteration_key
        self.title_label.setText(iteration_key)
        self._rebuild_images()

    # ------------------------------------------------------------------
    # Image / line rebuilding
    # ------------------------------------------------------------------

    def _rebuild_images(self):
        if not self._class_info:
            return
        if not self._has_hard and not self._has_soft:
            return

        k = self._selected_k()
        images_hard = []
        images_soft = []

        for c, ci in enumerate(self._class_info):
            J, K, L = ci['J'], ci['K'], ci['L']
            k_c = min(k, K - 1) if k is not None else None

            if self._has_hard:
                mask  = self._mlm_d_raw == c
                r_idx = self._mlo_d_raw[mask]
                l_idx = r_idx % L
                if k_c is not None:
                    k_idx = (r_idx // L) % K
                    l_idx = l_idx[k_idx == k_c]
                occ_l = np.zeros(L, dtype=float)
                np.add.at(occ_l, l_idx, 1)
                if ci['dimensions'] == 3:
                    images_hard.append(self._occ_to_image(occ_l, ci))
                else:
                    images_hard.append(occ_l)

            if self._has_soft:
                r0      = ci['r_offset']
                R_c     = self._R_cs[c]
                occ_jkl = self._occ_r_global[r0 : r0 + R_c].reshape(J, K, L)
                occ_l   = (occ_jkl.sum(axis=(0, 1)) if k_c is None
                           else occ_jkl[:, k_c, :].sum(axis=0))
                if ci['dimensions'] == 3:
                    images_soft.append(self._occ_to_image(occ_l, ci))
                else:
                    images_soft.append(occ_l)

        self._images_hard = images_hard or None
        self._images_soft = images_soft or None

        mode = self._mode()
        if mode == 'most_likely':
            self.current_images = self._images_hard
        elif mode == 'occupancy':
            self.current_images = self._images_soft

        self._show_class(self.class_spin.value())

    def _on_mode_changed(self):
        mode = self._mode()
        if mode == 'most_likely':
            self.current_images = getattr(self, '_images_hard', None)
        elif mode == 'occupancy':
            self.current_images = getattr(self, '_images_soft', None)
        self._show_class(self.class_spin.value())
