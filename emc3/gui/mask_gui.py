"""
show h5 / cxi files in directory
auto locate geometries for each file
when clicked a dataset will open with a given geometry
only datasets for which that geometry applies can be loaded from then on

Load the mask maker in the plotWidget
"""

import h5py
import numpy as np

from .cxi_gui import CXI_viewer
from .plot_widget import PlotWidget

import pyqtgraph as pg

from PyQt5.QtWidgets import (
        QApplication, QAction, QDialog, QPushButton, QVBoxLayout, QButtonGroup,
        QCheckBox, QHBoxLayout, QLabel, QLineEdit
        )

from PyQt5.QtCore import Qt


class MaskMakerOptions(QDialog):
    def __init__(self, parent=None, plot_item=None):
        super().__init__(parent, flags=Qt.Window)
        self.parent = parent
        self.setWindowTitle("Mask Maker options")
        layout = QVBoxLayout(self)

        # toggle / mask / unmask checkboxes
        self.toggle_checkbox = QCheckBox('toggle')
        self.mask_checkbox = QCheckBox('mask')
        self.unmask_checkbox = QCheckBox('unmask')
        self.mask_checkbox.setChecked(True)

        toggle_group = QButtonGroup(self)
        toggle_group.addButton(self.toggle_checkbox)
        toggle_group.addButton(self.mask_checkbox)
        toggle_group.addButton(self.unmask_checkbox)
        toggle_group.setExclusive(True)

        hbox = QHBoxLayout()
        hbox.addWidget(self.toggle_checkbox)
        hbox.addWidget(self.mask_checkbox)
        hbox.addWidget(self.unmask_checkbox)
        layout.addLayout(hbox)

        # rectangular ROI selection
        self.roi = pg.RectROI([-200,-200], [100, 100])
        # make sure ROI is drawn above image
        self.roi.setZValue(10)

        self.ROI_button = QPushButton('mask rectangular ROI')
        layout.addWidget(self.ROI_button)
        # self.ROI_button.clicked.connect(lambda : self.mask_ROI(self.roi))

        # histogram mask button
        hbox = QHBoxLayout()

        label = QLabel("mask histogram:")
        self.below_hist_button = QPushButton('below')
        self.above_hist_button = QPushButton('above')
        self.both_hist_button = QPushButton('both')

        hbox.addWidget(label)
        hbox.addWidget(self.below_hist_button)
        hbox.addWidget(self.above_hist_button)
        hbox.addWidget(self.both_hist_button)
        layout.addLayout(hbox)
        # hist_button.clicked.connect(self.mask_hist)

        # load current mask
        self.load_current_button = QPushButton("Load current mask")
        layout.addWidget(self.load_current_button)

        # save button
        save_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_field = QLineEdit()
        self.save_field.setText("mask.h5/data")  # set default value

        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.save_field)
        layout.addLayout(save_layout)

    def closeEvent(self, event):
        self.parent.options = None
        super().closeEvent(event)


class MaskMaker_plot_widget():
    """
    remember mask maker with each dataset click

    could be more modular, I think all I need is:
        - pyqtgraph viewbox
        - frame_shape
        - image_shape
        - data_to_display_coords_ravel
        - display_to_data_coords_ravel
    """
    def __init__(self, plot_widget):
        # plot widget doesn't change
        self.plot_widget = plot_widget

        self.mask = None
        self.options = None

    def update_filter(self, filter):
        self.plot_widget.filter = filter

    def update_data(self, data, name=None):
        # this also applies geom filter
        print(f'{self.plot_widget.filter=}')
        self.plot_widget.plot(data, name=name)

        data = self.plot_widget.current_plot.data

        self.check_data(data)

        # if this is the first call then set mask
        if self.mask is None:
            self.mask = np.ones(self.frame_shape, dtype=bool)
            self.mask_image = np.ones(self.image_shape, dtype=bool)

        # update new plot item and data
        self.plot_item = self.plot_widget.current_plot.plot_item
        self.hist = self.plot_widget.current_plot.hist
        self.data = data
        self.img_item = self.plot_widget.current_plot.img_item

        # add mask maker stuff
        self.add_menu_item()
        self.init()

    def check_data(self, data):
        c = [
                hasattr(data, 'frame_shape'),
                hasattr(data, 'frame_data'),
                hasattr(data, 'image_shape'),
                hasattr(data, 'data_to_display_coords_ravel'),
                hasattr(data, 'display_to_data_coords_ravel')
            ]

        if all(c):
            self.image_shape = data.image_shape
            self.frame_shape = data.frame_shape
            self.i_n = data.display_to_data_coords_ravel
            self.n_i = data.data_to_display_coords_ravel

        else:
            print(f'{c=}')
            e = 'data does not have required attributes for mask making'
            raise ValueError(e)

    def add_menu_item(self):
        menu = self.plot_item.menu
        make_mask = QAction("make mask", menu)
        make_mask.triggered.connect(lambda x: self.init())
        menu.addAction(make_mask)

    def init(self):
        self.add_options_menu()
        self.add_overlay()
        self.add_mouse_clicked()

        # im = self.i_n.reshape(self.image_shape)
        # super().plot(im, ())

    def add_options_menu(self):
        # prevent multiple popups
        if self.options is None:
            # add menu items to pop up window
            self.options = MaskMakerOptions(self.plot_widget, self.plot_item)

            # Connect popup buttons to main window functions
            self.options.ROI_button.clicked.connect(
                    lambda : self.mask_ROI(self.options.roi)
                    )

            self.options.below_hist_button.clicked.connect(
                    lambda : self.mask_hist('below')
                    )

            self.options.above_hist_button.clicked.connect(
                    lambda : self.mask_hist('above')
                    )

            self.options.both_hist_button.clicked.connect(
                    lambda : self.mask_hist('both')
                    )

            self.options.save_button.clicked.connect(
                    lambda : self.save(self.options.save_field.text())
                    )

            self.options.load_current_button.clicked.connect(self.load)

            self.options.show()

        self.plot_item.getViewBox().addItem(self.options.roi)

    def load(self):
        # I think this makes more sense
        m = ~self.data.frame_data.astype(bool)
        print(f'{np.sum(m)=} {m.dtype=}')
        self.mask[m] = self.mask_toggle(self.mask[m])
        self.update_overlay()

    def save(self, fnam):
        """
        fnam = root/filename.h5/data/dataset
        """
        # get extension ('.h5')
        extension = '.' + fnam.split('.')[1].split('/')[0]

        # get file name ('root/filename.h5')
        filename = fnam.split(extension)[0]
        filename = f'{filename}{extension}'

        # get dataset ('/data/dataset')
        dataset = fnam.split(extension)[1]

        print('saving mask (True is transperant False is blue) to:')
        print(f'to {filename} in {dataset=}')

        with h5py.File(filename, 'a') as f:
            if dataset in f:
                if (f[dataset].shape != self.mask.shape
                    or f[dataset].dtype != self.mask.dtype):
                    del f[dataset]
                    f[dataset] = self.mask
                else:
                    f[dataset][:] = self.mask
            else:
                f[dataset] = self.mask

    def mask_ROI(self, roi):
        sides   = [roi.size()[1], roi.size()[0]]

        n_ss_min, n_fs_min = roi.pos()
        n_ss_max, n_fs_max = n_ss_min + sides[1], n_fs_min + sides[0]

        n_ss = self.n_i // self.image_shape[1]
        n_fs = self.n_i % self.image_shape[1]
        m = (n_ss > n_ss_min) * (n_ss < n_ss_max) * \
            (n_fs > n_fs_min) * (n_fs < n_fs_max)

        self.mask.ravel()[m] = self.mask_toggle(self.mask.ravel()[m])
        self.update_overlay()

    def mask_hist(self, where='both'):
        min_max = self.hist.getLevels()

        if where == 'below':
            m = self.data.frame_data < min_max[0]
        elif where == 'above':
            m = self.data.frame_data > min_max[1]
        elif where == 'both':
            m = (self.data.frame_data < min_max[0]) * (self.data.frame_data > min_max[1])

        self.mask[m] = self.mask_toggle(self.mask[m])
        self.update_overlay()

    def mask_toggle(self, mask):
        if self.options.toggle_checkbox.isChecked():
            return ~mask
        elif self.options.mask_checkbox.isChecked():
            return False * mask
        elif self.options.unmask_checkbox.isChecked():
            return True + mask

    def add_overlay(self):
        self.mask_rgba = np.empty(self.image_shape + (4,), dtype=np.uint8)
        self.mask_img_item = pg.ImageItem(self.mask_rgba)
        self.plot_item.getViewBox().addItem(self.mask_img_item)
        self.update_overlay()

    def update_overlay(self):
        self.mask_image.ravel()[self.n_i] = self.mask.ravel()
        self.mask_rgba[~self.mask_image] = [0, 0, 255, 255]
        self.mask_rgba[self.mask_image] = [0, 0, 0, 0]
        self.mask_img_item.setImage(self.mask_rgba)

    def add_mouse_clicked(self):
        # Use SignalProxy to handle scene mouse clicks
        img = self.img_item

        img.setAcceptedMouseButtons(Qt.AllButtons)
        # Enable mouse tracking (not sure why we need this)
        # proxy = pg.SignalProxy(img.scene().sigMouseClicked, rateLimit=60, slot=None)

        def mouse_clicked(event):
            if event.button() == Qt.LeftButton:
                pos = event.scenePos()
                mouse_point = img.mapFromScene(pos)

                ss, fs = int(mouse_point.x()), int(mouse_point.y())

                if ( 0 <= fs < self.mask_image.shape[1] and
                    0 <= ss < self.mask_image.shape[0]):
                    i = self.i_n[self.mask_image.shape[1] * ss + fs]
                    if i > 0:
                        self.mask.ravel()[i] = self.mask_toggle(self.mask.ravel()[i])
                        self.update_overlay()
                event.accept()

        img.mouseClickEvent = mouse_clicked
        # proxy = pg.SignalProxy(img.scene().sigMouseClicked, rateLimit=60, slot=mouse_clicked)


class Mask_maker(CXI_viewer):
    def __init__(self, directory, parent=None):
        super().__init__(directory, parent=parent)

        # modify plot_widget
        self.mask_widget = MaskMaker_plot_widget(self.plot_widget)

        self.filter = None

    def on_item_path_clicked(self, path_list):
        """
        set filter and launch mask plot widget
        do not change later (user can close and reopen for that)
        """
        # path = "/" + "/".join(path_list)
        data_dict = self.h5_list_widget.path_to_file_dataset(path_list)

        if data_dict is None:
            return

        name = data_dict['dataset']
        fnam = data_dict['fnam']
        data = h5py.File(fnam)[name]

        modifiers = QApplication.keyboardModifiers()

        if self.filter is None:
            if fnam in self.filters:
                print(f'setting geometry filter for {fnam}')
                self.filter = self.filters[fnam]
            else:
                self.filter = lambda x: x
        else:
            # check if filter can be applied to new data
            if self.filter.check(data):
                # update filter data if needed
                print('clicked data is consistent with current filter')
            else:
                print('clicked data is not consistent with current filter')
                return

        # data = self.filter(data)
        print(f'{data=} {name=}')
        self.mask_widget.update_filter(self.filter)
        self.mask_widget.update_data(data, name=name)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C

    pg.setConfigOption('background', pg.mkColor(0.1))

    app = QApplication(sys.argv)

    # fnam = '/home/andyofmelbourne/Documents/2025/LCLS-CXI-1008449/data/r0087_radial_profiles.h5'
    if len(sys.argv) > 1:
        fnam = sys.argv[1]
    else:
        # fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits_no_mask.cxi'
        # fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'
        fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/'

    fnams = [str(f) for f in Path(fnam).glob('*.cxi')]

    geom_filter = {}
    for f in fnams:
        geom_filter[f] = get_geometry_filter(f)

    window = H5_viewer(fnam, filters=geom_filter)
    window.setWindowTitle("CXI viewer")
    window.resize(800, 400)
    window.show()

    sys.exit(app.exec_())
