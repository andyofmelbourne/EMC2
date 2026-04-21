import pickle
import numpy as np
import h5py

from .plot_widget import keys_mapping, PlotWidget
from .. import utils_cl
from ..tomograms import Tomograms, Tomograms_cl



class Show_fit_widget():
    def __init__(self, config_file, most_likely_model_d, most_likely_orientation_d):
        self.config = pickle.load(open(config_file, 'rb'))
        self.most_likely_orientation_d = most_likely_orientation_d
        self.most_likely_model_d = most_likely_model_d
        self.models = []
        self.tomos = {}
        self.w_d = None
        self.cl = None
        self.frame = None
        self.mask = None

    def get_tomo(self, d):
        """
        d is a global frame index (from cxi file)
        """
        d_local = np.where(self.config['classes'][0]['data'].frames == d)[0]
        if len(d_local)==1:
            d_local = d_local[0]
        else:
            print(f'frame {d} is not included in config')

        # get most likely model
        ci = self.most_likely_model_d[d_local]

        # get most likely orientation
        r = self.most_likely_orientation_d[d_local]

        print(f'fitting frame {d} to class {ci} for orientation {r}')

        # init stuff (if not done so already)
        self.init_frames(ci)

        # calculate tomogram
        W_ri = self.tomos[ci].calculate_tomogram(r, r+1, log=False, cpu=True)[0]

        # calculate frame
        c = self.config['classes'][ci]

        if hasattr(c['data'], 'B_di'):
            B_di = c['data'].B_di[d_local][0]
        else:
            B_di = 0

        F_dri = self.w_d[d_local] * W_ri + B_di

        # fill frame data
        #self.frame[self.mask] = F_dri**0.2
        self.frame[self.mask] = F_dri
        return self.frame

    def init_frames(self, ci):
        print(f'{ci=}')
        if ci in self.tomos:
            return

        c = self.config['classes'][ci]

        # load model
        if ci not in self.models:
            with h5py.File(c['model_file']) as f:
                c['model'].data = f['data'][()]

            self.models.append(ci)

        # load background
        # not needed if there is no background...
        self.B_di = None
        if hasattr(c['data'], 'B_di'):
            c['data'].B_di.load_data()

        # load fluence
        if self.w_d is None:
            with h5py.File(c['fluence_file']) as f:
                w_d = f['w_d'][()]
            self.w_d = w_d

        if self.cl is None:
            self.cl = utils_cl.opencl_init_cpu()

        c['mapper'].load_coords(c['data'].mask)

        tomos = Tomograms(
                c['mapper'],
                c['model'],
                c['data'].C_i,
                self.w_d)

        # calculate wsums_r
        tomos_cl = Tomograms_cl(tomos, self.cl['context'], self.cl['queue'])
        tomos_cl.load_buffers(1)

        self.tomos[ci] = tomos_cl

        if self.frame is None:
            self.mask = c['data'].mask
            self.frame = np.zeros(self.mask.shape, dtype=float)

def plugin(Widget, config_file, model_d, r_d):
    show_fit_widget = Show_fit_widget(config_file, model_d, r_d)

    # make subclass
    class Plot_fit_widget(Widget):
        fit_widget = show_fit_widget

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.is_showing_fit = False
            self.show_fit_slice = False

        def keyPressEvent(self, event):
            super().keyPressEvent(event)

            key = keys_mapping[event.key()]

            if key == 'T':
                if self.show_fit_slice != self.current_slice:
                    self.is_showing_fit = False

                if self.is_showing_fit:
                    self.is_showing_fit = False
                    self.plot()
                else:
                    self.is_showing_fit = True
                    self.show_fit_slice = self.current_slice

                    frame = self.fit_widget.get_tomo(self.current_slice)
                    im = self.data.apply(frame)
                    self.plot(data=im)

    return Plot_fit_widget


if __name__ == '__main__':
    import signal, sys
    import pyqtgraph as pg
    from .getters import GeomGetterCXI_h5
    from PyQt5.QtWidgets import (
            QApplication
            )

    # get a plot widget
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    pg.setConfigOption('background', pg.mkColor(0.1))
    pg.setConfigOptions(antialias=True)
    pg.setConfigOptions(imageAxisOrder='row-major')

    win = PlotWidget()

    wd = '/home/andyofmelbourne/Documents/2024/p7927/scratch/Ery/recon_2D_04'
    fnam = f'{wd}/Ery_all_hits_no_mask.cxi'
    name = 'entry_1/data_1/data'
    data = GeomGetterCXI_h5(fnam, name)

    fnam_iter = f'{wd}/iteration_info.h5'
    with h5py.File(fnam_iter, 'r') as f:
        model_d = f['/iteration_133/most_likely_model_d'][()]
        r_d = f['/iteration_133/most_likely_orientation_d'][()]
        frames = f['/iteration_132/frames'][()]

    config_file = 'config.pickle'
    p = lambda x: plugin(x, config_file, model_d, r_d)
    win.plot(data, name, plugin=p)
    win.current_plot.update_xmap(frames)

    # add cxi_viewer by default
    win.resize(800, 500)
    win.show()
    sys.exit(app.exec_())


