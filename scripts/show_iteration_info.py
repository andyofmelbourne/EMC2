import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import argparse
import math
import os
import h5py
from PyQt5 import QtGui, QtCore, QtWidgets
from collections import defaultdict
import signal

from context import emc2
from emc2 import utils 

def get_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description="""
    Show iteration information
    """)
    parser.add_argument('fnam', type=str, help='iteration_info file name')
    args = parser.parse_args()
    return args

# Get key mappings from Qt namespace
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)


# load config
args = get_args()
fnam      = args.fnam

#with h5py.File(fnam, 'r') as f:
#    iteration = f['iterations'][()]
iteration = 1

def get_slices(fnam, iteration):
    with h5py.File(fnam, 'r') as f:
        k = f'iteration_{iteration}/model_slices'
        if k not in f :
            return None
        
        Is = f[k][()]
        
        return utils.make_models_2D_image(Is)
            

def get_plots(fnam, iteration):
    # get plots
    plots  = []
    titles = []
    with h5py.File(fnam, 'r') as f:
        k = f'iteration_{iteration}'
        if k not in f :
            return None, None, None, None

        plots.append(f['Q'][()])
        titles.append('P logR')

        plots.append(f['beta'][()])
        titles.append('beta scaling')

        plots.append(f['P_gini'][()])
        titles.append('gini coefficient of P')
        
        g = f[k]
        plots.append(np.sort(g['P_gini_d'][()])[::-1])
        titles.append('P gini coefficient per frame')

        plots.append(np.sort(g['Q_d'][()])[::-1])
        titles.append('P logR per frame')

        #plots.append(np.sort(np.bincount(g['most_likely_model_d'][()])[::-1]))
        plots.append(np.bincount(g['most_likely_model_d'][()]))
        titles.append('model occupancy')

        R = g['occupancy_r'].shape[0]
        D = g['Q_d'].shape[0]

    return plots, titles, R, D

class GraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super(GraphicsLayoutWidget, self).__init__(*args, **kwargs)
        
        # get plots
        plots, titles, R, D = get_plots(fnam, iteration)
        
        pg_plots = []
        while True :
            for j in range(3):
                index = len(pg_plots)
                print(index, titles[index])
                pg_plots.append(self.addPlot(title=titles[index]))
                pg_plots[-1].plot(plots[index])
                if plots[index].shape[0] == R :
                    pg_plots[-1].setLabel('bottom', 'orientation')
                elif plots[index].shape[0] == D :
                    pg_plots[-1].setLabel('bottom', 'frame')
                
                if len(pg_plots) == len(titles) : 
                    break
           
            if len(pg_plots) == len(titles) : 
                break
                
            self.nextRow()

        self.pg_plots = pg_plots
        self.iteration = iteration
    
    def keyPressEvent(self, event):
        super(GraphicsLayoutWidget, self).keyPressEvent(event)
        key = keys_mapping[event.key()]
        #print("key press", key)
        
        if key == 'Right' :
            self.update_plots(self.iteration + 1)
        
        elif key == 'Left' :
            self.update_plots(self.iteration - 1)
    
    def update_plots(self, iteration):
        # get plots
        plots, self.titles, R, D = get_plots(fnam, iteration)
        
        if plots is None :
            return 
        
        self.iteration = iteration
        
        for plot, pg_plot in zip(plots, self.pg_plots) :
            pg_plot.plot(plot, clear = True)

        self.setWindowTitle(f'EMC summary: iteration {iteration}')

class ImageView(pg.ImageView):
    def __init__(self, iteration, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)
        self.update_plots(iteration)
        
    def keyPressEvent(self, event):
        super(ImageView, self).keyPressEvent(event)
        key = keys_mapping[event.key()]
        #print("key press", key)
        
        if key == 'Right' :
            self.update_plots(self.iteration + 1)
        
        elif key == 'Left' :
            self.update_plots(self.iteration - 1)
        
    def update_plots(self, iteration):
        im = get_slices(fnam, iteration)
        if im is None :
            return None
        self.iteration = iteration
        im[im == 0] = np.nan
        plot = self.getView()
        plot.setTitle(f'iteration: {iteration}')
        self.setImage(im**0.2, autoRange = False, autoLevels = False, autoHistogramRange = False)

app = pg.mkQApp()

# Enable antialiasing for prettier plots
pg.setConfigOption('background', pg.mkColor(0.1))
pg.setConfigOption('foreground', 'w')
pg.setConfigOptions(antialias=True)

win = GraphicsLayoutWidget(show=True)
win.resize(1000,600)
win.setWindowTitle(f'EMC summary: iteration {iteration}')


plot = ImageView(iteration = iteration, view = pg.PlotItem())
plot.show()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    pg.exec()
