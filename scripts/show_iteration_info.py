import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import argparse
import math
import os
from pathlib import Path
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
    parser.add_argument('--cxi', type=str, help='cxi file for frame viewing')
    args = parser.parse_args()

    # get sparse data
    directory = Path(args.fnam).parent
    fnam_sparse = directory.joinpath('/cachdir/')
    fnam_sparse = fnam_sparse.glob('*sparse.h5')
    for fnam in fnam_sparse:
        args.fnam_sparse = fnam.resolve()
        print(f'found sparse data: {args.fnam_sparse}')
        break
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
            return None, None, None, None
        
        Is = f[k][()]
        
        image, info = utils.make_models_2D_image(Is)
        return image, info['positions'], info['classes'], info['N']

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

        print('press "f" to display class images of last selection')
        print('press "s" to save selection to cxi file and good_classes.pickle')
        
        self.last_selected = None
        self.selection     = None
        self.N             = None
        self.classes       = None
        self.positions     = None
        self.pos           = None
        self.scatter_hover = None
        
        self.update_plots(iteration)
        
        
    def keyPressEvent(self, event):
        super(ImageView, self).keyPressEvent(event)
        key = keys_mapping[event.key()]
        #print("key press", key)
        
        if key == 'Right' :
            self.update_plots(self.iteration + 1)
        
        elif key == 'Left' :
            self.update_plots(self.iteration - 1)

    def init_scatter(self):
        # set hover: fill with grey
        self.scatter_hover = pg.ScatterPlotItem(
            size=self.N, 
            pen=None, 
            brush=None, 
            symbol='o', 
            pxMode=False, 
            hoverBrush = pg.mkBrush(255, 255, 255, 100), 
            hoverable=True
        )
        self.addItem(self.scatter_hover)
        self.scatter_hover.sigClicked.connect(self.clicked)
        
        self.selection = np.zeros(len(self.classes), dtype = bool) 
        self.update_selection()
    
    def clicked(self, points, ev):
        for p in ev:
            c = p.data()
            self.selection[c]  = ~self.selection[c]
            if self.selection[c] : self.last_selected = c
            self.update_selection()

    def update_selection(self):
        spots = []
        for c in range(len(self.classes)):
            pen  = pg.mkPen('g') if self.selection[c] else None
            spot = {'pos': self.positions[c], 'pen': pen, 'data': self.classes[c]}
            spots.append(spot)
        
        self.scatter_hover.setData(spots)

    def update_scatter(self):
        if self.scatter_hover is None :
            self.init_scatter()
            
    def update_plots(self, iteration):
        im, pos, classes, N = get_slices(fnam, iteration)
        
        if im is None :
            return None
            
        self.iteration = iteration
        im[im == 0] = np.nan
        plot = self.getView()
        plot.setTitle(f'iteration: {iteration}')
        self.setImage(im**0.2, autoRange = False, autoLevels = False, autoHistogramRange = False)
        #self.setImage(im, autoRange = False, autoLevels = False, autoHistogramRange = False)
        
        self.positions = pos
        self.classes   = classes
        self.N         = N
        self.update_scatter()

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
