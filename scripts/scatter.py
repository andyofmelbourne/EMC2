import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from collections import defaultdict

import signal
from pathlib import Path
import pyqtgraph.exporters

fnam       = '/home/andyofmelbourne/Documents/2024/p7927/scratch/2D-EMC/Ery_maxwell/iteration_info.h5'
#cxi_file   = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'
cxi_file = None

#fnam       = '/home/andyofmelbourne/Documents/2024/p7927/scratch/2D-EMC/Cube_maxwell/iteration_info.h5'
#cxi_file   = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Cube_all_hits.cxi'
labels_key = '/manual_selection'
#iteration = 484
iteration = 1

# load labels if any 
labels = None
if cxi_file and Path(cxi_file).is_file():
    # find cache
    a = Path(fnam).parent.joinpath('cachdir')
    stem = Path(cxi_file).stem
    b = list(a.glob(f'{stem}*sparse.h5'))
    
    data_file = None
    if len(b) == 1 :
        data_file = b[0]
    elif len(b) == 0 :  
        print(f'no data file found in cachdir {a} skipping labels')
    elif len(b) > 1 :  
        print('multiple files found in cachdir, skipping labels')
    
    print(f'{data_file=} {a=} {b=}')
    # load frame references
    if data_file :
        with h5py.File(data_file) as f:
            inds = f['/frame_index'][()]
    
        labels = {}
        with h5py.File(cxi_file) as f:
            g = f[labels_key]
            for k in g.keys():
                if 'crap' in k :
                    pen = (255, 0, 0, 100)
                elif 'sample' in k or 'good' in k:
                    pen = (0, 255, 0, 100)
                else :
                    pen = tuple(np.random.randint(0, 256, 3)) + (255,)
                labels[k] = (pen, np.where(g[k][()][inds])[0])
                print(f'loading label {k} with {len(labels[k][1])} indices')

# Get key mappings from Qt namespace
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)

# make unit-vectors
# -----------------
with h5py.File(fnam, 'r') as f:
    occ_dc = f[f'iteration_{iteration}/occupancy_dc'][()]

# now represent occ_dc on a 2D plane
theta = 2*np.pi*np.arange(occ_dc.shape[1]) / occ_dc.shape[1]
unit_vectors = np.zeros((occ_dc.shape[1], 2))
unit_vectors[:, 0] = np.cos(theta)
unit_vectors[:, 1] = np.sin(theta)

def get_plots(fnam, iteration):
    k = f'iteration_{iteration}/occupancy_dc'
    with h5py.File(fnam, 'r') as f:
        if k not in f:
            return None 
        
        occ_dc = f[k][()]
    
    occ_2d = np.dot(occ_dc, unit_vectors)
    return occ_2d



class GraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super(GraphicsLayoutWidget, self).__init__(*args, **kwargs)
        
        #self.scatter = self.addPlot(title=f'oocupancy per class per frame: iteration {iteration}')
        self.scatter = self.addPlot()
        #self.setWindowTitle(f'oocupancy per class per frame: iteration {iteration}')

        #self.exporter = pg.exporters.ImageExporter(self.scatter)
        self.exporter = pg.exporters.ImageExporter(self.scene())
        self.exporter.parameters()['width'] = 800
        self.exporter.parameters()['height'] = 800
        
        self.iteration = iteration
        
        # get data
        self.update_plots(iteration)
        
    
    def keyPressEvent(self, event):
        super(GraphicsLayoutWidget, self).keyPressEvent(event)
        key = keys_mapping[event.key()]
        #print("key press", key)
        
        if key == 'Right' :
            self.update_plots(self.iteration + 1)
        
        elif key == 'Left' :
            self.update_plots(self.iteration - 1)

        elif key == 'S' :
            self.exporter.export(f'scatter_{self.iteration:>04}.tif')
    
    def update_plots(self, iteration):
        # get plots
        occ_2d = get_plots(fnam, iteration)

        if occ_2d is None :
            return 
        
        self.iteration = iteration

        x = occ_2d[:, 0]
        y = occ_2d[:, 1]
        self.scatter.plot(x, y, pen=None, symbolPen=None, symbolSize=5, symbolBrush=(100, 100, 255, 50), clear = True)
        #self.scatter.plot(unit_vectors[:, 0], unit_vectors[:, 1], pen = None, symbolPen=pg.mkPen('w'), symbolSize=10, symbolBrush=None, hoverable=True, hoverBrush = pg.mkBrush(255, 255, 255, 100), data = np.arange(unit_vectors.shape[0]))
        sc = pg.ScatterPlotItem(
            unit_vectors[:, 0], 
            unit_vectors[:, 1], 
            pen=pg.mkPen('w'), 
            size=10, 
            brush=None,     
            hoverable=True, 
            hoverBrush = pg.mkBrush(255, 255, 255, 100), 
            data = np.arange(unit_vectors.shape[0])
        )
        self.scatter.addItem(sc)
        
        # show labels if any
        if labels : 
            self.scatter.addLegend()
            for key, (pen, label_inds) in labels.items():
                xl = x[label_inds]
                yl = y[label_inds]
                self.scatter.plot(xl, yl, pen = None, symbolPen=None, symbolSize=5, symbolBrush=pen, name = key)
        
        self.setWindowTitle(f'oocupancy per class per frame: iteration {iteration}')



pg.setConfigOption('background', 'k')
pg.setConfigOption('foreground', 'w')

app = pg.mkQApp("EMC scatter")
#mw = QtWidgets.QMainWindow()
#mw.resize(800,800)

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)
pg.setConfigOptions(imageAxisOrder='row-major')

win = GraphicsLayoutWidget(show=True, title="Per frame per class occupancy")
win.resize(600,600)
#win.setWindowTitle('pyqtgraph example: Plotting')

win.show()

timer = QtCore.QTimer()
timer.timeout.connect(lambda : win.update_plots(win.iteration + 1))
timer.start(1000)
#QtCore.QApplication.exec_()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    pg.exec()

