import numpy as np
import h5py
import pyqtgraph as pg
from tqdm import tqdm

from context import emc2
from emc2 import utils 

# (no. of pixels >= 20) >= 10

fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'
dset = '/entry_1/instrument_1/detector_1/saturated'
    
with h5py.File(fnam) as f:
    data = f['entry_1/data_1/data']
    sat  = np.zeros(data.shape[0], dtype = bool)
    
    for d in tqdm(range(data.shape[0])):
        sat[d] = np.sum(data[d]>=20) >= 10

    
print(f'writing {np.sum(sat)} saturated frame labels to {fnam}/{dset}')
with h5py.File(fnam, 'r+') as f:
    if dset in f : 
        if f[dset].shape == sat.shape :
            f[dset][:] = sat
        else :
            del f[dset]

    f[dset] = sat

