import numpy as np
import h5py

fnam = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'

with h5py.File(fnam, 'r') as f:
    B = f['/entry_1/instrument_1/detector_1/background'][()]
    binds = f['/entry_1/background_index'][()]
    bw    = f['/entry_1/background_weighting'][()]

B = np.sum(B.reshape((B.shape[0], -1)), axis = 1)

# estimated integrated background counts per frame
b = bw * B[binds]

with h5py.File(fnam, 'r+') as f:
    k = '/entry_1/background_counts'
    if k in f :
        if f[k].shape == b.shape :
            f[k] = b
        else :
            del f[k]
    
    if k not in f :
        f.create_dataset(
            k, 
            data = b,
            chunks = b.shape,
            compression = 'gzip',
            compression_opts = 1
        )
