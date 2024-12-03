import h5py
import numpy as np
import signal
from pathlib import Path
from collections import OrderedDict

fnam       = '/home/andyofmelbourne/Documents/2024/p7927/scratch/2D-EMC/Ery2/iteration_info.h5'
cxi_file   = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'
labels_key = '/manual_selection'
iteration = 1

# load labels if any 
labels = None
if cxi_file and Path(cxi_file).is_file():
    # find cache
    a = Path(fnam).parent.joinpath('cachdir')
    b = list(a.glob('*sparse.h5'))
    
    data_file = None
    if len(b) == 1 :
        data_file = b[0]
    elif len(b) == 0 :  
        print(f'no data file found in cachdir {a} skipping labels')
    elif len(b) > 1 :  
        print('multiple files found in cachdir, skipping labels')
    
    # load frame references
    if data_file :
        with h5py.File(data_file) as f:
            inds = f['/frame_index'][()]
    
        labels = OrderedDict()
        with h5py.File(cxi_file) as f:
            g = f[labels_key]
            for k in g.keys():
                labels[k] = np.where(g[k][()][inds])[0]
                print(f'loading label {k} with {len(labels[k])} indices')

# calculate occupancy per class per label per iteration
with h5py.File(fnam, 'r') as f:
    k = f'iteration_1/most_likely_model_d'
    
    classes = 1+np.max(f[k][()])

    iterations = f['iterations'][()]

occ_ilc = np.zeros((iterations, len(labels.keys()), classes), dtype = int)

with h5py.File(fnam, 'r') as f:
    for i in range(iterations):
        k = f'iteration_{i+1}/most_likely_model_d'
        
        if k in f :
            ml_d = f[k][()]
        
            for l, key in enumerate(labels):
                occ_ilc[i, l] = np.bincount(ml_d[labels[key]], minlength = classes)
        else :
            print('warning could not find {k}') 
        

print(occ_ilc[-1])

print(f'{occ_ilc.shape=}')

a = occ_ilc[:, 0, :]
b = occ_ilc[:, 1, :]
DOS = 1 - np.sum(a * b, axis=-1) / (np.sum(a**2, axis=-1) * np.sum(b**2, axis=-1))**0.5
for i in range(iterations):
    print(i, np.round(100 * DOS[i], 2), '%')


from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_tight_layout(True)
fig.set_size_inches(10, 5)


ax.bar(np.arange(DOS.shape[0]), DOS,  width = 1, align='edge', color='lightcoral', edgecolor='k', alpha=0.8, linewidth=1, zorder=3)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlim([0, DOS.shape[0]])
ax.set_title("Degree Of Separation between good and bad classes")
ax.set_ylabel("DOS")
ax.set_xlabel("iteration")
ax.set_yticks(np.arange(0, 1., 0.1))
plt.grid(axis='y', zorder=0)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))

plt.show()
