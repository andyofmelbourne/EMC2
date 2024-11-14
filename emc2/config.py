# data
# ----

# frame selection
cxi_file = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits.cxi'

# only process frames with more than 1000 counts
def filter(f):
    selection = f['entry_1/instrument_1/detector_1/photon_counts'][()] > 2000
    return selection

# split frames w more than 2x the minimum 
# number of photons
split_frames     = 2000
q_max            = None
res_max          = None
pixel_radius     = None
polarisation     = 'x' 

# model
# -----
pixels_per_voxel      = 4
model_length          = 32
symmetry              = 'inversion'
dimensions            = 2
models                = 8
interpolation_forward = 'linear'
interpolation_inverse = 'nearest'

# tomograms
# ---------
rotation_order   = 8

# likelihood 
# ----------
likelihood = 'Poisson_fluence_free'

# Frame model
# -----------
# basic: F_dri = C_i W_ri
frame_model = 'basic'

# Maximise
# --------
# maximise I rather than W's
maximise = 'I'
# skip frames with P_dr < P_thresh P_max_d 
P_thresh = 0.01 

# Iterations
# ----------
iterations    = 100
beta_strategy = 'exponential'
#beta_strategy = 'linear'
beta_start    = 0.001
beta_stop     = 0.1
gpu           = True
max_mem       = 4 # GB's

# parallelisation
# ---------------
nproc    = 1
