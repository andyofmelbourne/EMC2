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
interpolation_forward = 'linear'
interpolation_inverse = 'nearest'
symmetry              = 'inversion'
models                = 4
dimensions            = [3, 2, 2, 2]

# tomograms
# ---------
# (grid lenth, step size)
#pointing_fluctuations = (4, 400e-6)
pointing_fluctuations = False
rotation_order        = [4, 8, 8, 0]

# likelihood 
# ----------
#likelihood = 'Poisson'
likelihood = 'Poisson_fluence_free'

# Frame model
# -----------
# basic: F_dri = C_i W_ri
frame_model = 'basic'
# fluence: F_dri = w_d C_i W_ri
#frame_model = 'fluence'

# Maximise
# --------
# maximise I rather than W's
maximise = 'I'
# skip frames with P_dr < P_thresh P_max_d 
P_thresh = 0.01 
update_fluence = True

# Iterations
# ----------
iterations    = 100
beta_strategy = 'exponential'
#beta_strategy = 'linear'
beta_start    = 0.001
beta_stop     = 0.1
gpu           = True
max_mem       = 4 # GB's

