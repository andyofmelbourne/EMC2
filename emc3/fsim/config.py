import numpy as np

cxi_file = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/Ery_all_hits_no_mask.cxi'
mask_file = '/home/andyofmelbourne/Documents/2024/p7927/scratch/Ery/recon_3D_10/mask.h5'
mask_dset = 'data'
output_file = 'temp_Ery/test.cxi'

focal_spot_size = 200e-9
nhits = 8
dtype = np.uint8
#pulse_energy_factor = 1e-2 # modulate value in cxi file
pulse_energy_factor = 1e5 # modulate value in cxi file
#dtype = np.float32
minimum_fluence = 1e11 / 1e-12 # 1e13 photons / um^2 (SPB max/2)
#minimum_fluence = 1e8 / 1e-12 # 1e13 photons / um^2
photon_counting = True
detector_distance = 650e-3

max_pix_rad = False

background = {
    'type': 'constant',
    'mean': 0, # per frame
}

"""
sample = {
    'type': 'truncated_polyhedron',
    'name': 'truncated_polyhedron',
    'size': 30e-9,
    'truncation': 0.1,
    'formula': 'Au'
}
"""

sample = {
    'type': 'pdb',
    'pdb': '2gtl',
}
