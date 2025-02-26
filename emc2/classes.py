"""
class_0.h5
    model
    relative_fluence
    dq
    rotation_order
    maximise

    update_fluence
    update_model

    probability_matrix
    beta
    P_thresh
    likelihood
    frame_model

    cxi_file
    polarisation
    split_frames

    mapping (S, R)
    symmetry
    interpolation_forward
    xyz_offset
"""

import h5py
import logging

logger = logging.getLogger(__name__)

default = {
    'model': None,
    'relative_fluence': None,
    'dq': None,
    'rotation_order': None,
    'maximise': None,
    'update_fluence': None,
    'update_model': None,
    'probability_matrix': None,
    'beta': None,
    'P_thresh': 0.,
    'likelihood': None,
    'frame_model': None,
    'cxi_file': None,
    'polarisation': None,
    'split_frames': None,
    'mapping_matrix': None,
    'symmetry': None,
    'interpolation_forward': None,
    'xyz_offset': None,
    'xyz': None,
    'pixels_per_voxel': None,
    'frame_selection': None,
    'mask': None,
    'P_mask': None,
    'P_C': None,
    'P_xyz': None,
    'C': None,
    'wsums': None,
    'ksums': None,
    'orientation_index_r': None,
    'x_offset_r': None,
    'y_offset_r': None,
    'z_offset_r': None,
    'class_id': None
}


class Class():
    def __init__(self, **kwargs):
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def check(self):
        for key in default.keys():
            if not hasattr(self, key):
                raise ValueError(f'{key} has not been set!')

    def save(self, fnam, overwrite=True):
        self.check()

        if overwrite:
            logger.info(f'saving class info to {fnam}')
            with h5py.File(fnam, 'w') as f:
                for key in default.keys():
                    v = getattr(self, key)
                    logger.debug(f'saving {key} {type(v)}')
                    f[key] = v
        else:
            logger.info(f'updating class info in {fnam}')
            with h5py.File(fnam, 'r+') as f:
                for key in default.keys():
                    f[key][...] = getattr(self, key)

    def load(self, fnam, skip=[]):
        with h5py.File(fnam, 'r') as f:
            for key in default.keys():
                if key in skip:
                    continue

                v = f[key][()]
                if type(v) is bytes:
                    v = v.decode()
                setattr(self, key, v)
