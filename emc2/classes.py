"""
Figuring out when to update things based on config

change to model dq or model_length or symmetry

update_I script:
    Keep:
        - P_dr
    Update:
        - mapping matrix
        - model (empty or zeros)
        - tomogram sums
        - data cache
        - everything else

calculate_logR script:
    Change nothing
    We only want an update to the model to occur when the model is calculated


If we keep the model dq and size then we can update everything except the model
values

"""

import h5py
import logging
import sys
import time

from emc2 import utils

logger = logging.getLogger(__name__)

default = {
    'cxi_file': None,
    'split_frames': None,
    'pixels_per_voxel': None,
    'dimensions': None,
    'model_length': None,
    'model': None,
    'P_mask_padding': None,
    'symmetry': None,
    'interpolation_forward': None,
    'xyz_offset': None,
    'scale': None,
    'zero_padding': None,
    'relative_fluence': None,
    'dq': None,
    'rotation_order': None,
    'likelihood': None,
    'frame_model': None,
    'maximise': None,
    'update_fluence': None,
    'update_logR': None,
    'update_probability': None,
    'update_model': None,
    'polarisation': None,
    'probability_matrix': None,
    'beta': None,
    'P_thresh': None,
    'mapping_matrix': None,
    'xyz': None,
    'frame_selection': None,
    'mask': None,
    'P_mask': None,
    'P_C': None,
    'P_xyz': None,
    'C': None,
    'wsums': None,
    'P_wsums': None,
    'ksums': None,
    'orientation_index_r': None,
    'scale_r': None,
    'x_offset_r': None,
    'y_offset_r': None,
    'z_offset_r': None,
    'wavelength': None,
    'class_id': None
}


class Class(dict):
    def __init__(self, **kwargs):
        for key in default.keys():
            if key in kwargs:
                self.set(key, kwargs[key])

    def set(self, key, value):
        setattr(self, key, value)
        self[key] = getattr(self, key)

    def check(self):
        for key in default.keys():
            if not hasattr(self, key):
                raise ValueError(f'{key} has not been set!')

    def save(self, fnam, check=True, overwrite=True, skip=[]):
        if check:
            self.check()

        if overwrite:
            logger.info(f'saving class info to {fnam}')
            with h5py.File(fnam, 'w') as f:
                for key in default.keys():
                    if key not in skip and hasattr(self, key):
                        v = getattr(self, key)
                        logger.debug(f'saving {key} {type(v)}')
                        f[key] = v
        else:
            logger.info(f'updating class info in {fnam}')
            with h5py.File(fnam, 'r+') as f:
                for key in default.keys():
                    if key not in skip and hasattr(self, key):
                        v = getattr(self, key)
                        utils.write_h5(f, key, v, compression=False)
                        # f[key][...] = v

    def load(self, fnam, skip=[], overwrite=True):
        try:
            with h5py.File(fnam, 'r') as f:
                for key in default.keys():
                    if key in skip:
                        continue

                    if (
                        not overwrite
                        and getattr(self, key) is not None
                    ):
                        continue

                    if key in f:
                        v = f[key][()]
                    else:
                        logger.warning(f'{key} not found in class file using'
                                       f'default value {key}={default[key]}')
                        v = default[key]

                    if type(v) is bytes:
                        v = v.decode()

                    self.set(key, v)

        except (OSError, BlockingIOError) as e:
            print(e, file=sys.stderr)
            time.sleep(0.1)
            self.load(fnam, skip, overwrite)
