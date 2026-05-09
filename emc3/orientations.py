"""
use orix to evenly sample orientations within the point group

backwards compatibility:
    I have been adding inversion symmetry to point group strings
    so 'D6' really means 'D6h' etc. But not for P1
    I will hack this in but should change later
"""
import pyopencl as cl
import pyopencl.array
import numpy as np
import tqdm

from orix.quaternion import symmetry
from orix.sampling import get_sample_fundamental

def get_rotation_matrices(
    queue=None,
    context=None,
    rotation_order=10,
    dimensions=3,
    symmetry_str='P1'
):
    # angular spacing in deg.
    resolution = np.arctan2(1, rotation_order) * 180 / np.pi

    if dimensions == 3:
        try:
            # hack
            if symmetry_str == 'P1':
                point_group = getattr(symmetry, 'C1')
            elif symmetry_str == 'inversion':
                point_group = getattr(symmetry, 'C1').laue
            else:
                point_group = getattr(symmetry, symmetry_str).laue
        except AttributeError:
            raise ValueError(f"Symmetry '{pg_string}' not found in orix.")

        print(f'{resolution=} {symmetry_str=} {point_group=}')
        quats = get_sample_fundamental(resolution=resolution, point_group=point_group)

        R = quats.to_matrix()

    elif dimensions == 2:
        # --- 2D Logic (Manual) ---
        # Extract the fold-number from strings like 'D6h', 'C6', or '6'
        import re

        # hack
        if symmetry_str == 'P1':
            pg_string = 'C1'
        elif symmetry_str == 'inversion':
            pg_string = 'Ci'

        match = re.search(r'\d+', pg_string)
        n_fold = int(match.group()) if match else 1

        # Fundamental zone for n-fold symmetry is [0, 2pi/n)
        fz_limit = 2 * np.pi / n_fold

        # resolution in degrees converted to radians
        res_rad = np.radians(resolution)
        n_points = int(np.ceil(fz_limit / res_rad))
        angles = np.linspace(0, fz_limit, n_points, endpoint=False)

        # Vectorised creation of 2x2 rotation matrices
        c, s = np.cos(angles), np.sin(angles)
        # Reshape into (N, 2, 2)
        R = np.array([[c, -s], [s, c]]).transpose(2, 0, 1)

    return R
