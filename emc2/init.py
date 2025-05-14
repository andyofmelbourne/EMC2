import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from emc2 import geometry
from emc2 import data_getter
from emc2 import orientations
from emc2 import symmetry
from emc2 import classes

import logging

logger = logging.getLogger(__name__)


def add_geometry(class_c):
    logger.debug('adding geometry (start)')
    logger.info('calculating mask and pixel geometry')

    geom = geometry.geometry(
        voxel_cut=(0, class_c['zero_padding']),
        **class_c)

    class_c.update(geom)

    # get P_mask etc
    if 'P_mask_padding' in class_c:
        logger.debug('updating geometry for probability matrix')
        geom = geometry.geometry(
            voxel_cut=class_c['P_mask_padding'],
            **class_c)

    class_c['P_mask'] = geom['mask']
    class_c['P_C'] = geom['C']
    class_c['P_xyz'] = geom['xyz']

    logger.debug('adding geometry (stop)')


def check_data(class_c):
    logger.debug('checking data (start)')

    data_getter.Data_getter(
        mask=class_c['P_mask'],
        split_frames=class_c['split_frames'],
        cxi_file=class_c['cxi_file'],
        filter=class_c['filter'],
        frame_model=class_c['frame_model'],
        working_directory=class_c['working_directory'],
        delay_data_load=True
    )

    K_di = data_getter.Data_getter(
        mask=class_c['mask'],
        split_frames=class_c['split_frames'],
        cxi_file=class_c['cxi_file'],
        filter=class_c['filter'],
        frame_model=class_c['frame_model'],
        working_directory=class_c['working_directory'],
        delay_data_load=True
    )

    with h5py.File(K_di.sparse_fnam) as f:
        class_c['ksums'] = f['photon_sums'][()]
        class_c['frame_selection'] = f['frames'][()]
        class_c['frames'] = f['frames'].shape[0]

    logger.debug('checking data (stop)')


def blob(N, dq):
    sigma_z = 8 * 5.58661e+06
    sigma_x = 4 * 5.58661e+06
    i = dq * np.fft.fftshift(np.fft.fftfreq(N, 1/N))
    x = i[:, None, None]
    y = i[None, :, None]
    z = i[None, None, :]
    I = np.exp(-(x**2 + y**2)/(2*sigma_x**2) - z**2/(2*sigma_z**2))
    return I


def init_model(c):
    # initialise models
    N = c['model_length']
    if 'model_init' in c and c['model_init'] == 'blob':
        I = blob(N, c['dq'])
        I *= (np.random.random(I.shape) + 0.1)
    else:
        shape = c['dimensions'] * (N,)
        I = np.random.random(shape)
    c['model'] = I


def init_fluence(c):
    c['relative_fluence'] = np.ones(c['frames'], dtype=float)


def init_mapping(c, rotation_matrices):
    # calculate rotation matrices
    d, r = c['dimensions'], c['rotation_order']
    if (d, r) not in rotation_matrices:
        desc = f'calculating rotation matrices for class {c["class_id"]}'
        for _ in tqdm([1], desc=desc):
            rotation_matrices[(d, r)] = orientations.get_rotation_matrices(
                queue=c['queue'],
                context=c['context'],
                rotation_order=r,
                dimensions=d
            ).get()
        logger.info(
            f'generating rotation matrices for dimension {d} and rotation '
            f'order {r}: shape = {rotation_matrices[(d, r)].shape}'
        )
    else:
        logger.info(
            f'already have rotation matrices for '
            f'dimension {d} and rotation_order {r}'
        )

    if hasattr(c['xyz_offset'], '__len__'):
        t = c['xyz_offset']
        for dr in t:
            assert (len(dr) == 3)

        r_offsets = t
    else:
        r_offsets = [[0, 0, 0]]

    logger.info(f'{r_offsets=}')

    if 'scale' in c and c['scale'] is not None:
        scales = c['scale']
    else:
        scales = [1.]
        c['scale'] = scales

    logger.debug(f'{scales=}')
    logger.debug(f"{c['dimensions']=}")
    logger.debug(f"{c['symmetry']=}")
    logger.debug(f"{c['dq']=}")
    logger.debug(f"{c['wavelength']=}")
    logger.debug(f"{c['model_length']=}")

    # get symmetry opperators
    S0 = symmetry.get_non_voxel_operators(c['dimensions'], c['symmetry'])

    logger.debug(f"{S0.shape=}")

    # make transformation parameters
    # n_si = A_sr . (r-dr) / |r-dr| + b_sr
    # A_sr = S_s . R_r / wav dq
    # b_sr = - ( A_sr . (0, 0, 1) + i0)
    # T = {dr, b, A}
    R0 = rotation_matrices[(d, r)]

    # add third dimension for 2D
    if d == 2:
        R = np.zeros(R0.shape[:1] + (3, 3), dtype=float)
        R[:, :2, :2] = R0
        R[:, 2, 2] = 1

        S = np.zeros(S0.shape[:1] + (3, 3), dtype=float)
        S[:, :2, :2] = S0
        S[:, 2, 2] = 1
    else:
        R = R0
        S = S0

    T_sr = np.zeros(
        (S.shape[0], len(scales) * len(r_offsets) * R.shape[0], 5, 3),
        dtype=np.float32
    )
    orientation_index_r = np.empty(T_sr.shape[1], dtype=np.int32)
    scale_r = np.empty(T_sr.shape[1], dtype=np.float32)
    x_offset_r = np.empty(T_sr.shape[1], dtype=np.float32)
    y_offset_r = np.empty(T_sr.shape[1], dtype=np.float32)
    z_offset_r = np.empty(T_sr.shape[1], dtype=np.float32)
    for si, s in enumerate(S):
        index = 0
        for scale in scales:
            for dri, dr in enumerate(r_offsets):
                # R.shape = (M, 3, 3)
                # s.shape = (3, 3)
                # A.shape = (M, 3, 3)
                A = scale * s.dot(R).transpose(1, 0, 2) \
                        / (c['wavelength'] * c['dq'])
                b = c['model_length']//2 - A[:, :, 2]
                M = A.shape[0]
                T_sr[si, index:index+M, 0] = dr
                T_sr[si, index:index+M, 1] = b
                T_sr[si, index:index+M, 2:5] = A
                if si == 0:
                    orientation_index_r[index:index+M] = np.arange(M)
                    x_offset_r[index:index+M] = dr[0]
                    y_offset_r[index:index+M] = dr[1]
                    z_offset_r[index:index+M] = dr[2]
                    scale_r[index:index+M] = scale
                index += M

    c['mapping_matrix'] = T_sr
    c['orientation_index_r'] = orientation_index_r
    c['x_offset_r'] = x_offset_r
    c['y_offset_r'] = y_offset_r
    c['z_offset_r'] = z_offset_r
    c['scale_r'] = scale_r


def save_classes(c, overwrite=True, check=True):
    logger.info('init/save_class (start)')

    class_c = classes.Class(**c)

    # write class file
    fnam = Path.joinpath(
        Path(c['working_directory']),
        f'class_{c["class_id"]}.h5'
    )
    class_c.save(fnam, overwrite=overwrite, check=check)
    logger.info('init/save_class (stop)')

    return True
