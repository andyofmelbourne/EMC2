import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)


def geometry(
    cxi_file=None,
    polarisation=None,
    model_length=None,
    pixels_per_voxel=None,
    # zero_padding=None,
    voxel_cut=None,
    xyz_offset=None,
    **config
):
    """
    calculate:
        - mask   : pixel mask based on config
        - q      : Ewald sphere q-values in reference orientation
        - dq     : q-space voxel size of model
        - q_max  : 1 / full period resolution limit within pixel mask
        - C      : the solid angle and polarisation correction factor
    """
    zero_padding = None

    # calculate q-values
    logger.debug('loading datasets for geometry calculation (start)')
    with h5py.File(cxi_file) as f:
        mask = f['entry_1/instrument_1/detector_1/good_pixels'][()]

        # frames = f['entry_1/data_1/data'].shape[0]

        # pixel map
        xyz = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
        # wav = np.mean(
        #     f['/entry_1/instrument_1/source_1/photon_wavelength'][()]
        # )
        # saves time
        wav = f['/entry_1/instrument_1/source_1/photon_wavelength'][0]
        dx = f['entry_1/instrument_1/detector_1/x_pixel_size'][()]
        dy = f['entry_1/instrument_1/detector_1/y_pixel_size'][()]

        # get pixel area for non-square pixels (DSSC)
        key = 'entry_1/instrument_1/detector_1/pixel_area'
        if key in f:
            pixel_area = f[key][()]
        else:
            pixel_area = dx * dy
    logger.debug('loading datasets for geometry calculation (stop)')

    if hasattr(xyz_offset, '__len__'):
        xyz_offset = np.atleast_2d(xyz_offset)

        xyz_mean_offset = np.mean(xyz_offset, axis=0)
        xyz[0] += xyz_mean_offset[0]
        xyz[1] += xyz_mean_offset[1]
        xyz[2] += xyz_mean_offset[2]

        xyz_offset -= xyz_mean_offset

    # calculate pixel radius
    r = np.sum(xyz**2, axis=0)**0.5
    q = xyz.copy() / r
    q[2] -= 1
    q /= wav
    qr = np.sum(q**2, axis=0)**0.5

    if polarisation == 'x':
        P = 1 - (xyz[0] / r)**2
    elif polarisation == 'y':
        P = 1 - (xyz[1] / r)**2
    elif polarisation is None:
        P = np.ones(xyz.shape[1:])

    # solid angle correction
    Omega = pixel_area * xyz[2] / r**3

    # merged intensity to frame correction factor
    C = Omega * P

    # scale
    C /= C[mask].max()

    M = model_length

    # zero padding:
    # calculate the pixel mask with "zero padding" voxels subtracted
    # from each dimension of the model
    if zero_padding:
        M = M - 2 * zero_padding

    if 'q_max' in config and config['q_max']:
        q_max = config['q_max']

    elif 'res_max' in config and config['res_max']:
        q_max = 1 / config['res_max']

    elif 'pixel_radius' in config and config['pixel_radius']:
        rp = config['pixel_radius']
        z = xyz[2].ravel()[0]
        r = (rp**2 + z**2)**0.5
        q_max = (rp**2 + (z-r)**2)**0.5 / wav / r

    elif pixels_per_voxel and M:
        rp = dx * pixels_per_voxel * (M // 2)
        z = xyz[2].ravel()[0]
        r = (rp**2 + z**2)**0.5
        q_max = (rp**2 + (z-r)**2)**0.5 / wav / r

    else:
        raise ValueError('not enough information to calculate q-mask')

    mask[qr > q_max] = False

    # calculate model q-space voxel size
    # such that the zero pixel (i0) satisfies:
    #   np.fft.fftshift(np.fft.fftfreq(N))[i0] = 0
    if pixels_per_voxel:
        if (M % 2) == 0:
            dq = q_max / (M / 2 - 1)
        else:
            dq = 2 * q_max / (M - 1)

    # increase qmax for model if required
    q_min_model = qr[mask].min()

    if zero_padding:
        q_max_model = q_max + zero_padding * dq
        q_min_model -= zero_padding * dq
        q_min_model = min(0., q_min_model)
    else:
        q_max_model = q_max

    logger.debug(f'{q_max=} {q_max_model=}')

    if voxel_cut:
        n, m = voxel_cut
        qmax = q_max_model - m * dq
        qmin = q_min_model + n * dq

        mask[qr > qmax] = False
        mask[qr < qmin] = False

    # location of zero pixel in models along each axis
    i0 = np.float32(M//2)

    out = {
        'mask': mask,
        'C': np.ascontiguousarray(C[mask].astype(np.float32)),
        'q': np.ascontiguousarray(q[:, mask].astype(np.float32)),
        'xyz': np.ascontiguousarray(xyz[:, mask].astype(np.float32)),
        'dq': np.float32(dq),
        'i0': i0,
        'wavelength': wav,
        'q_max': q_max,
        'q_max_model': q_max_model,
        'q_min_model': q_min_model,
        'xyz_offset': xyz_offset
    }
    return out
