import numpy as np

class Detector():
    """
    keeps information about an x-ray detector in a particular event or run:
        x, y and z coordinates of each pixel with respect to the sample
        x-ray beam parameters
        pixel mask (good or bad)

    Arguments:
    ----------
    xyz: array_like
    wavelength: float
    polarisation: 'x', 'y' or None
    mask: array_like
    """

    def __init__(self, xyz, wavelength, polarisation, mask, pixel_area, scale_correction=True):
        self.mask = mask
        self.wavelength = wavelength
        self.xyz = xyz
        self.polarisation = polarisation
        self.mask = mask
        self.pixel_size = pixel_area**0.5

        # distance from sample
        self.r = r = np.sum(xyz**2, axis=0)**0.5

        # assume normal
        self.detector_distance = z = xyz[2].ravel()[0]

        # polarisation correction:
        if polarisation == 'x':
            P = 1 - (xyz[0] / r)**2
        elif polarisation == 'y':
            P = 1 - (xyz[1] / r)**2
        elif p is None:
            P = np.ones(xyz.shape[1:])

        self.P = P

        # solid angle correction
        self.Omega = pixel_area * xyz[2] / r**3

        # merged intensity to frame correction factor (scaled)
        # diffraction = C x W
        self.C = self.Omega * self.P

        if scale_correction:
            self.C /= self.C.max()

        # low-angle dq
        self.dq = pixel_area**0.5 / wavelength / z

        # q-magnitude
        q = xyz / r
        q[2] -= 1
        q /= wavelength
        self.qr = qr = np.sum(q**2, axis=0)**0.5
        self.q = q

        self.qmin = qr[mask].min()
        self.qmax = qr[mask].max()

        self.shape = mask.shape


def Detector_cxi(
    cxi_file=None,
    mask=None,
    scale_correction=True,
    detector_distance=False,
    xyz_map=None
    ):
    import h5py
    with h5py.File(cxi_file) as f:
        # optionally over-ride geometry
        k = 'entry_1/instrument_1/detector_1/xyz_map'
        shape = f[k].shape
        if xyz_map is not None and xyz_map.shape == shape:
            xyz = xyz_map
        elif not xyz_map:
            xyz = f[k][()]
        else:
            raise ValueError(f'xyz_map has the wrong shape! {xyz_map.shape=} {shape=}')

        k1 = 'entry_1/instrument_1/detector_1/pixel_area'
        k2 = 'entry_1/instrument_1/detector_1/x_pixel_size'
        k3 = 'entry_1/instrument_1/detector_1/y_pixel_size'
        if k1 in f:
            pixel_area = f[k1][()]
        elif k2 in f and k3 in f:
            pixel_area = f[k2][()] * f[k3][()]
        elif k2 in f:
            pixel_area = f[k2][()]**2

        wav = f['/entry_1/instrument_1/source_1/photon_wavelength'][0]

        if mask is None:
            mask = f['entry_1/instrument_1/detector_1/good_pixels'][()]

    polarisation = 'x'

    # optionally override detector distance
    if detector_distance:
        xyz[2] = detector_distance

    detector = Detector(xyz, wav, polarisation, mask, pixel_area, scale_correction)

    return detector
