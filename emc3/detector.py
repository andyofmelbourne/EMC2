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

    def __init__(self, xyz, wavelength, polarisation, mask, pixel_area):
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


def Detector_cxi(cxi_file=None, mask=None):
    import h5py
    with h5py.File(cxi_file) as f:
        xyz = f['entry_1/instrument_1/detector_1/xyz_map'][()]
        pixel_area = f['entry_1/instrument_1/detector_1/pixel_area'][()]
        wav = f['/entry_1/instrument_1/source_1/photon_wavelength'][0]

        if mask is None:
            mask = f['entry_1/instrument_1/detector_1/good_pixels'][()]

    polarisation = 'x'

    detector = Detector(xyz, wav, polarisation, mask, pixel_area)

    return detector
