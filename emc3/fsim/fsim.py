"""
Units:

Fourier transform of shape function of a sample (say a polyhedron):
    F(q) = int e^{-2 pi i q . r} f(r) dV   units: m^-3
here, f(r) is unit-less.
Note: I could have chosen f = electron density which would change the units

The differential scattering cross-section is the area of the beam diverted
to a given solid angle:
    d sig / d Omega   units: m^2 / steradian



"""
import sys
import numpy as np
import runpy
from .. import utils_cl
from .. import detector
from . import polyhedron
from .density_pdb_test import Density_pdb
import h5py
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import scipy.constants as sc
import periodictable as pt
# You must explicitly import the submodule for some versions
import periodictable.nsf
import periodictable.cromermann

class I_calc():
    def __init__(self, q, mask, sample, wavelength):
        """
        q: ndarray (3,) + mask.shape

        mask: ndarray
            True: good pixel
            False: bad pixel

        sample: dict
            'type': 'truncated_polyhedron'
            'size': float,
            'truncation': float

        object(R) produces Fourier intensities of rotated object for un-masked q-values
        """
        q = q[:, mask].T

        cl_stuff = utils_cl.opencl_init()

        if sample['type'] == 'truncated_polyhedron':
            verts = polyhedron.get_truncated_octahedron(sample['size'], sample['truncation'])

            i_calc = polyhedron.Fourier_transform_convex_polyhedron_cl(
                cl_stuff['context'],
                cl_stuff['queue'],
                q.shape[:-1],
                verts,
                q,
                scale=True,
                apply_scale_to_output=False
            )
            #I0 = poly_calc().copy()

            # get atomic form factor
            self.form_factor = get_atomic_form_factor(q, sample['formula'], wavelength)

            # xray scattering length densities for molecules
            self.sld = get_complex_contrast(sample['formula'], wavelength)

            # get scale factor for i_calc
            self.scale = self.sld * i_calc.scale**6 * self.form_factor

            self.verts = verts

        elif sample['type'] == 'pdb':
            i_calc = Density_pdb(
                cl_stuff['queue'],
                cl_stuff['context'],
                q,
                sample['pdb'],
                biomol=True
            )

            # get scale factor for i_calc
            self.scale = sc.value('classical electron radius')**2

            self.q = q
        else:
            raise ValueError(f"sample type {config.sample['type']} not supported")

        self.i_calc = i_calc
        self.sample = sample

    def __call__(self, R):
        if self.sample['type'] == 'truncated_polyhedron':
            # faster to rotate verts than q-values
            verts = R.dot(self.verts.T).T
            I = self.i_calc(vertices=verts)
        elif self.sample['type'] == 'pdb':
            q = R.T.dot(self.q.T).T
            I = np.abs(self.i_calc(q=q))**2
        return I


class B_calc():
    def __init__(self, **kwargs):
        if kwargs['type'] != 'constant':
            raise ValueError(f"type={kwargs['type']} not supported")

        shape = kwargs['mask'].shape
        mean = kwargs['mean']
        self.mean = mean
        self.frame = np.zeros(np.sum(kwargs['mask']), dtype=np.float32)

        self.size = self.frame.size
        self.mean_per_pixel = mean / self.size

        self.background = np.zeros((1,) + shape, dtype=np.float32)
        self.background.fill(self.mean_per_pixel)

        self.background_counts = []
        self.background_index = []
        self.background_weighting = []

        if mean > 0:
            self.no_back = False
        else:
            self.no_back = True

    def __call__(self):
        """
        overall background strength is drawn from a poisson
        distribution at the mean level
        """
        if self.no_back:
            self.background_weighting.append(1)
            self.background_index.append(0)
            self.background_counts.append(0)
            return 0

        counts = np.random.poisson(self.mean)
        level = counts / self.size

        self.background_weighting.append(counts / self.mean)
        self.background_index.append(0)
        self.background_counts.append(counts)

        self.frame.fill(level)
        return self.frame


class Beam():
    def __init__(self, size, pulse_photons, minimum_fluence):
        """
        size: scalar
            standard deviations of Gaussian beam in metres

        pulse_photons: scalar
            total number of photons in beam over exsposure period

        minimum_fluence: float
            minimum allowed incident fluence in J/m^2

        object = Beam()
        object.random_fluence(): float

        Notes:
        Fluence distribution is given by:
            f(x, y) = A exp[-r^2 / (2 sig^2)]
            A = pulse_photons / (2 pi sig^2)

        r_max:
            f(r) = minimum_fluence
            r = sqrt[ -2 sig^2 ln(minimum_fluence / A) ]
        """
        self.A = pulse_photons / (2 * np.pi * size**2)
        self.r_max = (-2 * size**2 * np.log(minimum_fluence / self.A))**0.5
        self.size = size

    def peak_fluence(self):
        return self.A

    def random_fluence(self):
        # get random distance with r < r_max
        r = np.random.random()**0.5 * self.r_max

        # sample fluence dist.
        f = self.A * np.exp(-r**2 / (2 * self.size**2))
        return f


def get_atomic_form_factor(q, formula, wavelength):
    """
    calculate the amtomic form factor for 'formula' using cromer mann coefficients:
        f(s) = c + \sum_i=1^4 a_i exp[ -b_i s^2 ]

        s = sin(theta_scat) / wavelength

    return f(s) / f(0)

    |q| = sin(theta) / wavelength
    where theta is the geometric angle between incomming and scattered waves

    q.shape = (pixels, 3)

    formula = 'Au'

    theta_scat = arcsin(wavelength * |q|) / 2

    s = sin(theta_scat) / wavelength
      = sin(arcsin(wavelength * |q|) / 2) / wavelength
    """
    qmag = np.sum(q**2, axis=-1)**0.5
    theta_scat = np.arcsin(qmag * wavelength) / 2
    s = np.sin(theta_scat) / wavelength

    form_factor = pt.cromermann.fxrayatstol(formula, s * 1e-10)
    form_factor /= pt.cromermann.fxrayatstol(formula, [0])
    return form_factor


def get_complex_contrast(mat_formula, wavelength, sol_formula=None, mat_density=None, sol_density=None):
    """
    Calculates the complex scattering length density (SLD) contrast.
    Returns the squared absolute value of the contrast in units of A^-4.

    Scattering Length: This is the "strength" of a single scatterer (like an atom). Units: metres (m).
    Scattering Length Density: This is the scattering length per unit volume. Units: m/m^3 = m^-2
    output = |Scattering Length Density|^2 = m^-4

    # --- Usage ---
    # Gold: Au, density ~19.3 g/cm^3
    # Solvent (Water): H2O, density ~1.0 g/cm^3
    contrast_sq = get_complex_contrast("Au", 19.3, "H2O", 1.0)

    print(f"Complex Contrast Factor (|Δρ|^2): {contrast_sq.item():.4e} Å^-4")
    """
    if mat_density is not None:
        mat_density *= 1e-3

    if sol_density is not None:
        sol_density *= 1e-3

    # The sld function handles the conversion using the classical electron radius.
    # Results are in 10^-6 A^-2
    m_sld = pt.xray_sld(mat_formula, wavelength=1e10 * wavelength, density=mat_density)
    m_sld = m_sld[0] + 1j * m_sld[1]

    if sol_formula is not None:
        if sol_density is None and sol_formula == 'H2O':
            sol_density = 1e-3

        s_sld = pt.xray_sld(sol_formula, wavelength=1e10 * wavelength, density=sol_density)
    else:
        s_sld = (0., 0.)

    s_sld = s_sld[0] + 1j * s_sld[1]

    # 3. Compute complex contrast (convert from 10^-6 A^-2 to m^-2)
    delta_rho = 1e14 * (m_sld - s_sld)

    # 4. Return the squared absolute contrast factor (|Δρ|^2)
    # This is the scalar value that scales your physical intensity
    return np.abs(delta_rho)**2


def get_dtype_max(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).max
    else:
        # Handle other types like string, bool, etc. if necessary
        raise TypeError(f"Data type {dtype} is not a numeric type with machine limits.")


if __name__ == '__main__':
    config_file = sys.argv[1]
    config = runpy.run_path(config_file)

    if config['mask_file'] is not None:
        with h5py.File(config['mask_file']) as f:
            mask = f[config['mask_dset']][()]
    else:
        mask = None

    detector_distance = config.get('detector_distance', False)

    det = detector.Detector_cxi(config['cxi_file'], mask, scale_correction=False, detector_distance=detector_distance)

    if 'max_pix_rad' in config and config['max_pix_rad']:
        x, y = det.xyz[:2]
        r = (x**2 + y**2)**0.5 / det.pixel_size
        mask *= r < config['max_pix_rad']

    frame = np.zeros(det.shape, config['dtype'])

    with h5py.File(config['cxi_file']) as f:
        pulse_energy = f['entry_1/instrument_1/source_1/pulse_energy'][()]
        wavelength = f['entry_1/instrument_1/source_1/photon_wavelength'][()]

    wavelength = np.mean(wavelength)
    pulse_energy = np.max(pulse_energy)

    if config['pulse_energy_factor'] is not None:
        pulse_energy *= config['pulse_energy_factor']

    E_photon = sc.h * sc.c / wavelength
    pulse_photons = pulse_energy / E_photon

    # focal spot size = 2 x sigma
    beam = Beam(config['focal_spot_size']/2., pulse_photons, config['minimum_fluence'])
    i_calc = I_calc(det.q, det.mask, config['sample'], wavelength)
    b_calc = B_calc(mask=det.mask, **config['background'])

    D = config['nhits']
    shape = (D,) + det.shape
    rng = np.random.default_rng()
    C = det.C[mask].copy()

    C = C * i_calc.scale

    C_scale = C.max()
    C /= C_scale

    saturation = get_dtype_max(config['dtype'])

    with h5py.File(config['output_file'], 'w') as f:
        frames = f.create_dataset(
            'entry_1/instrument_1/detector_1/data',
            shape=shape,
            dtype=config['dtype'],
            compression='gzip',
            compression_opts=1,
            chunks=(1,) + shape[1:]
        )

        f['entry_1/instrument_1/detector_1/good_pixels'] = det.mask
        f['entry_1/instrument_1/detector_1/xyz_map'] = det.xyz
        f['entry_1/instrument_1/detector_1/pixel_size'] = det.pixel_size
        f['entry_1/instrument_1/detector_1/x_pixel_size'] = det.pixel_size
        f['entry_1/instrument_1/detector_1/y_pixel_size'] = det.pixel_size

        f['entry_1/instrument_1/source_1/photon_energy'] = E_photon * np.ones((D,))
        f['entry_1/instrument_1/source_1/pulse_energy'] = pulse_energy * np.ones((D,))
        f['entry_1/instrument_1/source_1/photon_wavelength'] = wavelength * np.ones((D,))

        counts = []
        lit = []
        fluence = []
        sat = []
        # accumulate scale factors
        for d in tqdm(range(D)):
            # choose random sample orientation
            R = Rotation.random().as_matrix()

            # get Fourier transform of object
            I = i_calc(R)

            # scale by fluence
            #w = beam.random_fluence()
            w = beam.peak_fluence()
            fluence.append(w)

            I *= w * C_scale * C

            # background
            I += b_calc()

            # poisson sampling
            if config['photon_counting']:
                photons = rng.poisson(I)
            else:
                photons = I

            counts.append(np.sum(photons))
            lit.append(np.sum(photons>0))

            sat.append(np.sum(photons>saturation))

            frame.fill(0)
            frame[det.mask] = np.clip(photons, 0, saturation)

            # output
            frames[d] = frame

        f['entry_1/background_counts'] = np.array(b_calc.background_counts)
        f['entry_1/background_index'] = np.array(b_calc.background_index)
        f['entry_1/background_weighting'] = np.array(b_calc.background_weighting)
        f['entry_1/instrument_1/detector_1/background'] = b_calc.background

        f['entry_1/instrument_1/detector_1/photon_counts'] = np.array(counts)
        f['entry_1/instrument_1/detector_1/lit_pixels'] = np.array(lit)
        f['entry_1/instrument_1/detector_1/saturated_pixels'] = np.array(sat)
        f['entry_1/instrument_1/source_1/fluence'] = fluence
        f['entry_1/data_1/data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
