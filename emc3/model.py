import numpy as np
import h5py
from tqdm import tqdm

from . import symmetry
from . import utils
from .tomograms import Tomograms

class Model():
    """
    Store information about a model and provide initialisation routnes
    """

    def __init__(
            self,
            shape=None,
            dq=None,
            i0=None,
            symmetry='P1',
            class_id=None,
            ):

        self.shape = shape
        self.size = np.prod(shape)
        self.dq = dq
        self.data = None
        self.dtype = np.float64
        self.ndim = len(shape)
        self.symmetry = symmetry

        if class_id is None:
            class_id = id(self)

        self.class_id = class_id

        # require cube
        for s in shape:
            assert (s==shape[0])

        if i0 is None:
            self.i0 = shape[0]//2
        else:
            self.i0 = i0

        self.qmax = dq * max(abs(self.i0), abs(shape[0]-1-self.i0))

    def init_random(self):
        self.data = np.random.random(self.shape)


def update_I_class(c):
    P_dr = c['P_dr']
    C_i = c['data'].C_i
    w_d = c['fluence']
    likelihood = c['likelihood']
    frame_model = c['frame_model']
    ksums = c['data'].data_sum
    mapper = c['mapper']
    D, R = P_dr.shape
    I = c['data'].shape[1]
    N = c['model'].shape[0]

    N_n = np.zeros(c['model'].data.size, dtype=float)
    D_n = np.zeros(c['model'].data.size, dtype=float)

    assert (D == c['data'].shape[0])

    mapper.load_coords(c['data'].mask)

    tomos = Tomograms(
            mapper,
            c['model'],
            c['data'].C_i,
            c['fluence'])

    wsums_r = np.empty(R)
    for r in range(R):
        wsums_r[r] = np.sum(tomos.calculate_tomogram(r))

    N_ri = np.dot(P_dr.T, c['data'][:])

    if (
        likelihood == 'Poisson' and
        frame_model == 'basic'
    ):
        D_ri = C_i[None, :] * \
                np.sum(P_dr, axis=0)[:, None]

    elif (
        likelihood == 'Poisson'
        and frame_model == 'fluence'
    ):
        D_ri = C_i[None, :] * np.dot(w_d, P_dr)[:, None]

    elif (
        likelihood == 'Poisson_fluence_free'
        and frame_model == 'basic'
    ):
        N_ri *= wsums_r[:, None]
        D_ri = C_i[None, :] * \
            np.dot(ksums, P_dr)[:, None]

    else:
        raise ValueError(f'could not parse likelihood {class_c.likelihood}'
                         f'and frame_model {class_c.frame_model}')

    n_sri = np.empty(mapper.shape + (I,), dtype=int)

    for s in range(mapper.shape[0]):
        for r in range(mapper.shape[1]):
            n_sri[s, r] = mapper.calculate_mapping_ravel(r, s)

    # now merge N_ri and D_ri to I-space
    if c['maximise'] == 'W':
        D_ri[D_ri == 0] = 1.
        N_ri /= D_ri
        D_ri[:] = 1.

    for s in tqdm(range(n_sri.shape[0]), leave=False):
        for r in range(n_sri.shape[1]):
            N_n += np.bincount(
                n_sri[s, r],
                N_ri[r],
                minlength=c['model'].data.size
            )

            D_n += np.bincount(
                n_sri[s, r],
                D_ri[r],
                minlength=c['model'].data.size
            )

    # apply symmetry
    i0 = N // 2

    sym = symmetry.Symmetry(
            c['model'].i0,
            c['model'].shape,
            c['model'].symmetry
    )

    N_n = sym.apply_symmetry(
        N_n.reshape(c['model'].shape),
    )

    D_n = sym.apply_symmetry(
        D_n.reshape(c['model'].shape),
    )

    # I = N / D
    D_n[D_n == 0] = 1.
    N_n /= D_n

    print(f'{N_n.shape=}')
    return N_n


# basic all in memory cpu update
def update_I(config):
    for c in config['classes']:
        if not c['update_model']:
            continue

        c['model'].data = update_I_class(c)

    models = [c['model'].data for c in config['classes']]

    dq = config['classes'][0]['model'].dq

    utils.save_model_slices(
            models,
            dq,
            config['working_directory']
            )



