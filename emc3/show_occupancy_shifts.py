import pickle
import numpy as np
import pyqtgraph as pg
import sys
from pathlib import Path
import h5py


if __name__ == '__main__':
    """
    this breaks if there is more than one mapper
    """
    config_file = sys.argv[1]
    iter_file = sys.argv[2]
    iteration = int(sys.argv[3])

    if not Path(config_file).is_file():
        err = f'could not find config file {config_file}'
        raise ValueError(err)

    if not Path(iter_file).is_file():
        err = f'could not find iteration info file {iter_file}'
        raise ValueError(err)

    config = pickle.load(open(config_file, 'rb'))

    # Mapping matrix:
    #   M_sr -> M_sjklxx:
    #       s = symmetry index (S_s)
    #       j = offset index (dr_j)
    #       k = scale index (scale_k)
    #       l = orientation index (R_l)

    # check if there is only one mapper
    mapper = config['classes'][0]['mapper']
    for ci, c in enumerate(config['classes']):
        if mapper is not c['mapper']:
            raise ValueError(f'class {ci} has a different mapper to class 0, thus breaking this script')

    # unique offset vectors
    offsets = mapper.offsets

    # check if there is more than one offset
    if offsets.ndim == 0:
        raise ValueError(f'offsets there are no offsets!')
    elif offsets.ndim > 2:
        raise ValueError(f'cannot understand offsets array with dimensions != 2 {offsets.ndim=}!')

    C = len(config['classes'])

    # offset index for each r
    offsets_ind_r = -np.ones(C * mapper.shape[0] * mapper.shape[1], dtype=int)

    # indices for each class
    S, J, K, L, _, _ = mapper.M_sjkl.shape

    # global occupance for each r-index
    with h5py.File(iter_file) as f:
        occupancy_r = f[f'/iteration_{iteration}/occupancy_r'][()]

    assert(offsets_ind_r.shape[0] == len(occupancy_r))

    r_offset = 0
    for ci in range(C):
        _, J_r, _, _ = np.indices((S, J, K, L))

        offsets_ind_r[r_offset: r_offset + J_r.size] = J_r.ravel()

        r_offset += mapper.shape[0] * mapper.shape[1]

    assert(r_offset == len(occupancy_r))

    print(f'{offsets=}')
    a = np.bincount(offsets_ind_r, weights=occupancy_r)
    pg.plot(a)
    pg.show(a.reshape(5,5))
    pg.exec()
