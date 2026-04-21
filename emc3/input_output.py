from pathlib import Path
import runpy
import h5py
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_option(d, thing):
    if thing in d:
        return d[thing]
    else:
        return False


def load_config(path):
    logger.debug(f'\nloading configuration file from {path}')
    p = Path(path)

    # returns a dict
    config = runpy.run_path(str(p.absolute()))

    config.update(set_working_directory(p))

    return config


def set_working_directory(path):
    wd = str(Path(path).resolve().parent)
    logger.debug(f'working directory: {wd}')
    return {'working_directory': wd}


def set_iteration_info_fnam(working_directory):
    fnam = Path.joinpath(Path(working_directory), 'iteration_info.h5')
    return {'iteration_info': str(fnam)}


def get_iterations(**config):
    fnam = Path(config['working_directory']).joinpath('iteration_info.h5')
    if fnam.is_file:
        with h5py.File(fnam) as f:
            iterations = f['iterations'][()]
    else:
        iterations = 0
    return iterations


def get_iteration_number(config):
    if get_option(config, 'restart'):
        iteration = 0
        logger.debug(f'"restart" is True setting iteration to {iteration}')
    else:
        # check iteration info file
        fnam = Path(config['iteration_info'])
        if fnam.is_file():
            with h5py.File(fnam, 'r') as f:
                iteration = f['iterations'][()]
            logger.debug(f'Getting iteration number from {fnam}. '
                         f'Setting iteration to {iteration}')
        else:
            iteration = 0
            logger.debug(f'Setting iteration to {iteration}')

    return iteration

def write_h5(f, k, v, compression=True, chunks=None):
    if compression:
        compression = 'gzip'
    else:
        compression = None

    if not hasattr(v, 'shape') or len(v.shape) == 0:
        if k in f:
            del f[k]
        f[k] = v
    else:
        if k in f:
            if f[k].shape == v.shape and f[k].dtype == v.dtype:
                f[k][:] = v
            else:
                del f[k]

        if k not in f:
            if not chunks and compression:
                chunks = v.shape
            f.create_dataset(k, data=v, chunks=chunks, compression=compression)

def save_iteration_info(
    P_max_d,
    Q_d,
    Q_old_d,
    class_max_d,
    local_rmax_d,
    occupancy_dc,
    occupancy_r,
    working_directory,
    beta,
    sparse_file=''
):
    fnam = Path(working_directory).joinpath('iteration_info.h5')

    # get iteration number
    if fnam.is_file():
        with h5py.File(fnam, 'r') as f:
            N = f['iterations'][()]
    else:
        N = 0

    logger.info(f'saving iteration info to {fnam} '
                f'for iteration {N}')

    # initialise or resize datasets
    if N == 0:
        with h5py.File(fnam, 'w') as f:
            f['iterations'] = N+1

            f.create_dataset(
                'beta',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'Q',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'dQ',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'P_gini',
                shape=(1,),
                maxshape=(None,),
                dtype=np.float32
            )

            f.create_dataset(
                'orientation_changes',
                shape=(1,),
                maxshape=(None,),
                dtype=np.int32
            )

            f.create_dataset(
                'class_changes',
                shape=(1,),
                maxshape=(None,),
                dtype=np.int32
            )
    else:
        # resize
        keys = ['beta', 'Q', 'dQ', 'P_gini', 'orientation_changes', 'class_changes']
        with h5py.File(fnam, 'r+') as f:
            for key in keys:
                f[key].resize(N+1, axis=0)

    # write other results to iteration_{N}
    with h5py.File(fnam, 'r+') as f:
        k = f'iteration_{N}'
        if k in f:
            g = f[k]
        else:
            g = f.create_group(k)

        write_h5(g, 'occupancy_r', occupancy_r)
        write_h5(g, 'P_gini_d', P_max_d)
        write_h5(g, 'Q_d', Q_d)
        write_h5(g, 'most_likely_model_d', class_max_d)
        write_h5(g, 'occupancy_dc', occupancy_dc)
        write_h5(g, 'most_likely_orientation_d', local_rmax_d)
        write_h5(g, 'sparse_file', sparse_file)
        f['iterations'][...] = N+1
        f['beta'][N] = beta
        f['Q'][N] = np.mean(Q_d)
        if N > 0:
            f['dQ'][N] = np.mean(Q_old_d - f[f'iteration_{N-1}/Q_d'][()])
        else:
            f['dQ'][N] = 0
        f['P_gini'][N] = np.mean(P_max_d)

        # write differences
        if N > 0:
            k0 = f'iteration_{N-1}'
            g0 = f[k0]

            mlm_d0 = g0['most_likely_model_d'][()]
            mlm_d1 = class_max_d

            # if the number of patterns has changed
            # numbers might be rubish
            D = min(mlm_d0.shape[0], mlm_d1.shape[0])
            dc = np.sum(mlm_d0[:D] != mlm_d1[:D])
            f['class_changes'][N] = dc

            mlo_d0 = g0['most_likely_orientation_d'][()]
            mlo_d1 = local_rmax_d

            D = min(mlo_d0.shape[0], mlo_d1.shape[0])
            do = np.sum(mlo_d0[:D] != mlo_d1[:D])
            f['orientation_changes'][N] = do
        else:
            f['class_changes'][N] = Q_d.shape[0]
            f['orientation_changes'][N] = Q_d.shape[0]

    return True
