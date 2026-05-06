from pathlib import Path
import runpy
import h5py
import logging
import numpy as np

from . import profiling

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

@profiling.timed
def save_iteration_info(stats, working_directory):
    """
    Write per-iteration statistics to iteration_info.h5.

    Parameters
    ----------
    stats : dict
        Output of ``probability.calculate_P``.  Required keys:
        ``P_max_d``, ``Q_d``, ``Q_old_d``, ``class_max_d``,
        ``local_rmax_d``, ``occupancy_dc``, ``occupancy_r``, ``beta``.
        Optional: ``sparse_file``.
    working_directory : str or Path

    Returns
    -------
    dict
        A copy of *stats* augmented with the computed scalars:
        ``N`` (iteration index), ``orientation_changes``,
        ``class_changes``, ``dQ``.
    """
    P_max_d      = stats['P_max_d']
    Q_d          = stats['Q_d']
    Q_old_d      = stats['Q_old_d']
    class_max_d  = stats['class_max_d']
    local_rmax_d = stats['local_rmax_d']
    occupancy_dc = stats['occupancy_dc']
    occupancy_r  = stats['occupancy_r']
    beta         = stats['beta']
    sparse_file  = stats.get('sparse_file', '')

    fnam = Path(working_directory).joinpath('iteration_info.h5')

    if fnam.is_file():
        with h5py.File(fnam, 'r') as f:
            N = f['iterations'][()]
    else:
        N = 0

    logger.info(f'saving iteration info to {fnam} for iteration {N}')

    scalar_keys = ['beta', 'Q', 'dQ', 'P_gini', 'orientation_changes', 'class_changes']

    if N == 0:
        with h5py.File(fnam, 'w') as f:
            f['iterations'] = N + 1
            for k in scalar_keys:
                f.create_dataset(k, shape=(1,), maxshape=(None,), dtype=np.float32)
    else:
        with h5py.File(fnam, 'r+') as f:
            for k in scalar_keys:
                f[k].resize(N + 1, axis=0)

    # defaults (first iteration: treat all frames as changed)
    orientation_changes = int(Q_d.shape[0])
    class_changes       = int(Q_d.shape[0])
    dQ                  = 0.

    with h5py.File(fnam, 'r+') as f:
        grp = f.require_group(f'iteration_{N}')

        write_h5(grp, 'occupancy_r',              occupancy_r)
        write_h5(grp, 'P_gini_d',                 P_max_d)
        write_h5(grp, 'Q_d',                      Q_d)
        write_h5(grp, 'most_likely_model_d',      class_max_d)
        write_h5(grp, 'occupancy_dc',             occupancy_dc)
        write_h5(grp, 'most_likely_orientation_d', local_rmax_d)
        write_h5(grp, 'sparse_file',              sparse_file)

        f['iterations'][...] = N + 1
        f['beta'][N]         = beta
        f['Q'][N]            = np.mean(Q_d)
        f['P_gini'][N]       = np.mean(P_max_d)

        if N > 0:
            prev = f[f'iteration_{N-1}']

            dQ = float(np.mean(Q_old_d - prev['Q_d'][()]))
            f['dQ'][N] = dQ

            mlm_prev = prev['most_likely_model_d'][()]
            D = min(mlm_prev.shape[0], class_max_d.shape[0])
            class_changes = int(np.sum(mlm_prev[:D] != class_max_d[:D]))
            f['class_changes'][N] = class_changes

            mlo_prev = prev['most_likely_orientation_d'][()]
            D = min(mlo_prev.shape[0], local_rmax_d.shape[0])
            orientation_changes = int(np.sum(mlo_prev[:D] != local_rmax_d[:D]))
            f['orientation_changes'][N] = orientation_changes
        else:
            f['dQ'][N]                  = dQ
            f['class_changes'][N]       = class_changes
            f['orientation_changes'][N] = orientation_changes

    return {
        **stats,
        'N':                   N,
        'orientation_changes': orientation_changes,
        'class_changes':       class_changes,
        'dQ':                  dQ,
    }


@profiling.timed
def save_output(config):
    """
    Save per-iteration model slices and data info to iteration_info.h5.

    Loads model arrays from each class's model_file, writes model slices,
    then writes frame/DOS data. Optionally computes DOS if 'frame_labels'
    is present in config.
    """
    from . import utils
    from .degree_of_separation import calculate_DOS

    wd  = config['working_directory']
    dq  = config['classes'][0]['model'].dq

    models = []
    for c in config['classes']:
        with h5py.File(c['model_file']) as f:
            models.append(f['data'][()])
    utils.save_model_slices(models, dq, wd)

    frames_d     = config['classes'][0]['data'].frames
    D            = config['classes'][0]['data'].source_shape[0]
    frame_labels = config.get('frame_labels')

    if frame_labels is not None:
        m_d = config['most_likely_model_d']
        A_d, B_d, *_ = frame_labels.values()
        DOS = calculate_DOS(A_d[frames_d], B_d[frames_d], m_d)
    else:
        DOS = None

    utils.save_data_info(frames_d, frame_labels, DOS, D, wd)


def print_iteration_stats(stats):
    """Print a one-line summary of iteration statistics to stdout."""
    beta   = stats['beta']
    Q      = float(np.mean(stats['Q_d']))
    P_gini = float(np.mean(stats['P_max_d']))

    parts = [f"beta={beta:.3f}", f"Q={Q:.2f}", f"P_gini={P_gini:.4f}"]

    if 'dQ' in stats:
        parts.append(f"dQ={stats['dQ']:.2f}")

    if 'orientation_changes' in stats and 'class_changes' in stats:
        D      = len(stats['Q_d'])
        total  = stats['orientation_changes'] + stats['class_changes']
        parts.append(f"changes={total / D:.1%}")

    if 'N' in stats:
        parts.insert(0, f"iter={stats['N']}")

    print("  ".join(parts))
