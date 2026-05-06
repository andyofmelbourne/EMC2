"""
should find a way to remove:
/home/amorgan/.cache/pyopencl/pyopencl-compiler-cache-v2-py3.11.9.final.0/lock'-
before every run
"""
import pickle
import emc3
from time import time, sleep
import runpy
from pathlib import Path
import h5py
import shutil
import numpy as np
import sys
import os
import tempfile


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

print(f'{rank=} {size=} {name=}')
sys.stdout.flush()

from emc3.run import calculate_logR, update_models


def _assert_logR(config, cids):
    for ci in cids:
        c = config['classes'][ci]
        if not c['update_logR']:
            continue
        with h5py.File(c['logR_file']) as f:
            logR = f['logR_dr'][()]
        assert np.all(np.isfinite(logR)), \
            f"class {ci}: logR_dr has non-finite values " \
            f"(nan={np.isnan(logR).sum()}, inf={np.isinf(logR).sum()})"
        print(f"  class {ci}: logR OK  shape={logR.shape}  "
              f"range=[{logR.min():.3f}, {logR.max():.3f}]")


def _assert_probability(config):
    """
    Each class file holds only that class's orientation slice.
    The combined P across all classes must sum to 1 per frame.
    """
    D = config['classes'][0]['P_data'].shape[0]

    P_all = np.zeros((D, config['R']), dtype=float)
    for c in config['classes']:
        if not c['update_probability']:
            continue
        r0 = c['r_offset']
        R_c = c['mapper'].shape[1]
        with h5py.File(c['probability_matrix_file']) as f:
            P_c = f['P_dr'][()]
        assert np.all(P_c >= 0), \
            f"class {c['class_id']}: P_dr has negative values"
        P_all[:, r0:r0 + R_c] = P_c

    sums = P_all.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), \
        f"combined P_dr rows don't sum to 1 " \
        f"(min={sums.min():.6f}, max={sums.max():.6f})"
    print(f"  P_dr OK  combined shape=({D}, {config['R']})  "
          f"sums=[{sums.min():.6f}, {sums.max():.6f}]")


def _assert_model(config, cids):
    for ci in cids:
        c = config['classes'][ci]
        if not c['update_model']:
            continue
        with h5py.File(c['model_file']) as f:
            model = f['data'][()]
        assert np.all(np.isfinite(model)), \
            f"class {ci}: model has non-finite values"
        assert np.all(model >= 0), \
            f"class {ci}: model has negative values"
        print(f"  class {ci}: model OK  shape={model.shape}  "
              f"range=[{model.min():.4f}, {model.max():.4f}]")


restart = True
refresh_config = False

config_file = 'config.pickle'
config_script = 'config.py'
profile_dir = Path('profile')
iters = 3
beta_start = 0.01
# beta_start = 1.00
beta_stop = 1.0


def backup(config, n=1, restore=False):
    if rank == 0:
        fnam = Path(config['fluence_file'])
        fnam2 = fnam.stem + f'_bak{n}.h5'
        if restore:
            shutil.copy(str(fnam2), str(fnam))
        else:
            shutil.copy(str(fnam), str(fnam2))

        for c in config['classes']:
            fnam = Path(c['model_file'])
            fnam2 = fnam.stem + f'_bak{n}.h5'
            if restore:
                shutil.copy(str(fnam2), str(fnam))
            else:
                shutil.copy(str(fnam), str(fnam2))

    comm.Barrier()

if rank == 0 and restart:
    # remove iteration info file
    Path('iteration_info.h5').unlink(missing_ok=True)

    # remove chunked data
    for fnam in Path('./').glob('*chunk*.h5'):
        Path(fnam).unlink(missing_ok=True)

    # remove temporary data files
    for fnam in Path('./').glob('data_*.h5'):
        Path(fnam).unlink(missing_ok=True)

    # remove temporary data files
    for fnam in Path('./').glob('back_*.h5'):
        Path(fnam).unlink(missing_ok=True)

comm.Barrier()

# ensure that only one rank makes the pickle
# file to prevent multiple ranks saving data
if rank == 0 and (restart or refresh_config):
    # make pickle file
    config = runpy.run_path(config_script)['make_pickle_file'](
        output_file='config.pickle')

comm.Barrier()

emc3.profiling.setup(profile_dir)
config = pickle.load(open(config_file, 'rb'))

Nframes = config['classes'][0]['data'].shape[0]

if restart and rank == 0:
    # initialise model files
    for ci, c in enumerate(config['classes']):
        c['model'].init_random()

        with h5py.File(c['model_file'], 'w') as f:
            f['data'] = c['model'].data
            f['dq'] = c['model'].dq

    # init fluence
    with h5py.File(config['fluence_file'], 'w') as f:
        f['w_d'] = np.ones(Nframes, dtype=float)

comm.Barrier()

cids = list(range(len(config['classes'])))
my_classes = cids[rank::size]

beta = beta_start
last_change = -100
iter_stats = None  # stats returned by calculate_P + save_iteration_info

for i in range(iters):
    # beta scheduling from previous iteration's in-memory stats.
    # Rank 0 holds iter_stats; broadcast just the two change counts.
    n_changes = None
    if rank == 0 and iter_stats is not None:
        n_changes = iter_stats['orientation_changes'] + iter_stats['class_changes']
    n_changes = comm.bcast(n_changes, root=0)

    if n_changes is not None:
        change = n_changes / Nframes
        print(f'{rank=} {change=} {last_change=} {i=}')
        sys.stdout.flush()

        """
        if change < 0.05 and last_change != (i - 1) and beta != beta_stop:
            beta *= 2
            last_change = i

        if beta == beta_stop and change < 0.05 and last_change != (i - 1):
            break
        """

    beta *= 2
    beta = min(beta, beta_stop)

    t0 = time()
    calculate_logR(config_file, config, p_per_device=2, cids=my_classes)
    time_logR = time() - t0

    _assert_logR(config, my_classes)
    comm.Barrier()

    if rank == 0:
        t0 = time()
        stats = emc3.probability.calculate_P(config, beta)
        time_prob = time() - t0
        _assert_probability(config)
        iter_stats = emc3.input_output.save_iteration_info(
            stats, config['working_directory']
        )
        emc3.input_output.print_iteration_stats(iter_stats)

    comm.Barrier()

    t0 = time()
    update_models(config_file, config, p_per_device=2, cids=my_classes)
    time_I = time() - t0

    _assert_model(config, my_classes)

    comm.Barrier()

    if rank == 0:
        emc3.input_output.save_output(config)
        print(f'{time_I=}')
        print(f'{time_prob=}')
        print(f'{time_logR=}')

    comm.Barrier()
