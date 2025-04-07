import argparse
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm

from context import emc2
from emc2 import utils
from emc2 import classes
from emc2 import get_script_logger


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""Update model intensities"""
    )

    parser.add_argument(
        'class_file',
        type=str,
        help='file name class file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help="h5 file to write 'model' dataset to. \
        Default is input class file."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    working_directory = Path(args.class_file).parent
    cachedir = working_directory.joinpath('cachdir')

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('update_I_merge (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(
        args.class_file,
        skip=['mapping_matrix', 'probability_matrix']
    )

    N = class_c.model.size

    # reduce c_n files
    c_n = None

    search_string = f'class_{class_c.class_id}_c_n_*.h5'
    fnams = list(cachedir.glob(search_string))

    if len(fnams) == 0:
        raise ValueError(
            f'Error no files matching the pattern {search_string} '
            f'where found in the directory: {cachedir}'
        )

    for fnam in fnams:
        with h5py.File(fnam, 'r') as f:
            if c_n is None:
                c_n = f['c_n'][()]
            else:
                c_n += f['c_n'][()]

    # fnam = f'class_{class_c.class_id}_c_n.h5'
    # with h5py.File(fnam, 'w') as f:
    #     f['c_n'] = c_n

    # merge a, b, n
    search_string = f'class_{class_c.class_id}_abn_*.h5'
    fnams = list(cachedir.glob(search_string))
    if len(fnams) == 0:
        raise ValueError(
            f'Error no files matching the pattern {search_string} '
            f'where found in the directory: {cachedir}'
        )

    # write datasets one at a time to speed up io
    fnam_out = f'class_{class_c.class_id}_abn.h5'
    fnam_out = cachedir.joinpath(fnam_out)

    n_inds = []
    n_counts = []
    for fnam in fnams:
        with h5py.File(fnam, 'r') as g:
            n_inds.append(g['n_indices'][()])
            n_counts.append(g['n_counts'][()])

    N_buffer = np.sum([n[()] for n in n_counts])
    with h5py.File(fnam_out, 'w') as f:
        f['buffer_size'] = N_buffer

    print(f'{N_buffer=}')

    n_indices = np.zeros((N+1,), dtype=int)
    for dset in ['a_dsri', 'b_dsri']:
        t_out_n = np.empty((N_buffer,), dtype=np.float32)

        t_s = []
        for fnam in fnams:
            with h5py.File(fnam, 'r') as g:
                t_s.append(g[dset][()])

        index = 0
        for n in tqdm(range(N), desc=f'writing & merging {dset}'):
            n_indices[n] = index
            M = 0
            M2 = 0
            for k in range(len(fnams)):
                i, j = n_inds[k][n: n+2]
                M = j-i
                t_out_n[index: index+M] = t_s[k][i:j]
                index += M
                M2 += M

        n_indices[N] = index
        with h5py.File(fnam_out, 'r+') as f:
            f[dset] = t_out_n

    with h5py.File(fnam_out, 'r+') as f:
        f['n_indices'] = n_indices
        f['c_n'] = c_n

    logger.info('update_I_merge (stop)')
