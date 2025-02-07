"""
calculate wsums_r (\sum_i C_i W_ri) for a single class
"""
import argparse
from pathlib import Path
import h5py
import time
import numpy as np

from context import emc2
from emc2 import classes
from emc2 import tomograms_new
from emc2 import utils_cl
from emc2 import utils
from emc2 import probability_new


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=r"""
        calculate wsums_r (\sum_i C_i W_ri) for a single class
        """
    )
    parser.add_argument('class_file', type=str, help='file name class file')
    parser.add_argument(
        '--rot_chunk',
        type=int,
        default=0,
        help='calculate for a subset of rotations'
    )
    parser.add_argument(
        '--rot_chunks',
        type=int,
        default=1,
        help='number of blocks to split rotations over'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file)

    working_directory = Path(args.class_file).parent

    # load opencl
    print('\nloading opencl context and devices:')
    opencl_stuff = utils_cl.opencl_init(device_no=args.rot_chunk)

    # initialise tomograms
    mapper = tomograms_new.Mapper(
        class_c.model.ndim,
        class_c.xyz,
        class_c.mapping_matrix,
        opencl_stuff['context'],
        opencl_stuff['queue'],
        interpolation=class_c.interpolation_forward
    )

    W_ri = tomograms_new.Tomograms(mapper, class_c.model)

    R = W_ri.shape[0]

    if args.rot_chunk is not None:
        if not args.rot_chunks:
            raise ValueError('must specify total number of rot_chunks')
        r0, r1, dr = utils.chunker_mpi(args.rot_chunks, R)
        r0 = r0[args.rot_chunk]
        r1 = r1[args.rot_chunk]
        dr = dr[args.rot_chunk]
    else:
        r0, r1, dr = 0, R, R

    wsums_r = probability_new.calculate_wsums_r(class_c.C, W_ri, r0, r1)

    # save
    def write():
        try:
            with h5py.File(args.class_file, 'r+') as f:
                f['wsums'][r0: r1] = wsums_r

        except OSError:
            # try again
            print('waiting to try writing to file again')
            time.sleep(np.random.random())
            write()

    write()

    print('done')
