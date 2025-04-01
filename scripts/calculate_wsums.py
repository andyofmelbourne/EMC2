r"""
calculate wsums_r (\sum_i C_i W_ri) for a single class
"""
import argparse
from pathlib import Path
import h5py
import time
import numpy as np

from context import emc2
from emc2 import classes
from emc2 import tomograms
from emc2 import utils_cl
from emc2 import utils
from emc2 import probability

from emc2 import get_script_logger


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

    parser.add_argument(
        '--P_mask',
        action='store_true',
        help='calculate tomogram sums within P_mask'
    )

    parser.add_argument(
        '--mask',
        action='store_true',
        help='calculate tomogram sums within mask'
    )
    args = parser.parse_args()
    return args


def main(
    class_c,
    xyz,
    C,
    opencl_stuff,
    class_file,
    rot_chunk=None,
    rot_chunks=None,
    out='wsums'
):
    # initialise tomograms
    mapper = tomograms.Mapper(
        class_c.model.ndim,
        class_c.model.shape[0],
        xyz,
        class_c.mapping_matrix,
        opencl_stuff['context'],
        opencl_stuff['queue'],
        interpolation=class_c.interpolation_forward
    )

    W_ri = tomograms.Tomograms(mapper, class_c.model)

    R = W_ri.shape[0]

    if rot_chunk is not None:
        if not rot_chunks:
            raise ValueError('must specify total number of rot_chunks')
        r0, r1, dr = utils.chunker_mpi(rot_chunks, R)
        r0 = r0[rot_chunk]
        r1 = r1[rot_chunk]
        dr = dr[rot_chunk]
    else:
        r0, r1, dr = 0, R, R

    logger.info('calculating tomogram sums')
    wsums_r = probability.calculate_wsums_r(C, W_ri, r0, r1)
    logger.info('finished calculating tomogram sums')

    # save
    def write():
        try:
            with h5py.File(class_file, 'r+') as f:
                f[out][r0: r1] = wsums_r

        except OSError:
            logger.debug('waiting to try writing to file again')
            time.sleep(np.random.random())
            write()

    write()


if __name__ == "__main__":
    args = get_args()

    working_directory = Path(args.class_file).parent

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )
    logger.info('calculate_wsums (start)')

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file, skip=['probability_matrix'])

    # load opencl
    opencl_stuff = utils_cl.opencl_init(device_no=class_c.class_id)

    if args.P_mask:
        xyz = class_c.P_xyz
        C = class_c.P_C
        out = 'P_wsums'

        main(
            class_c,
            class_c.P_xyz,
            class_c.P_C,
            opencl_stuff,
            args.class_file,
            rot_chunk=args.rot_chunk,
            rot_chunks=args.rot_chunks,
            out=out
        )

    if args.mask:
        xyz = class_c.xyz
        C = class_c.C
        out = 'wsums'

        main(
            class_c,
            class_c.xyz,
            class_c.C,
            opencl_stuff,
            args.class_file,
            rot_chunk=args.rot_chunk,
            rot_chunks=args.rot_chunks,
            out=out
        )

    logger.info('calculate_wsums (stop)')
