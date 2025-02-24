"""
calculate logR for a single class
"""
import argparse
from pathlib import Path
import sys
import pickle
import numpy as np

from context import emc2
from emc2 import classes
from emc2 import data_getter
from emc2 import tomograms_new
from emc2 import utils_cl
from emc2 import probability_new
from emc2 import probability_background
from emc2 import utils
from emc2 import get_script_logger


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""calculate logR"""
    )

    parser.add_argument(
        'class_file',
        type=str,
        help='file name class file'
    )

    parser.add_argument(
        '--data_chunk',
        type=int,
        default=0,
        help='calculate for a subset of frames'
    )

    parser.add_argument(
        '--data_chunks',
        type=int,
        default=1,
        help='number of blocks to split frames over'
    )

    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('wb'),
        default=sys.stdout.buffer,
        help="Python pickle output file. \
            The result is written as a dictionary"
    )

    args = parser.parse_args()
    return args


def main(
    K_di, W_ri, C_i, w_d, wsums_r, likelihood, frame_model, P_thresh
):
    logR_dr, wsums_r = probability_new.calc_logR(
        K_di,
        W_ri,
        C_i,
        w_d,
        wsums_r,
        likelihood=likelihood,
        frame_model=frame_model,
        P_thresh=P_thresh
    )
    return logR_dr


def main_background(
    model, K_di, C_i, w_d, wsums_r, M_srn, xyz_i,
    context, queue, interpolation_forward,
    likelihood
):
    B_di = data_getter.Data_getter_background(K_di)

    F_dri = tomograms_new.Frames(
        context, queue, K_di, B_di, w_d,
        model, M_srn, xyz_i, C_i
    )

    logR_dr = probability_background.calc_logR(
        wsums_r,
        F_dri,
        likelihood=likelihood
    )

    return logR_dr


if __name__ == "__main__":
    args = get_args()

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file)

    working_directory = Path(args.class_file).parent

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )

    mpi_split_frames = (args.data_chunk, args.data_chunks)

    # load data
    logger.info(f'loading frames with frame_model={class_c.frame_model}')
    K_di = data_getter.Data_getter(
        mask=class_c.P_mask,
        split_frames=class_c.split_frames,
        cxi_file=class_c.cxi_file,
        filter=class_c.frame_selection,
        working_directory=working_directory,
        mpi_split_frames=mpi_split_frames,
        frame_model=class_c.frame_model
    )

    # load opencl
    print('\nloading opencl context and devices:', file=sys.stderr)
    opencl_stuff = utils_cl.opencl_init(device_no=args.data_chunk)

    if class_c.frame_model == 'background':
        logR_dr = main_background(
            class_c.model,
            K_di,
            class_c.P_C,
            class_c.relative_fluence,
            class_c.wsums,
            class_c.mapping_matrix,
            class_c.P_xyz,
            opencl_stuff['context'],
            opencl_stuff['queue'],
            class_c.interpolation_forward,
            class_c.likelihood
        )
    else:
        # initialise tomograms
        mapper = tomograms_new.Mapper(
            class_c.model.ndim,
            class_c.model.shape[0],
            class_c.P_xyz,
            class_c.mapping_matrix,
            opencl_stuff['context'],
            opencl_stuff['queue'],
            interpolation=class_c.interpolation_forward
        )

        W_ri = tomograms_new.Tomograms(mapper, class_c.model)

        logR_dr = main(
            K_di,
            W_ri,
            class_c.P_C,
            class_c.relative_fluence,
            class_c.wsums,
            class_c.likelihood,
            class_c.frame_model,
            class_c.P_thresh
        )

    assert (np.all(np.isfinite(logR_dr)))

    # pipe to std out
    file = sys.stdout.buffer
    d0 = K_di.d_start_mpi[args.data_chunk]
    d1 = K_di.d_stop_mpi[args.data_chunk]
    logger.info(f'writing logR chunk {d0}-{d1} to {args.class_file}')
    msg = {
        'file': args.class_file,
        'mode': 'r+',
        'probability_matrix': {
            'slice': slice(d0, d1),
            'data': logR_dr
        }
    }
    pickle.dump(msg, file)

    print('done', file=sys.stderr)
