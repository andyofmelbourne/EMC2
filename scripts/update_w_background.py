import argparse
from pathlib import Path
import sys
import h5py
import pickle
import numpy as np

from context import emc2
from emc2 import classes
from emc2 import data_getter
from emc2 import tomograms
from emc2 import utils_cl
from emc2 import utils
from emc2 import get_script_logger
from emc2 import input_output
from emc2 import model_update_background_sparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=utils.MyFormatter,
        description="""\
        update relative fluence with background over data chunks"""
    )

    parser.add_argument(
        'config',
        type=str,
        help='config file name'
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


if __name__ == '__main__':
    args = get_args()

    working_directory = Path(args.config).parent

    logger = get_script_logger.get_logger(
        working_directory=working_directory
    )

    # load config file
    config = input_output.load_config(args.config)

    models = len(config['classes'])

    class_files = [
        Path(working_directory).joinpath(f'class_{c}.h5')
        for c in range(models)
    ]

    mpi_split_frames = (args.data_chunk, args.data_chunks)

    # load class file (assume masking is the same for all classes)
    class_0 = classes.Class()
    class_0.load(class_files[0])

    # testing
    # class_0.frame_model = 'background'

    # load data
    logger.info(f'loading frames with frame_model={class_0.frame_model}')
    K_di = data_getter.Data_getter(
        mask=class_0.P_mask,
        split_frames=class_0.split_frames,
        cxi_file=class_0.cxi_file,
        filter=class_0.frame_selection,
        working_directory=working_directory,
        mpi_split_frames=mpi_split_frames,
        frame_model=class_0.frame_model
    )

    B_di = data_getter.Data_getter_background(K_di)

    # load opencl
    opencl_stuff = utils_cl.opencl_init(device_no=args.data_chunk)

    d0 = K_di.d_start_mpi[args.data_chunk]
    d1 = K_di.d_stop_mpi[args.data_chunk]
    dd = d1-d0

    # initialise all tomograms
    Wsums_cr = []
    W_cri = []
    P_cdr = []
    for class_file in class_files:
        class_c = classes.Class()
        class_c.load(class_file, skip=['probability_matrix'])

        mapper = tomograms.Mapper(
            class_c.model.ndim,
            class_c.model.shape[0],
            class_c.P_xyz,
            class_c.mapping_matrix,
            opencl_stuff['context'],
            opencl_stuff['queue'],
            interpolation=class_c.interpolation_forward
        )

        W_ri = tomograms.Tomograms(mapper, class_c.model)

        W_cri.append(W_ri)
        Wsums_cr.append(class_c.wsums)

        with h5py.File(class_file) as f:
            P_cdr.append(f['probability_matrix'][d0:d1])

        C_i = class_c.P_C

    w0_d = class_c.relative_fluence
    w_d = model_update_background_sparse.w_update(
        P_cdr, K_di, B_di, W_cri, Wsums_cr, C_i)

    rms = np.mean((w0_d[d0:d1] - w_d)**2)**0.5
    logger.info(f'rms difference for w_d {d0}-{d1}: {rms}')
    logger.info(f'{np.mean(w_d)=}')

    # pipe to std out
    file = args.output
    # import numpy as np
    # pickle.dump(np.arange(10), file)
    for class_file in class_files:
        logger.info(f'writing w_d chunk {d0}-{d1} to {class_file}')
        msg = {
            'file': class_file,
            'mode': 'r+',
            'relative_fluence': {
                'slice': slice(d0, d1),
                'data': w_d
            }
        }
        pickle.dump(msg, file)
