import argparse
import h5py
import numpy as np

from context import emc2
from emc2 import input_output
from emc2 import get_script_logger
from emc2 import classes
from emc2 import init


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="check if the configuration file has been updated and"
                    "update class file accordingly"
    )

    parser.add_argument(
        'config_file',
        type=str,
        help='configuration file name'
    )

    parser.add_argument(
        'class_file',
        type=str,
        help='class file name'
    )

    parser.add_argument(
        '--prob',
        action='store_true',
        help='only update datasets relevant to probability calc. '
             'usefull when updating model size'
    )

    parser.add_argument(
        '--skip_data',
        action='store_true',
        help='skip checking data cache'
    )

    parser.add_argument(
        '--data',
        action='store_true',
        help='only checking data cache'
    )

    args = parser.parse_args()
    return args


changed = {
    'cxi_file': None,
    'split_frames': None,
    'pixels_per_voxel': None,
    'dimensions': None,
    'model_length': None,
    'P_mask_padding': None,
    'symmetry': None,
    'interpolation_forward': None,
    'xyz_offset': None,
    'scale': None,
    'zero_padding': None,
    'rotation_order': None,
    'likelihood': None,
    'frame_model': None,
    'maximise': None,
    'update_fluence': None,
    'update_model': None,
    'update_probability': None,
    'polarisation': None,
    'P_thresh': None,
    'frame_selection': None
}


def compare(a, b):
    if isinstance(a, str):
        same = a == b
    elif hasattr(a, '__len__'):
        if len(a) != len(b):
            same = False
        else:
            same = np.allclose(np.array(a), np.array(b))
    else:
        same = a == b
    return same


if __name__ == '__main__':
    args = get_args()

    config = {}

    # set working directory as the directory in which config.py resides
    config.update(
        input_output.set_working_directory(args.config_file)
    )

    logger = get_script_logger.get_logger(
        working_directory=config['working_directory']
    )
    logger.info('check config (start)')

    # load config file
    logger.info('load config (start)')
    config.update(
        input_output.load_config(args.config_file)
    )
    logger.info('load config (stop)')

    with h5py.File(args.class_file) as f:
        cid = f['class_id'][()]

    class_config = config['classes'][cid]

    # add global variables to class dict
    class_config['cxi_file'] = config['cxi_file']
    class_config['split_frames'] = config['split_frames']
    class_config['working_directory'] = config['working_directory']

    # add frame selection
    if not args.skip_data:
        with h5py.File(config['cxi_file']) as g:
            frame_selection = np.where(config['filter'](g))[0]
            class_config['frames'] = len(frame_selection)
            class_config['filter'] = frame_selection
            class_config['frame_selection'] = frame_selection

    # load class file
    class_file = classes.Class()
    class_file.load(args.class_file, skip=['probability_matrix', 'model'])

    if args.skip_data:
        class_config['frame_selection'] = class_file['frame_selection']
        class_config['filter'] = class_file['frame_selection']
        class_config['frames'] = len(class_file['frame_selection'])

    if args.prob:
        class_config['model_length'] = class_file['model_length']
        class_config['pixels_per_voxel'] = class_file['pixels_per_voxel']
        class_config['dimensions'] = class_file['dimensions']

    # see what's changed
    for key in changed.keys():
        same = compare(class_file[key], class_config[key])
        changed[key] = not same

    # for k, v in changed.items():
    #     print(k, v)

    update_geometry_keys = [
        'pixels_per_voxel',
        'model_length',
        'P_mask_padding',
        'xyz_offset',
        'zero_padding',
        'polarisation'
    ]

    update_mapping_keys = [
        'pixels_per_voxel',
        'model_length',
        'symmetry',
        'xyz_offset',
        'rotation_order',
        'dimensions',
        'scale'
    ]

    update_model_keys = [
        'model_length',
        'dimensions'
    ]

    update_data_keys = [
        'frame_selection',
        'frame_model'
    ]

    update_param_keys = [
        'P_thresh',
        'update_model',
        'update_fluence',
        'update_probability',
        'maximise',
        'frame_model',
        'likelihood',
        'interpolation_forward'
    ]

    update_geometry = any([changed[key] for key in update_geometry_keys])

    update_mapping = (
        any([changed[key] for key in update_mapping_keys])
        and not args.data
    )

    update_model = (
        any([changed[key] for key in update_model_keys])
        and not args.prob
        and not args.data
    )

    update_data = (
        any([changed[key] for key in update_data_keys])
        or update_geometry
    )

    update_fluence = (
        changed['frame_selection']
        and not args.data
    )

    update_params = (
        any([changed[key] for key in update_param_keys])
        and not args.data
    )

    if args.skip_data:
        update_data = False

    # test
    update_geometry = True
    update_model = True
    update_mapping = True

    update = any([update_geometry, update_model, update_data, update_mapping])

    logger.debug(f'{args.prob=}')
    logger.debug(f'{update_geometry=}')
    logger.debug(f'{update_mapping=}')
    logger.debug(f'{update_model=}')
    logger.debug(f'{update_data=}')
    logger.debug(f'{update=}')

    if update_params:
        with h5py.File(args.class_file, 'r+') as f:
            for key in update_param_keys:
                f[key][...] = class_config[key]

    if update:
        for key in class_file:
            if key not in class_config:
                class_config[key] = class_file[key]

        if update_geometry:
            # print('\nupdating geometry:')
            init.add_geometry(class_config)

        if update_model:
            init.init_model(class_config)

        if update_fluence:
            init.init_fluence(class_config)

        if update_mapping:
            # print('\nupdating mapping:')
            from emc2 import utils_cl
            class_config.update(utils_cl.opencl_init())
            init.init_mapping(class_config, {})

        if update_data:
            print(f"updating data with {len(class_config['filter'])} "
                  f"frames")
            init.check_data(class_config)

        if not args.data:
            with h5py.File(args.class_file) as f:
                D = class_config['frames']
                R = class_config['mapping_matrix'].shape[1]

                shape0 = f['probability_matrix'].shape
                shape1 = (D, R)

                if shape0 != shape1:
                    # initialise probability matrix
                    class_config['probability_matrix'] = np.empty(
                        (D, R),
                        dtype=np.float64
                    )

                    class_config['wsums'] = np.zeros((R,), dtype=float)
                    class_config['P_wsums'] = np.zeros((R,), dtype=float)

            # save class_file
            init.save_classes(class_config, overwrite=False, check=False)

    logger.info('check config (stop)')
