from pathlib import Path
import runpy
import h5py
import logging

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


