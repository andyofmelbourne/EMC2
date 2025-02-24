from pathlib import Path
import runpy
import h5py
import sys

def get_option(d, thing):
    if thing in d :
        return d[thing]
    else :
        return False

def load_config(path):
    print(f'\nloading configuration file from {path}', file=sys.stderr)
    p = Path(path)
    
    # returns a dict
    config = runpy.run_path(str(p.absolute()))
    
    return config

def set_working_directory(path):
    wd = str(Path(path).resolve().parent)
    print(f'working directory: {wd}', file=sys.stderr)
    return {'working_directory': wd}

def set_iteration_info_fnam(working_directory):
    fnam = Path.joinpath(Path(working_directory), 'iteration_info.h5')
    return {'iteration_info': str(fnam)}

def get_iterations(**config):
    fnam = os.path.join(config['working_directory'], 'iteration_info.h5')
    if os.path.exists(fnam):
        with h5py.File(fnam) as f:
            iterations = f['iterations'][()]
    else :
        iterations = 0
    return iterations

def get_iteration_number(config):
    if get_option(config, 'restart') :
        iteration = 0
        print(
            f'"restart" is True setting iteration to {iteration}',
            file=sys.stderr
        )
    else :
        # check iteration info file
        fnam = Path(config['iteration_info'])
        if fnam.is_file() :
            with h5py.File(fnam, 'r') as f:
                iteration = f['iterations'][()]
            print(
                f'Getting iteration number from {fnam}. \
                Setting iteration to {iteration}',
                file=sys.stderr
            )
        else :
            iteration = 0
            print(f'Setting iteration to {iteration}', file=sys.stderr)
    
    return iteration
        
        

        
