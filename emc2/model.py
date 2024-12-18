import h5py
import numpy as np
from pathlib import Path

from .utils_cl import to_gpu, to_gpu_2D_image, to_gpu_3D_image
from . import utils


def check_model_file(models_fnam, dimensions, model_length):
    print(f'checking for existing model file {models_fnam}')
    if not Path(models_fnam).is_file() :
        print(f'file not found')
        return False
    
    with h5py.File(models_fnam, 'r') as f:
        
        for c, d in enumerate(dimensions) :
            if f'model_{c}' not in f:
                print(f'file does not have model {c}')
                return False
            
            m = f[f'model_{c}']
            
            if len(m.shape) != d :
                print(f'file does not have models of correct dimesion {len(m.shape)} != {d}')
                return False
            
            if m.shape[0] != model_length :
                print(f'file does not have models of correct shape {m.shape[0]} != {model_length}')
                return False
    
    return True
                
def load_models(Nmodels, models_fnam):
    I = []
    with h5py.File(models_fnam, 'r') as f:
        # https://github.com/mpi4py/mpi4py/issues/177
        w = f['relative_fluence'][()].newbyteorder('=')
        
        for c in range(Nmodels):
            # https://github.com/mpi4py/mpi4py/issues/177
            I.append(f[f'model_{c}'][()].newbyteorder('='))
    
    return I, w

def init_models(dimensions, model_length):
    # so different mpi ranks produce the same results 
    # and so that rerunning produces same results
    I = []
    np.random.seed(1)
    print('intialising models with random numbers')
    for d in dimensions :
        shape = d * (model_length,)
        
        i = np.random.random(shape).astype(np.float32)
        I.append(np.ascontiguousarray(i))
        
    return I, None

def get_models(iteration, models_fnam, dimensions, model_length):
    # load or initialise models
    if iteration == 0 or not check_model_file(models_fnam, dimensions, model_length) :
        print('iteration is zero, initialising models')
        I, w = init_models(dimensions, model_length)
    else :
        I, w = load_models(len(dimensions), models_fnam)
    return I, w

def load_models_cl(dimensions, I, queue, context): 
    I_cl = []
    for c, d in enumerate(dimensions):
        if d == 3 :
            I_cl.append(to_gpu_3D_image(I[c], queue, context))
        
        elif d == 2 :
            I_cl.append(to_gpu_2D_image(I[c], queue, context))
        
        else :
            raise ValueError(f'could parse dimension {d} for class {c}')

    return I_cl

class Models():
    """
    read dimensions list 
    
    dimensions = 2 # or
    dimensions = [2, 3, 2, 2, 3]
    """
    def __init__(self, no_gpu = True, **config):
        self.models_fnam = f"{config['working_directory']}/models.h5"
        self.Nmodels = config['models']
        
        # side length of model dimensions
        self.model_length = config['model_length']
        
        self.dimensions = utils.int_to_list(self.Nmodels, config['dimensions'], 'dimensions')
        
        self.I, self.w = get_models(config['iteration'], self.models_fnam, self.dimensions, self.model_length)
        
        # per pattern relative fluence
        if self.w is None :
            self.w = np.ones((config['frames'],), dtype = float)
        
        # I_cl is a list of 3D or 2D images
        if not no_gpu :
            self.context = config['context']
            self.queue   = config['queue']
            self.I_cl    = self.load_models_cl(self.dimensions, self.I, self.context, self.queue)
        
        # location of q=0 pixel in model
        self.i0    = np.float32(self.model_length//2)
        self.dq    = config['dq']
        self.q_max = config['q_max_model']

