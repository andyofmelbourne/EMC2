import h5py
import numpy as np
from pathlib import Path

from .utils_cl import to_gpu, to_gpu_2D_image_stack, to_gpu_3D_image
from . import utils


def get_models(iteration = 0, models = 1, dimensions = 3, model_length = 64, **config):
    model_shape = (models,) + dimensions * (model_length,)
    
    # load or initialise models
    if iteration > 0 :
        print('*************', iteration)
        with h5py.File('models.h5') as f:
            if f['data'].shape == model_shape :
                I = f['data'][()]
    else :
        # this is to make sure that different processors have the same model
        np.random.seed(1)
        I = np.random.random(model_shape)
        
        # save models 
        utils.save_models(I, **config)
    
    return np.ascontiguousarray(I.astype(np.float32))

class Models():
    def __init__(self, **config):
        self.I  = get_models(**config)
        
        # location of q=0 pixel in model
        self.i0 = np.float32(self.I.shape[-1]//2)
        self.dq = config['dq']
        
        # if dimension = 2 then load image stack
        if config['dimensions'] == 2 :
            pointer, read_only_flag = self.I.__array_interface__['data']
            self.I_cl = to_gpu_2D_image_stack(self.I, context = config['context'], queue = config['queue'])
        
        # if dimension = 3 then load a list of 3D images
        elif config['dimensions'] == 3 :
            self.I_cl = []
            for model in range(self.I.shape[0]):
                self.I_cl.append(to_gpu_3D_image(self.I[model], context = config['context'], queue = config['queue']))
