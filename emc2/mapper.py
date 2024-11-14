import numpy as np
from pathlib import Path

from .utils_cl import to_gpu, to_gpu_2D_image_stack, to_gpu_3D_image
from .orientations import get_rotations_3D, get_rotations_2D

def Mapper():
    def __init__(self, opencl_stuff, **config):
        self.mask, q, dq, frames = calculate_mask(**config)
        
        self.qx_cl = to_gpu(q[0], **opencl_stuff)
        self.qy_cl = to_gpu(q[1], **opencl_stuff)
        
        self.I  = get_models(**config)
        
        # if dimension = 2 then load image stack
        if config['dimension'] == 2 :
            self.I_cl = to_gpu_2D_image_stack(self.I, **opencl_stuff)
            self._calculate_tomograms = calculate_tomograms_2D
        
        # if dimension = 3 then load a list of 3D images
        elif config['dimension'] == 3 :
            self.I_cl = []
            for model in range(self.I.shape[0]):
                self.I_cl.append(to_gpu_3D_image(self.I[model], **opencl_stuff))
            self._calculate_tomograms = calculate_tomograms_3D
        
        # this has repeated entries for 
        self.R_cl = get_rotation_matrices(opencl_stuff, rotation_order = 10, dimension = 3)
        
        # number of pixels in mask
        self.I = np.sum(self.mask)
        
        # number of models
        self.models       = config['models']
        self.model_pixels = config['dimension'] * config['model_length']
        
        # number of orientations
        self.rotations = self.R_cl.shape[0]
        
        # number of frames
        self.rotations = self.R_cl.shape[0]
    
    def calculate_tomograms(self, r0, r1, i0, i1):
        models    = np.arange(r0, r1, 1) // self.model_pixels
        rotations = np.arange(r0, r1, 1) %  self.model_pixels
        pixels    = np.arange(r0, r1, 1) %  self.model_pixels
        
        W_cl = self._calculate_tomograms(
                    models, 
                    rotations, 
                    pixels, 
                    self.R_cl, 
                    self.I_cl, 
                    self.qx_cl, 
                    self.qy_cl) 
        
        return W_cl
    
    def calculate_frames(self, W, d0, d1, r0, r1, i0, i1):
        F = self.C[i0: i1] * W
        return F
        
    def logR(self, frames, d0, d1, r0, r1, i0, i1):
        # model to tomograms W_ri
        W_cl = self.calculate_tomograms(r0, r1, i0, i1) 
        
        # tomograms to frame models 
        F = self.calculate_frames(W, d0, d1, r0, r1, i0, i1)
        
        # calculate pixel-wise log-likelihood
        logR = frames * np.log(F) - F
        return logR

        

        


def get_rotation_matrices(opencl_stuff, rotation_order = 10, dimension = 3):
    if dimension == 3 :
        R_cl = get_rotations_3D(rotation_order, opencl_stuff)
    
    elif dimension == 2 :
        R_cl = get_rotations_2D(rotation_order, opencl_stuff)
    
    else :
        raise ValueError(f'dimension {dimension} not supported')
    return R_cl


