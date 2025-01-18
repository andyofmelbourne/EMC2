import h5py
import numpy as np

def geometry(**config):
    """
    calculate: 
        - mask  : pixel mask based on config 
        - q     : Ewald sphere q-values in reference orientation
        - dq    : q-space voxel size of model
        - q_max : 1 / full period resolution limit within pixel mask
        - C     : the solid angle and polarisation correction factor
    """
    
    # calculate q-values
    with h5py.File(config['cxi_file']) as f:
        mask = f['entry_1/instrument_1/detector_1/good_pixels'][()] 
        
        #frames = f['entry_1/data_1/data'].shape[0]
         
        # pixel map 
        xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()] 
        wav  = np.mean(f['/entry_1/instrument_1/source_1/photon_wavelength'][()]) 
        dx   = f['entry_1/instrument_1/detector_1/x_pixel_size'][()] 
        dy   = f['entry_1/instrument_1/detector_1/y_pixel_size'][()] 
        
        # get pixel area for non-square pixels (DSSC)
        key = 'entry_1/instrument_1/detector_1/pixel_area'
        if key in f :
            pixel_area = f[key][()]
        else :
            pixel_area = dx * dy

    k = 'xy_offset'
    if k in config and config[k] :
        xyz[0] += config[k][0]
        xyz[1] += config[k][1]
        print(f'apply xy offset to cxi geometry of {config[k]} m')
        
    
    # calculate pixel radius
    r = np.sum(xyz**2, axis=0)**0.5
    q = xyz.copy() / r
    q[2] -= 1
    q /= wav
    qr = np.sum(q**2, axis=0)**0.5
    
    if config['polarisation'] == 'x' :
        P = 1 - (xyz[0] / r)**2
    elif config['polarisation'] == 'y' :
        P = 1 - (xyz[1] / r)**2
    elif config['polarisation'] == 'None' :
        P = np.ones(dshape[1:])
    
    # solid angle correction  
    Omega = pixel_area * xyz[2] / r**3
    
    # merged intensity to frame correction factor
    C = Omega * P
    
    # scale 
    C /= C[mask].max()

    M = config['model_length']
    k = 'zero_padding' 
    if k in config and config[k] :
        M = M - 2 * config[k]
    
    if 'q_max' in config and config['q_max'] :
        q_max = config['q_max']
    
    elif 'res_max' in config and config['res_max'] :
        q_max = 1 / config['res_max']
    
    elif 'pixel_radius' in config and config['pixel_radius'] :
        rp    = config['pixel_radius']
        z     = xyz[2].ravel()[0]
        r     = (rp**2 + z**2)**0.5
        q_max = (rp**2 + (z-r)**2)**0.5 / wav / r
    
    elif config['pixels_per_voxel'] and config['model_length'] :
        rp    = dx * config['pixels_per_voxel'] * (M // 2)
        z     = xyz[2].ravel()[0]
        r     = (rp**2 + z**2)**0.5
        q_max = (rp**2 + (z-r)**2)**0.5 / wav / r
    
    else :
        raise ValueError('not enough information to calculate q-mask')
    
    mask[qr > q_max] = False
    
    # calculate model q-space voxel size 
    # such that the zero pixel (i0) satisfies:
    #   np.fft.fftshift(np.fft.fftfreq(N))[i0] = 0
    if config['model_length'] :
        #M = config['model_length']
        
        if config['pixels_per_voxel'] :
            if (config['model_length'] % 2) == 0 :
                dq = q_max / (M / 2 - 1)
            else :
                dq = 2 * q_max / (M - 1)

        # increase qmax for model if required
        k = 'zero_padding' 
        if k in config and config[k] :
            q_max_model = q_max + 2 * config[k] * dq
        else :
            q_max_model = q_max 

        print(f'{q_max=} {q_max_model=}')

    else :
        raise ValueError('need "model_length" to define model voxel size')
        
    # location of zero pixel in models along each axis
    i0 = np.float32(config['model_length']//2)
    
    out = {\
    'mask'       : mask,
    'C'          : np.ascontiguousarray(C[mask].astype(np.float32)),
    'q'          : np.ascontiguousarray(q[:, mask].astype(np.float32)),
    'xyz'        : np.ascontiguousarray(xyz[:, mask].astype(np.float32)),
    'dq'         : np.float32(dq),
    'i0'         : i0,
    'wavelength' : wav,
    'q_max'      : q_max,
    'q_max_model': q_max_model}
    return out
