# cache results using photon sparse format in pickle files
# all dimension in dataset except the first are ravelled (the pixel coordinates are flattened)
import h5py
import numpy as np
from tqdm import tqdm
import os
import pickle
import pathlib
import math

from . import utils_cl
from . import utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def _split_frame(frame, photons, N):
    """
    randomly place each photon in frame into one of N 
    split frames
    """
    # the frame index for each photon
    n = np.random.randint(0, N, photons)
    
    # the pixel location of each photon
    loc  = np.searchsorted(np.cumsum(frame), np.arange(1, 1+photons))

    # add the frame offset to each pixel location
    #loc += n * frame.size
    
    out = np.zeros((N,) + frame.shape, dtype = frame.dtype)
    
    # put photons into out
    np.add.at(out, (n, loc), 1)
    return out

def split_frame(frame, photons, target):
    if not target :
        return frame[None, :], [photons]
        
    N   = np.clip(math.floor(photons/target), 1, None)
    if N == 1 :
        return frame[None, :], [photons]
    else :
        frames = _split_frame(frame, photons, N)
        return frames, np.sum(frames, axis=1)




class Data_getter():
    """Save selected frames and pixels in sparse format
    
    Load dataset in memory and keep it there for quick access"""
    def __init__(
        self, 
        mask         = None,
        split_frames = None, 
        cxi_file     = None, 
        filter       = None, 
        dataset      = '/entry_1/data_1/data', 
        background_dataset       = '/entry_1/instrument_1/detector_1/background', 
        background_inds_dataset  = '/entry_1/background_index', 
        background_weights_dataset  = '/entry_1/background_weighting', 
        split        = None, 
        cachedir     = None, 
        frame_model  = None, 
        mpi_split_frames = False, 
        working_directory = './',
        **kwargs
    ):
        
        
        if cachedir is None :
            cachedir = os.path.join(working_directory, 'cachdir')
            # create cachedir if needed
            if not os.path.exists(cachedir) and rank == 0 :
                os.mkdir(cachedir)
            self.cachedir = cachedir
        comm.barrier()
        
        self.fnam    = cxi_file
        stem         = pathlib.Path(cxi_file).stem
        self.dataset = dataset
        self.cache   = {}
        self.sparse_fnam = f'{cachedir}/{stem}-sparse.h5'
        self.filter = filter
        self.mask   = mask
        # convert None to False for h5
        if split_frames is None :
            split_frames = False
        self.split_frames = split_frames

        # check if sparse file exists
        self.sparse_file = pathlib.Path(self.sparse_fnam).is_file()
        self.loaded      = False

        self.frame_model = frame_model

        if self.frame_model == 'background' and self.split_frames :
            err = "frame_model = 'background' is incompatible with split_frames = True"
            raise ValueError(err)

        # background
        self.background_dataset          = background_dataset         
        self.background_inds_dataset     = background_inds_dataset    
        self.background_weights_dataset  = background_weights_dataset 
        
        # check if the filter or mask has changed
        if self.sparse_file == True : 
            self.sparse_file = self.check_sparse()
        
        if not self.sparse_file and rank == 0 :   
            # make sure this only done once 
            # by the first rank
            self.save_sparse() 
         
        comm.barrier()
        self.sparse_file = True
        
        if not self.loaded and mpi_split_frames:
            self.load_sparse_parallel()
        
        elif not self.loaded and not mpi_split_frames:
            self.load_sparse()

        # index frames 
        self.frame_inds_indices = np.concatenate(([0], np.cumsum(self.litpix)))
        self.frame_indices = np.arange(self.shape[0])
        self.pixel_indices = np.arange(self.pixels)
    
    def save_sparse(self):
        inds    = []
        photons = []
        litpix  = []
        photon_sums = []
        frame_index = []
        split_count = 0
        
        with h5py.File(self.fnam, 'r') as f:
            frames = np.where(self.filter(f))[0]
            for d in tqdm(frames, desc = 'extracting data into sparse format'):
                frame   = f[self.dataset][d][self.mask].ravel() 
                p       = np.sum(frame)
                i = 0
                for frame, p in zip(*split_frame(frame, p, self.split_frames)):
                    i += 1
                    inds.append(np.where(frame > 0)[0])
                    photons.append(frame[inds[-1]].copy())
                    litpix.append(len(inds[-1]))
                    photon_sums.append(p)
                    frame_index.append(d)
                
                split_count += i > 1

        if self.frame_model == 'background':
            background_inds = []
            background_weighting = []
            with h5py.File(self.fnam, 'r') as f:
                binds = f[self.background_inds_dataset]
                bw    = f[self.background_weights_dataset]
                back  = f[self.background_dataset]
                
                for d in tqdm(frames, desc = 'extracting background information'):
                    background_inds.append(binds[d])
                    background_weighting.append(bw[d])

                self.background = np.ascontiguousarray(back[()][:, self.mask].astype(np.float32))
            
            self.background_inds      = np.array(background_inds)
            self.background_weighting = np.array(background_weighting)
    
        self.photons     = np.concatenate(photons)
        self.litpix      = np.array(litpix)
        self.inds        = np.concatenate(inds)
        self.photon_sums = np.array(photon_sums)
        self.frame_shape = self.mask.shape
        self.pixels      = len(frame)
        self.shape       = (len(self.litpix), self.pixels) 
        self.frame_index = np.array(frame_index)
            
        for _ in tqdm(range(1), desc = 'saving data in sparse format'):
            with h5py.File(self.sparse_fnam, 'w') as out:
                out['photons']      = self.photons
                out['litpix']       = self.litpix
                out['inds']         = self.inds
                out['frames']       = frames
                out['mask']         = self.mask
                out['photon_sums']  = photon_sums
                out['shape']        = self.shape
                out['frame_shape']  = self.frame_shape
                out['pixels']       = self.pixels
                out['split_frames'] = self.split_frames
                out['frame_index']  = self.frame_index
                 
                if self.frame_model == 'background':
                    out['background']           = self.background
                    out['background_inds']      = self.background_inds
                    out['background_weighting'] = self.background_weighting
        
        if self.split_frames :
            print(f'split {split_count} into {self.shape[0] - len(frames)} frames')
        
        self.sparse_file = True
    
    # not perfect
    # have to manually delete if 
    #   data changes but selected frames are same
    def check_sparse(self):
        # make sure the frame selection and mask 
        # are consistent with 
        # those in the sparse file
        with h5py.File(self.fnam, 'r') as f:
            frames = np.where(self.filter(f))[0]

        with h5py.File(self.sparse_fnam, 'r') as f:
            if not np.allclose(self.mask, f['mask'][()]):
                return False
            if not np.allclose(frames, f['frames'][()]):
                return False
            if f['split_frames'][()] != self.split_frames :
                return False
        return True
    
    def load_sparse(self):
        for _ in tqdm(range(1), desc = 'loading sparse photons from file'):
            with h5py.File(self.sparse_fnam, 'r') as f:
                self.photons = f['photons'][()]
                self.litpix  = f['litpix'][()]
                self.inds    = f['inds'][()]
                self.shape   = f['shape'][()]
                self.pixels  = f['pixels'][()]
                self.mask    = f['mask'][()]
                self.frame_shape = f['frame_shape'][()]
                self.photon_sums = f['photon_sums'][()]
                
                if self.frame_model == 'background':
                    self.background           = f['background'][()]
                    self.background_inds      = f['background_inds'][()]
                    self.background_weighting = f['background_weighting'][()]
        
        self.d_start_mpi  = 0
        self.d_stop_mpi   = len(self.litpix)
        self.total_frames = len(self.litpix)
        
        self.shape       = self.litpix.shape + (self.pixels,)
        self.loaded = True
    
    def load_sparse_parallel(self):
        """
        split frames over processors
        We should be able to use this for a single process also
        """
        for _ in tqdm(range(1), desc = 'loading sparse photons from file'):
            with h5py.File(self.sparse_fnam, 'r') as f:
                litpix  = f['litpix'][()]
                
                total_frames = litpix.shape[0]
                
                d_start, d_stop, dd = utils.chunker_mpi(
                    size, 
                    litpix.shape[0]
                )
                # get coordinates of first and last index
                frame_inds = np.concatenate(([0], np.cumsum(litpix)))
                i0 = frame_inds[d_start[rank]]
                i1 = frame_inds[d_stop[rank]]
                
                self.photons = f['photons'][i0:i1]
                self.inds    = f['inds'][i0:i1]
                self.pixels  = f['pixels'][()]
                self.mask    = f['mask'][()]
                
                self.frame_shape = f['frame_shape'][()]
                self.photon_sums = f['photon_sums'][d_start[rank]: d_stop[rank]]
                self.litpix      = f['litpix'][d_start[rank]: d_stop[rank]]
                self.shape       = self.litpix.shape + (self.pixels,)

                if self.frame_model == 'background':
                    self.background           = f['background'][d_start[rank]: d_stop[rank]]
                    self.background_inds      = f['background_inds'][d_start[rank]: d_stop[rank]]
                    self.background_weighting = f['background_weighting'][d_start[rank]: d_stop[rank]]
        
        self.total_frames = total_frames
        self.d_start_mpi = d_start
        self.d_stop_mpi  = d_stop
        self.loaded = True

    def parse_key(self, key):
        # if key is a tuple of length 2 then we can do
        if isinstance(key, tuple) and len(key) == 2:
            frames = self.frame_indices[key[0]]
            pixels = self.pixel_indices[key[1]]
        
        # anything else must be a slice of frames (right?)
        else :
            frames = self.frame_indices[key]
            pixels = self.pixel_indices
        return frames, pixels
                
    def __getitem__(self, key):
        frames, pixels = self.parse_key(key)
        
        out   = np.zeros((len(frames), len(pixels)), dtype = self.photons.dtype)
        frame = np.zeros((self.pixels,), dtype = self.photons.dtype)
        for i, d in enumerate(frames) :
            frame.fill(0)
            j0, j1                   = self.frame_inds_indices[d: d+2]
            frame[self.inds[j0: j1]] = self.photons[j0: j1]
            out[i]                   = frame[pixels]
            
        return out

class Data_getter_background():
    def __init__(self, data_getter):
        self.data_getter = data_getter
    
    def __getitem__(self, key):
        frames, pixels = self.data_getter.parse_key(key)
         
        out   = np.zeros((len(frames), len(pixels)), dtype = self.data_getter.background.dtype)
        for i, d in enumerate(frames) :
            bind   = self.data_getter.background_inds[d]
            b      = self.data_getter.background_weighting[d]
            out[i] = b * self.data_getter.background[bind, pixels]
            
        return out
    
class Data_getter_full_frames():
    def __init__(self, data_getter):
        self.data_getter = data_getter
    
    def __getitem__(self, key):
        K = self.data_getter.__getitem__(key)
         
        out = np.zeros((K.shape[0],) + self.data_getter.mask.shape, 
                       dtype = self.data_getter.photons.dtype)
        for d in range(K.shape[0]):
            out[d, self.data_getter.mask] = K[d]
        return out


class Data_getter_cl(Data_getter):
    def __init__(self, queue = None, **kwargs):
        super().__init__(**kwargs)
        
        self.queue = queue
        self.shape = self.shape[::-1]
        
        self.out_cl = None
        
    def __getitem__(self, key):
        # transpose key
        if isinstance(key, tuple) and len(key) == 2:
            keyT = key[::-1]
        else : 
            keyT = (slice(None), key)
        out = super().__getitem__(keyT)
        
        # transpose: K_id
        out = out.T
        
        if self.out_cl is None or \
            self.out_cl.shape[1] < out.shape[1] or \
            self.out_cl.shape[0] < out.shape[0]:
            self.out_cl = utils_cl.to_gpu(out, None, self.queue)
        else :
            self.out_cl = utils_cl.to_gpu(out, self.out_cl, self.queue)
        
        return self.out_cl

