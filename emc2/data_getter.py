# cache results using photon sparse format in pickle files
# all dimension in dataset except the first are ravelled
# (the pixel coordinates are flattened)
import h5py
import numpy as np
from tqdm import tqdm
import os
import pathlib
import math
import psutil
import logging
from . import utils

logger = logging.getLogger(__name__)


def _split_frame(frame, photons, N):
    """
    randomly place each photon in frame into one of N
    split frames
    """
    # the frame index for each photon
    n = np.random.randint(0, N, photons)

    # the pixel location of each photon
    loc = np.searchsorted(np.cumsum(frame), np.arange(1, 1+photons))

    # add the frame offset to each pixel location
    # loc += n * frame.size

    out = np.zeros((N,) + frame.shape, dtype=frame.dtype)

    # put photons into out
    np.add.at(out, (n, loc), 1)
    return out


def split_frame(frame, photons, target):
    if not target:
        return frame[None, :], [photons]

    N = np.clip(math.floor(photons/target), 1, None)
    if N == 1:
        return frame[None, :], [photons]
    else:
        frames = _split_frame(frame, photons, N)
        return frames, np.sum(frames, axis=1)


def transpose_data(
    K_di,
    dataset='/entry_1/data_1/data',
    fnam='dataT.h5',
    working_directory=None,
    cachedir=None,
    dtype=None,
    max_mem_mb=4000
):
    """
    transpose dataset and write to cxi file
    for the purposes of spawning another Data_getter
    K_id
    """
    if cachedir is None:
        cachedir = os.path.join(working_directory, 'cachdir')
        # create cachedir if needed
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)
        cachedir = cachedir

    fnam_out = os.path.join(cachedir, fnam)
    logger.info(f'saving transpose of data to {fnam_out}/{dataset}')

    D, I = K_di.shape

    # determine chunk size
    mb = (D*I) * np.dtype(K_di.dtype).itemsize / 1024**2
    chunksize = int(I * mb / max_mem_mb)

    if dtype is None:
        dtype = K_di.dtype

    with h5py.File(fnam_out, 'w') as f:
        data = f.create_dataset(
            dataset,
            shape=(I, D),
            dtype=dtype,
            chunks=(1, D),
            compression='gzip',
            compression_opts=1,
            shuffle=True
        )

        # loop over pixel chunks
        for i0, i1, di in tqdm(
            utils.chunker(chunksize, K_di.shape[1]), desc='tranposing data set'
        ):
            data[i0:i1] = K_di[:, i0:i1].T


def get_sparse_fnam(
    cxi_file,
    working_directory,
    mask,
    cachedir=None,
    fnam_append='-sparse-{pixels}.h5'
):
    if cachedir is None:
        cachedir = os.path.join(working_directory, 'cachdir')
        # create cachedir if needed
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)

    stem = pathlib.Path(cxi_file).stem
    fnam_append = fnam_append.format(pixels=np.sum(mask))
    sparse_fnam = f'{cachedir}/{stem}{fnam_append}'
    return sparse_fnam


class Data_getter():
    """Save selected frames and pixels in sparse format

    Load dataset in memory and keep it there for quick access"""
    def __init__(
        self,
        mask=None,
        split_frames=None,
        cxi_file=None,
        filter=None,
        dataset='/entry_1/data_1/data',
        background_dataset='/entry_1/instrument_1/detector_1/background',
        background_inds_dataset='/entry_1/background_index',
        background_weights_dataset='/entry_1/background_weighting',
        cachedir=None,
        frame_model=None,
        sparse_fnam=None,
        fnam_append='-sparse-{pixels}.h5',
        mpi_split_frames=False,
        working_directory='./',
        load_dense=True,
        delay_data_load=False,
        **kwargs
    ):
        self.fnam = cxi_file
        self.rank = 0
        self.size = 1
        self.load_dense = load_dense

        if sparse_fnam is None:
            self.sparse_fnam = get_sparse_fnam(
                cxi_file,
                working_directory,
                mask,
                cachedir,
                fnam_append
            )

        # check if sparse file exists
        # if another process is running this at the same time
        # there could be a race condition
        self.sparse_file = pathlib.Path(self.sparse_fnam).is_file()
        self.loaded = False

        self.dataset = dataset

        if mpi_split_frames:
            self.rank = mpi_split_frames[0]
            self.size = mpi_split_frames[1]

        # filter can be False, None, ndarray or function
        self.filter = filter
        self.mask = mask
        # convert None to False for h5
        if split_frames is None:
            split_frames = False
        self.split_frames = split_frames

        self.frame_model = frame_model

        if self.frame_model == 'background' and self.split_frames:
            err = "frame_model = 'background' is "\
                  "incompatible with split_frames = True"
            raise ValueError(err)

        # background
        self.background_dataset = background_dataset
        self.background_inds_dataset = background_inds_dataset
        self.background_weights_dataset = background_weights_dataset

        # check if the filter or mask has changed
        if self.sparse_file is True:
            logger.debug('checking existing sparse file...')
            self.sparse_file = self.check_sparse()
            logger.debug(self.sparse_file)

        if not self.sparse_file and self.rank == 0:
            # make sure this only done once
            # by the first rank
            self.save_sparse()

        self.sparse_file = True

        if not delay_data_load:
            self.load()

    def load(self):
        if not self.loaded and self.size > 1:
            self.load_sparse_parallel()

        elif not self.loaded and self.size == 1:
            self.load_sparse()

        # index frames
        self.frame_inds_indices = np.concatenate(([0], np.cumsum(self.litpix)))
        self.frame_indices = np.arange(self.shape[0])
        self.pixel_indices = np.arange(self.pixels)

        # load dense data
        # if it takes up less than 20% of available memory
        self.dense_data = None
        if self.load_dense:
            mem_avail = psutil.virtual_memory().available
            mem_data = self.size * np.dtype(self.dtype).itemsize
            logger.debug(f'Available memory {mem_avail/1024**3} gb')
            logger.debug(f'Dense data size  {mem_data/1024**3} gb')
            if mem_data < (.2 * mem_avail):
                logger.debug('loading dense dataset')
                self.dense_data = self[:, :]

    def save_sparse(self):
        inds = []
        photons = []
        litpix = []
        photon_sums = []
        frame_index = []
        split_count = 0

        with h5py.File(self.fnam, 'r') as f:
            if callable(self.filter):
                frames = np.where(self.filter(f))[0]
            elif (
                isinstance(self.filter, np.ndarray)
                and self.filter.shape[0] == f[self.dataset].shape[0]
            ):
                frames = np.where(self.filter)[0]
            else:
                frames = np.arange(f[self.dataset].shape[0])

            desc = f'extracting data into sparse format {self.fnam}'
            for d in tqdm(frames, desc=desc):
                frame = f[self.dataset][d][self.mask].ravel()
                p = np.sum(frame)
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
            background_sums = []
            with h5py.File(self.fnam, 'r') as f:
                binds = f[self.background_inds_dataset]
                bw = f[self.background_weights_dataset]
                back = f[self.background_dataset]

                desc = 'extracting background information'
                for d in tqdm(frames, desc=desc):
                    background_inds.append(binds[d])
                    background_weighting.append(bw[d])

                self.background = np.ascontiguousarray(
                    back[()][:, self.mask].astype(np.float32)
                )

            self.background_inds = np.array(background_inds)
            self.background_weighting = np.array(background_weighting)
            background_sums = np.sum(self.background, axis=-1)
            self.background_sums = self.background_weighting \
                * background_sums[self.background_inds]

        self.photons = np.concatenate(photons)
        self.litpix = np.array(litpix)
        self.inds = np.concatenate(inds)
        self.photon_sums = np.array(photon_sums)
        self.frame_shape = self.mask.shape
        self.pixels = len(frame)
        self.shape = (len(self.litpix), self.pixels)
        self.size = len(self.litpix) * self.pixels
        self.frame_index = np.array(frame_index)

        for _ in tqdm(range(1), desc='saving data in sparse format'):
            with h5py.File(self.sparse_fnam, 'w') as out:
                out['photons'] = self.photons
                out['litpix'] = self.litpix
                out['inds'] = self.inds
                out['frames'] = frames
                out['mask'] = self.mask
                out['photon_sums'] = photon_sums
                out['shape'] = self.shape
                out['frame_shape'] = self.frame_shape
                out['pixels'] = self.pixels
                out['split_frames'] = self.split_frames
                out['frame_index'] = self.frame_index

                if self.frame_model == 'background':
                    out['background'] = self.background
                    out['background_inds'] = self.background_inds
                    out['background_weighting'] = self.background_weighting
                    out['background_sums'] = self.background_sums

        if self.split_frames:
            logger.info(f'split {split_count} into '
                        f'{self.shape[0] - len(frames)} frames')

        self.sparse_file = True

    # not perfect
    # have to manually delete if
    #   data changes but selected frames are same
    def check_sparse(self):
        # make sure the frame selection and mask
        # are consistent with
        # those in the sparse file
        if hasattr(self.filter, 'dtype') and self.filter.dtype == bool:
            frames = np.where(self.filter)[0]

        elif (
            hasattr(self.filter, 'dtype')
            and (self.filter.dtype == np.int32
                 or self.filter.dtype == np.int64)
        ):
            frames = self.filter

        elif callable(self.filter):
            with h5py.File(self.fnam, 'r') as f:
                if self.filter:
                    frames = np.where(self.filter(f))[0]
                else:
                    frames = np.arange(f[self.dataset].shape[0])

        else:
            err = f'cannot parse filter type {type(self.filter)}'
            raise ValueError(err)

        with h5py.File(self.sparse_fnam, 'r') as f:
            if (
                self.mask.shape != f['mask'].shape
                or not np.allclose(self.mask, f['mask'][()])
            ):
                return False

            if (
                frames.shape != f['frames'].shape
                or not np.allclose(frames, f['frames'][()])
            ):
                return False

            if f['split_frames'][()] != self.split_frames:
                return False

            if self.frame_model == 'background' and 'background' not in f:
                return False

        return True

    def load_sparse(self):
        for _ in tqdm(range(1), desc='loading sparse photons from file'):
            with h5py.File(self.sparse_fnam, 'r') as f:
                self.photons = f['photons'][()]
                self.litpix = f['litpix'][()]
                self.inds = f['inds'][()]
                self.shape = f['shape'][()]
                self.pixels = f['pixels'][()]
                self.mask = f['mask'][()]
                self.frame_shape = f['frame_shape'][()]
                self.photon_sums = f['photon_sums'][()]

                if self.frame_model == 'background':
                    # https://github.com/mpi4py/mpi4py/issues/177
                    # I don't know why sometimes datasets are read with
                    # dtype = dtype('<f4')
                    self.background = f['background'][()].newbyteorder('=')
                    self.background_inds = f['background_inds'][()]
                    self.background_weighting = f['background_weighting'][()]
                    self.background_sums = f['background_sums'][()]

        self.d_start_mpi = [0]
        self.d_stop_mpi = [len(self.litpix)]
        self.total_frames = len(self.litpix)
        self.dtype = self.photons.dtype

        self.shape = self.litpix.shape + (self.pixels,)
        self.size = len(self.litpix) * self.pixels
        self.loaded = True

    def load_sparse_parallel(self):
        """
        split frames over processors
        We should be able to use this for a single process also
        """
        rank = self.rank
        size = self.size
        desc = f'loading sparse photons from file for data_chunk {rank}'
        for _ in tqdm(range(1), desc=desc):
            with h5py.File(self.sparse_fnam, 'r') as f:
                litpix = f['litpix'][()]

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
                self.inds = f['inds'][i0:i1]
                self.pixels = f['pixels'][()]
                self.mask = f['mask'][()]

                d0, d1 = d_start[rank], d_stop[rank]
                self.frame_shape = f['frame_shape'][()]
                self.photon_sums = f['photon_sums'][d0:d1]
                self.litpix = f['litpix'][d0:d1]
                self.shape = self.litpix.shape + (self.pixels,)

                if self.frame_model == 'background':
                    self.background = f['background'][:]
                    self.background_inds = f['background_inds'][d0:d1]
                    self.background_weighting = \
                        f['background_weighting'][d0:d1]
                    self.background_sums = f['background_sums'][()]

        self.total_frames = total_frames
        self.d_start_mpi = d_start
        self.d_stop_mpi = d_stop
        self.dtype = self.photons.dtype
        self.loaded = True

    def parse_key(self, key):
        # if key is a tuple of length 2 then we can do
        if isinstance(key, tuple) and len(key) == 2:
            frames = self.frame_indices[key[0]]
            pixels = self.pixel_indices[key[1]]

        # anything else must be a slice of frames (right?)
        else:
            frames = self.frame_indices[key]
            pixels = self.pixel_indices
        return frames, pixels

    def getitem_dense(self, key):
        return self.dense_data[key]

    def sparse(self, d):
        j0, j1 = self.frame_inds_indices[d: d+2]
        inds = self.inds[j0: j1]
        K = self.photons[j0: j1]
        return K, inds

    def getitem_sparse(self, key):
        frames, pixels = self.parse_key(key)

        out = np.zeros((len(frames), len(pixels)), dtype=self.photons.dtype)
        frame = np.zeros((self.pixels,), dtype=self.photons.dtype)
        for i, d in enumerate(frames):
            frame.fill(0)
            j0, j1 = self.frame_inds_indices[d: d+2]
            frame[self.inds[j0: j1]] = self.photons[j0: j1]
            out[i] = frame[pixels]

        return out

    def __getitem__(self, key):
        if self.dense_data is not None:
            return self.getitem_dense(key)
        else:
            return self.getitem_sparse(key)


class Data_getter_background():
    def __init__(self, data_getter):
        self.data_getter = data_getter
        self.shape = data_getter.shape
        self.dtype = data_getter.background.dtype
        self.background_sums = data_getter.background_sums

    def sparse(self, d, pixels):
        bind = self.data_getter.background_inds[d]
        b = self.data_getter.background_weighting[d]
        B = b * self.data_getter.background[bind, pixels]
        return B

    def __getitem__(self, key):
        frames, pixels = self.data_getter.parse_key(key)

        out = np.zeros((len(frames), len(pixels)), dtype=self.dtype)
        for i, d in enumerate(frames):
            bind = self.data_getter.background_inds[d]
            b = self.data_getter.background_weighting[d]
            out[i] = b * self.data_getter.background[bind, pixels]

        return out


class Data_getter_full_frames():
    def __init__(self, data_getter):
        self.data_getter = data_getter

    def __getitem__(self, key):
        K = self.data_getter.__getitem__(key)

        out = np.zeros((K.shape[0],) + self.data_getter.mask.shape,
                       dtype=self.data_getter.photons.dtype)
        for d in range(K.shape[0]):
            out[d, self.data_getter.mask] = K[d]
        return out


class Data_getter_cl(Data_getter):
    def __init__(self, queue=None, **kwargs):
        super().__init__(**kwargs)

        self.queue = queue
        self.shape = self.shape[::-1]

        self.out_cl = None

        from . import utils_cl
        self.utils_cl = utils_cl

    def __getitem__(self, key):
        # transpose key
        if isinstance(key, tuple) and len(key) == 2:
            keyT = key[::-1]
        else:
            keyT = (slice(None), key)
        out = super().__getitem__(keyT)

        # transpose: K_id
        out = out.T

        if (
            self.out_cl is None
            or self.out_cl.shape[1] < out.shape[1]
            or self.out_cl.shape[0] < out.shape[0]
        ):
            self.out_cl = self.utils_cl.to_gpu(out, None, self.queue)
        else:
            self.out_cl = self.utils_cl.to_gpu(out, self.out_cl, self.queue)

        return self.out_cl
