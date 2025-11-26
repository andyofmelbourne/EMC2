# load raw data
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

from scipy.sparse import csr_matrix


class RawDataGetterBase():
    """
    An abstract interface to raw data

    example:
        a = RawDataGetterBase()
        array_like = a[frame_number, pixel_number]
    """
    dtype = None
    shape = None
    size = None
    nbytes = None
    frames = None
    mask = None
    source_dtype = None
    source_shape = None
    fnam = None
    source_shape = None
    source_dtype = None

    def __init__(self):
        pass

    def __getitem__(self, key):
        """
        parse self[key] like a numpy array
        """
        pass


class RawDataGetterCXI(RawDataGetterBase):
    """
    An interface for loading masked data from a hdf5 file:
        data = RawDataGetterCXI('example.h5', '/data', mask, frames)

        is equivalent to:

        data = h5py.File('example.h5')['/data'][frames][:, mask]
    """

    def __init__(
            self,
            cxi_file,
            data_path='/entry_1/data_1/data',
            mask=None,
            frames=None,
            dtype=None,
            fnam=None):

        self.cxi_file = cxi_file
        self.data_path = data_path

        self.check_data(mask, frames, dtype)

        self.is_loaded = False

        # for saving / loading
        if fnam is None:
            self.fnam = self._get_fnam()

    def _get_fnam(self):
        fnam = f'data_{id(self)}.h5'
        return fnam

    def check_data(self, mask, frames, dtype):
        with h5py.File(self.cxi_file) as f:
            data = f[self.data_path]

            self.source_dtype = data.dtype
            self.source_shape = data.shape

            if frames is None:
                self.frames = np.arange(data.shape[0])
            else:
                self.frames = frames

            if mask is None:
                self.mask = np.ones(data.shape[1:], dtype=bool)
            else:
                self.mask = mask
                assert (mask.shape == data.shape[1:])

            n_pixels = np.sum(self.mask)

            if dtype is None:
                self.dtype = data.dtype
            else:
                self.dtype = dtype

            self.shape = (len(self.frames), n_pixels)
            self.size = self.shape[0] * self.shape[1]
            self.nbytes = self.size * self.dtype.itemsize

    def init_data(self):
        self.data = np.empty(self.shape, dtype=self.dtype)
        self.data_sum = np.empty(self.shape[0], dtype=float)

    def load_data(self):
        self.init_data()

        with h5py.File(self.cxi_file) as f:
            data = f[self.data_path]

            it = tqdm(
                    enumerate(self.frames),
                    total=self.shape[0],
                    desc='loading data')

            for i, d in it:
                self.data[i] = data[d][self.mask]
                self.data_sum[i] = np.sum(self.data[i])

        self.is_loaded = True

    def __getitem__(self, key):
        if self.is_loaded is False:
            raise ValueError('must call self.load_data() first!')
        return self.data[key]


class SparseData():
    """
    A wrapper around a Compressed Sparse Row matrix.

    This object makes it simple to incrementally build a
    csr matrix (found here: from scipy.sparse import csr_matrix)

    It can also be treated as a dense numpy array object, but only
    when indexing by column:

        dense_nm = np.random.random(8, 7) > 0.8

        s_nm = SparseData((8, 7), bool)

        for n in range(8):
            s_nm.add_row(dense_nm[n])

        dense_nm[3:7] == s_nm[3:7] <-- dense numpy array

        s_nm.csr[3:7] <-- csr_matrix (good for matmul)
    """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(shape)
        self.buffer = None
        self.last_key = None

        self.dtype_inds = np.min_scalar_type(shape[1])

        assert (len(self.shape) == 2)

        self._init_vars()

    def _init_vars(self):
        shape = self.shape

        self.row_inds = []
        self.non_zero_values = []
        self.indptr = np.zeros(1+shape[0], dtype=np.int32)

        self.litpix = np.empty(shape[0], dtype=np.int32)
        self.total_row_counts = np.empty(shape[0], dtype=np.int32)

        self.is_loaded = False
        self.csr = None

    def add_row(self, ar):
        """
        Add a frame to the sparse object
        """
        ms = np.where(ar>0)[0]
        ms = ms.astype(self.dtype_inds)
        self.row_inds.append(ms)

        self.non_zero_values.append(ar[ms].astype(self.dtype))

        n = len(self.row_inds)-1
        self.litpix[n] = len(ms)
        self.total_row_counts[n] = np.sum(self.non_zero_values[n])

        if n == (self.shape[0]-1):
            self.construct_csr_matrix()

    def construct_csr_matrix(self):
        self.row_inds = np.concatenate(self.row_inds)
        self.non_zero_values = np.concatenate(self.non_zero_values)
        self.indptr[1:] = np.cumsum(self.litpix)

        self._construct_csr_matrix()

    def _construct_csr_matrix(self):
        # prevent data duplication
        self.csr = csr_matrix(self.shape, dtype=self.dtype)
        self.csr.data = self.non_zero_values
        self.csr.indices = self.row_inds
        self.csr.indptr = self.indptr

        self.is_loaded = True

    def get_sparse(self, d):
        """
        return non-zero counts & row indices
        """
        i0, i1 = self.indptr[d:d+2]
        inds = self.row_inds[i0:i1]
        counts = self.non_zero_values[i0:i1]
        return inds, counts

    def __getitem__(self, key):
        if not self.is_loaded:
            raise ValueError('need to load all frames before calling!')

        if key == self.last_key:
            return self.buffer

        self.last_key = key
        self.buffer = self.csr[key].toarray()

        return self.buffer

    def save_to_file(self, fnam):
        if not self.is_loaded:
            raise ValueError('need to load all frames before saving!')

        with h5py.File(fnam, 'w') as f:
            f['data'] = self.non_zero_values
            f['indices'] = self.row_inds
            f['indptr'] = self.indptr
            f['total_row_counts'] = self.total_row_counts

    def load_saved_data(self, fnam):
        if self.is_loaded:
            return

        with h5py.File(fnam, 'r') as f:
            self.non_zero_values = f['data'][()]
            self.row_inds = f['indices'][()]
            self.indptr = f['indptr'][()]
            self.total_row_counts = f['total_row_counts'][()]

        self.litpix = np.diff(self.indptr)

        self._construct_csr_matrix()

    def unload(self):
        """
        delete data to free memory
        """
        self._init_vars()


class RawDataGetterSparseCXI(RawDataGetterCXI):
    """
    An interface for loading masked data from a hdf5 file:
        data = RawDataGetterCXI('example.h5', '/data', mask, frames)

        is equivalent to:

        data = h5py.File('example.h5')['/data'][frames][:, mask]

    This class stores the data in a sparse array.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_data(self):
        self.data = SparseData(self.shape, dtype=self.dtype)
        self.data_sum = np.empty(self.shape[0], dtype=float)

    def load_data(self):
        self.init_data()

        with h5py.File(self.cxi_file) as f:
            data = f[self.data_path]

            it = tqdm(
                    enumerate(self.frames),
                    total=self.shape[0],
                    desc='loading data')

            for i, d in it:
                self.data.add_row(data[d][self.mask])

        self.data_sum[:] = self.data.total_row_counts

        assert (self.data.is_loaded)
        self.is_loaded = True

    def save_to_file(self):
        self.data.save_to_file(self.fnam)

        # add mask and frames info
        with h5py.File(self.fnam, 'r+') as f:
            f['mask'] = self.mask
            f['frames'] = self.frames

    def load_from_file(self):
        self.data.load_saved_data(self.fnam)

    def unload(self):
        """
        delete data to free memory
        """
        self.data.unload()


class Data():
    """
    Interface to data from an x-ray pixel detector over a number of events.
    """

    def __init__(self, detector, raw_data):
        self.detector = detector
        self.raw_data = raw_data
        self.data_sum = raw_data.data_sum

    def __getitem__(self, key):
        return self.raw_data[key]


class DataCXI(RawDataGetterCXI):
    """
    Interface to data from an x-ray pixel detector over a number of events.
    """
    def __init__(self, cxi_file, detector, mask, frames, **kwargs):
        super().__init__(cxi_file, mask=mask, frames=frames)
        self.detector = detector
        # polarisation and solid angle correction factor:
        # N_i = C_i I(q_i)
        self.C_i = detector.C[mask]


class DataSparseCXI(RawDataGetterSparseCXI):
    """
    Interface to data from an x-ray pixel detector over a number of events.

    RawDataGetterBase
      -> RawDataGetterCXI
        -> RawDataGetterSparseCXI (uses SparseData)
          -> DataSparseCXI
    """
    def __init__(self, cxi_file, detector, mask, frames, background=False, background_file=None, **kwargs):
        super().__init__(cxi_file, mask=mask, frames=frames)
        self.detector = detector
        # polarisation and solid angle correction factor:
        # N_i = C_i I(q_i)
        self.C_i = detector.C[mask]

        if background:
            self.B_di = BackCXI(self, file=background_file)

    def save_to_file(self):
        """
        add pixel_size and xyz coords to file for viewing
        """
        super().save_to_file()
        self.B_di.save_to_file()
        if not self.is_loaded:
            raise ValueError('need to load all frames before saving!')

        with h5py.File(self.fnam, 'r+') as f:
            f['pixel_size'] = self.detector.pixel_size
            f['xyz_map'] = self.detector.xyz

    def load_data(self):
        super().load_data()
        self.B_di.load_data()

    def load_from_file(self):
        super().load_from_file()
        self.B_di.load_from_file()

    def unload(self):
        super().unload()
        self.B_di.unload()


class BackCXI():
    """
    B_di = BackSparseCXI(data_cxi_obj, 'B_ji', 'j_d', 'b_d')

    is equivalent to:
        B_di[d, i] = b_d[d] B_ji[j_d[d], mask][i]

    where mask = data_sparse_cxi_obj.mask
    and 'B_ji', 'j_d' and 'b_d' are datasets stored in:
        data_sparse_cxi_obj.cxi_file

    pixel slicing is not supported:
        B_di[(1,2,3)]      good
        B_di[:10, :]       good
        B_di[:10, :10]     bad
        B_di[:10][:, :10]  good
    """
    def __init__(
            self,
            data,
            file=None,
            B_ji='/entry_1/instrument_1/detector_1/background',
            j_d='/entry_1/background_index',
            b_d='/entry_1/background_weighting',
            dtype=np.float32,
            ):

        self.dtype = dtype
        self.data = data
        self.shape = data.shape
        self.frame_inds = np.arange(self.shape[0])
        self.data_source = [B_ji, j_d, b_d]
        self.is_loaded = False
        if file is None:
            file = data.cxi_file
        self.file = file

        self.B_ji = None
        self.j_d = None
        self.b_d = None
        self.fnam = f'back_{id(self)}.h5'

    def __getitem__(self, key):
        assert (self.is_loaded)

        # get frames
        frames = np.atleast_1d(self.frame_inds[key])

        shape = (len(frames), self.data.shape[1])
        out = np.zeros(shape, dtype=self.dtype)

        js = self.j_d[frames]
        out[:] = self.b_d[frames, None] * self.B_ji[js]
        return out

    def load_data(self):
        frames = self.data.frames
        a, b, c = self.data_source
        with h5py.File(self.file, 'r') as f:
            self.B_ji = f[a][()][:, self.data.mask]
            self.j_d = f[b][()][frames]
            self.b_d = f[c][()][frames]

        self.B_ji = np.ascontiguousarray(self.B_ji.astype(self.dtype))
        self.j_d = np.ascontiguousarray(self.j_d.astype(np.int32))
        self.b_d = np.ascontiguousarray(self.b_d.astype(self.dtype))

        # data_sum = sum_i B_di
        self.data_sum = self.b_d * np.sum(self.B_ji, axis=1)[self.j_d]

        self.is_loaded = True

    def save_to_file(self):
        if not self.is_loaded:
            raise ValueError('need to load all frames before saving!')

        with h5py.File(self.fnam, 'w') as f:
            f['B_ji'] = self.B_ji
            f['j_d'] = self.j_d
            f['b_d'] = self.b_d
            f['B_d'] = self.data_sum

    def load_from_file(self):
        if self.is_loaded:
            return
        print('loading background')

        with h5py.File(self.fnam, 'r') as f:
            self.B_ji = f['B_ji'][()]
            self.j_d = f['j_d'][()]
            self.b_d = f['b_d'][()]
            self.data_sum = f['B_d'][()]

        # hack
        fnam = 'fluence.h5'
        if Path(fnam).is_file():
            with h5py.File('fluence.h5') as f:
                if 'b_d' in f:
                    self.b_d = np.ascontiguousarray(f['b_d'][()].astype(np.float32))

        self.is_loaded = True

    def unload(self):
        if self.is_loaded:
            del self.B_ji
            del self.j_d
            del self.b_d
            self.B_ji = None
            self.j_d = None
            self.b_d = None

            self.is_loaded = False


class DataSparseCXI_full_frame():
    """
    load data saved by DataSparseCXI as a numpy like object:
        fnam = 'data_126376212075904.h5'
        a = DataSparseCXI_full_frame(fnam)
        a[0] yields an array with the same shape as the detector mask
    """
    def __init__(self, fnam):
        self.fnam = fnam
        self.load()

    def load(self):
        # load sparse data
        with h5py.File(self.fnam, 'r') as f:
            self.mask = f['mask'][()]
            self.dtype = f['data'].dtype
            self.pixel_size = f['pixel_size'][()]
            self.frames = f['frames'][()]
            self.xyz_map = f['xyz_map'][()]

        self.shape = (len(self.frames),) + self.mask.shape
        self.unmasked_pixels = np.sum(self.mask)
        self.masked_shape = (len(self.frames), self.unmasked_pixels)
        self.data = SparseData(self.masked_shape, self.dtype)
        self.data.load_saved_data(self.fnam)

    def __getitem__(self, key):
        masked_data = np.atleast_2d(self.data[key])
        s = masked_data.shape

        out = np.zeros((s[0],) + self.shape[1:], dtype=self.dtype)

        for i, d in enumerate(masked_data):
            out[i][self.mask] = d

        return np.squeeze(out)
