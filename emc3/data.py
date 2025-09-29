# load raw data
import h5py
import numpy as np
from tqdm import tqdm

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
            dtype=None):

        self.cxi_file = cxi_file
        self.data_path = data_path

        self.check_data(mask, frames, dtype)

        self.is_loaded = False

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

        self.dtype_inds = np.min_scalar_type(shape[1])

        assert (len(self.shape) == 2)

        # instead of storing the locations of both
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

        # prevent data duplication
        self.csr = csr_matrix(self.shape, dtype=self.dtype)
        self.csr.data = self.non_zero_values
        self.csr.indices = self.row_inds
        self.csr.indptr = self.indptr

        self.is_loaded = True

    def __getitem__(self, key):
        if not self.is_loaded:
            raise ValueError('need to load all frames before calling!')

        return self.csr[key].toarray()


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
    """
    def __init__(self, cxi_file, detector, mask, frames, **kwargs):
        super().__init__(cxi_file, mask=mask, frames=frames)
        self.detector = detector
        # polarisation and solid angle correction factor:
        # N_i = C_i I(q_i)
        self.C_i = detector.C[mask]
