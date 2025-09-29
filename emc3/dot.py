import numpy as np
import psutil

def dot_numpy(A, B):
    return A.dot(B)

# sum_i K_di . W_ri
# sum_d P_dr . K_di

class ChunkerBase():
    def __init__(self, a):
        self.A = a
        self.shape = a.shape
        self.dtype = a.dtype
        self.size = a.size
        self.nbytes = a.nbytes

    def set_chunksize(self, Mc):
        pass

    def __getitem__(self, key):
        return self.A[key]

def make_chunker(ar):
    tar = type(ar)
    if issubclass(tar, ChunkerBase):
        return ar
    else:
        return ChunkerBase(ar)
    # else:
    #     raise ValueError(f'array must be ndarray or Chunker not {tar}')


class AdotB():
    """
    Object for calculating: C_mn = sum_k A_mk B_kn

    A and B must be objects with the following properties:
        A.set_chunksize(Mc) --> prepares A for loading chunks of size (Mc, K)
        B.set_chunksize(Nc) --> prepares B for loading chunks of size (K, Nc)

        A[i:j] --> returns a numpy array A[i:j, :]

        A.shape
        A.size
        A.dtype
        A.nbytes

    C_mn can be loaded in chunks:
        C_mn.get_chunk(mc, nc)
            = sum_k A_mk[mc Mc: (mc+1) Mc, k] B_nk[k, nc Nc: (nc + 1) Nc]

    if the last chunk extends beyond the bounds of A and/or B then the result
    is also truncated.

    or the full array can be calculated: C_nm()
    """
    def __init__(self, A, B):
        self.A = make_chunker(A)
        self.B = make_chunker(B)

        self.dtype = np.result_type(A.dtype, B.dtype)
        self.shape = (A.shape[0], B.shape[1])
        self.size = np.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize

        self.determine_chunksize()

    def determine_chunksize(self):
        """
        set chunksize for N = chunksize M
        so that each chunk leads to: (C, K) . (K, C) = (C, C)
        memory required is at least:
            mem = d_A C K + d_B K C + d_C C C

        where C is the chunksize and d_A is the itemsize for A
            mem = C K (d_A + d_B) + d_C C^2
            C_max = (-b + sqrt(b^2 + 4 d_C mem)) / 2 d_C
            b = K (d_A + d_B)

        Mc = min(C_max, M)
        Nc = min(C_max, N)
        """
        mem_avail = psutil.virtual_memory().available
        a = self.dtype.itemsize
        b = self.A.shape[1] * (self.A.dtype.itemsize + self.B.dtype.itemsize)
        c = - mem_avail
        C_max = int((-b + (b**2 - 4 * a * c)**0.5) / (2 * a))

        M, N = self.shape
        K = self.A.shape[1]

        self.N_chunksize = min(C_max, N)
        self.M_chunksize = min(C_max, M)

        self.N_chunks = int(np.ceil(N/self.N_chunksize))
        self.M_chunks = int(np.ceil(M/self.M_chunksize))

        self.A.set_chunksize((self.M_chunksize, K))
        self.B.set_chunksize((K, self.N_chunksize))

    def get_chunk(self, mc, nc):
        C = self.M_chunksize
        self.m_inds = (mc * C, min((mc+1) * C, self.A.shape[0]))

        C = self.N_chunksize
        self.n_inds = (nc * C, min((nc+1) * C, self.B.shape[1]))

        Achunk = self.A[self.m_inds[0]: self.m_inds[1], :]
        Bchunk = self.B[:, self.n_inds[0]: self.n_inds[1]]
        return Achunk @ Bchunk

    def __call__(self):
        C_mn = np.empty(self.shape, self.dtype)
        for mc in range(self.M_chunks):
            for nc in range(self.N_chunks):
                chunk = self.get_chunk(mc, nc)
                m0, m1 = self.m_inds
                n0, n1 = self.n_inds
                C_mn[m0:m1, n0:n1] = chunk
        return C_mn
