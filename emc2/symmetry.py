import numpy as np


class Symmetry_old():
    def __init__(self, i0, shape, symmetry = 'inversion'):
        """
        symmetry = 'P1', 'inversion', 'C6' or 'D6'
        """
        self.n = np.arange(np.prod(shape))

        if len(shape) == 3 :
            self.i = self.n // (shape[1] * shape[2])
            self.j = self.n // shape[2] % shape[1]
            self.k = self.n % shape[2]
        elif len(shape) == 2 :
            self.i = self.n // shape[1]
            self.j = self.n  % shape[1]

        self.N = shape[0]

        self.i0 = i0

        # only cubes for now
        assert(np.allclose(shape, self.N))
        assert(symmetry in ['P1', 'inversion', 'C6', 'D6'])
        assert(len(shape) in [2, 3])

        if   symmetry == 'inversion' and len(shape) == 3 :
            self.get_asymmetric_unit   = self.get_asymmetric_unit_inversion_3D
            self.get_symmetry_partners = self.get_symmetry_partners_inversion_3D
            self.get_asymmetric_unit_mapping =\
            self.get_asymmetric_unit_mapping_inversion_3D

        elif symmetry == 'inversion' and len(shape) == 2 :
            self.get_asymmetric_unit   = self.get_asymmetric_unit_inversion_2D
            self.get_symmetry_partners = self.get_symmetry_partners_inversion_2D
            self.get_asymmetric_unit_mapping =\
            self.get_asymmetric_unit_mapping_inversion_2D

        elif symmetry == 'D6' and len(shape) == 3:
            self.get_asymmetric_unit   = self.get_asymmetric_unit_D6_3D
            self.get_symmetry_partners = self.get_symmetry_partners_D6_3D
            self.get_asymmetric_unit_mapping =\
            self.get_asymmetric_unit_mapping_D6_3D

        elif symmetry == 'C6' and len(shape) == 3:
            self.get_asymmetric_unit   = self.get_asymmetric_unit_C6_3D
            self.get_symmetry_partners = self.get_symmetry_partners_C6_3D
            self.get_asymmetric_unit_mapping =\
            self.get_asymmetric_unit_mapping_C6_3D

        elif symmetry == 'D6' and len(shape) == 2:
            self.get_asymmetric_unit   = self.get_asymmetric_unit_inversion
            self.get_symmetry_partners = self.get_symmetry_partners_inversion_2D
            self.get_asymmetric_unit_mapping =\
            self.get_asymmetric_unit_mapping_inversion_2D

        elif symmetry == 'P1':
            self.get_asymmetric_unit   = self.get_asymmetric_unit_P1
            self.get_symmetry_partners = self.get_symmetry_partners_P1
            self.get_asymmetric_unit_mapping =\
            self.get_asymmetric_unit_mapping_P1

    def get_asymmetric_unit_P1(self):
        return self.n

    def get_asymmetric_unit_inversion_2D(self):
        """
        return the set of raveled indices inside
        the asymmetric unit

          -4-3-2-1 0 1 2 3 (i-i0)
        -4 - - - - - - - -
        -3 - - - - - 0 0 0
        -2 - - - - - 0 0 0
        -1 - - - - - 0 0 0
         0 - - - - - 0 0 0
         1 - - - - 0 0 0 0
         2 - - - - 0 0 0 0
         3 - - - - 0 0 0 0
        (j-j0)
        - asymmetric unit
        0 outside asymmetric unit
          (related by symmetry to other values)
        """
        i, j, i0 = self.i, self.j, self.i0
        m = (i <= i0)
        m[j == 0] = True
        m[(j > i0) * (i == i0)] = False
        return self.n[m]

    def get_asymmetric_unit_inversion_3D(self):
        m = self.i <= self.i0
        m[self.i0, self.i0+1:] = False
        m[self.i0, :, self.i0+1:] = False
        return self.n[m]

    def get_asymmetric_unit_inversion(self):
        return self.n[self.i <= self.i0]

    def get_asymmetric_unit_D6_3D(self):
        i0 = self.i0
        return self.n[(self.i <= i0) * (self.j <= i0) * (self.k <= i0)]

    def get_asymmetric_unit_C6_3D(self):
        i0 = self.i0
        return self.n[(self.j <= i0) * (self.k <= i0)]

    def get_symmetry_partners_C6_3D(self, n):
        """
        pixel coordinates: (i, j, k)
        real  coordinates: (x, y, z) = (i-i0, j-i0, k-i0)
        flattened coord  : (i, j, k) = (n / (N*N), n / N, n % N)
        inversion:
            assymetryic unit = x <= 0
                               i <= i0
                               n <= i0 * N^2
            mapping:           x2 = -x
                               i2 = -i + 2 i0
                               n2 = -n + 2 N^2 * i0

        C6 (2-fold about z + inversion):
            assymetryic unit = y, z <= 0
                               j, k <= i0
            mapping:           x, y, z  = (+-x, +-y, +-z)
                               i2 = -i + 2 * i0
                               j2 = -j + 2 * i0
                               k2 = -k + 2 * i0
        """
        nout = []
        i, j, k = self.i[n], self.j[n], self.k[n]

        im = -i + 2 * self.i0
        jm = -j + 2 * self.i0
        km = -k + 2 * self.i0

        for (i2, j2, k2) in [(i, j, k), (im, jm, k), (im, jm, km), (i, j, km)]:
            if (i2 >= 0) and (i2 < self.N) and (j2 >= 0) and (j2 < self.N) and (k2 >= 0) and (k2 < self.N):
                nout.append( i2 * self.N**2 + j2 * self.N + k2 )
        return np.unique(nout)

    def get_symmetry_partners_D6_3D(self, n):
        """
        pixel coordinates: (i, j, k)
        real  coordinates: (x, y, z) = (i-i0, j-i0, k-i0)
        flattened coord  : (i, j, k) = (n / (N*N), n / N, n % N)
        inversion:
            assymetryic unit = x <= 0
                               i <= i0
                               n <= i0 * N^2
            mapping:           x2 = -x
                               i2 = -i + 2 i0
                               n2 = -n + 2 N^2 * i0

        D6 (2-fold about x + 2-fold about z + inversion):
            assymetryic unit = x, y, z <= 0
                               i, j, k <= i0
            mapping:           x, y, z  = (+-x, +-y, +-z)
                               i2 = -i + 2 * i0
                               j2 = -j + 2 * i0
                               k2 = -k + 2 * i0
        """
        nout = []
        i, j, k = self.i[n], self.j[n], self.k[n]

        for i2 in [i, -i + 2 * self.i0]:
            for j2 in [j, -j + 2 * self.i0]:
                for k2 in [k, -k + 2 * self.i0]:
                    if (i2 >= 0) and (i2 < self.N) and (j2 >= 0) and (j2 < self.N) and (k2 >= 0) and (k2 < self.N):
                        nout.append( i2 * self.N**2 + j2 * self.N + k2 )
        return np.unique(nout)

    def get_symmetry_partners_P1(self, n):
        return [n]

    def get_symmetry_partners_inversion_3D(self, n):
        nout = []
        i, j, k = self.i[n], self.j[n], self.k[n]
        i0 = self.i0

        for (i2, j2, k2) in [(i, j, k), (-i + 2 * i0, -j + 2 * i0, -k + 2 * i0)]:
            if (i2 >= 0) and (i2 < self.N) and (j2 >= 0) and (j2 < self.N) and (k2 >= 0) and (k2 < self.N):
                nout.append( i2 * self.N**2 + j2 * self.N + k2)
        return np.unique(nout)

    def get_symmetry_partners_inversion_2D(self, n):
        nout = []
        i, j = self.i[n], self.j[n]
        i0 = self.i0

        for (i2, j2) in [(i, j), (-i + 2 * i0, -j + 2 * i0)]:
            if (i2 >= 0) and (i2 < self.N) and (j2 >= 0) and (j2 < self.N):
                nout.append( i2 * self.N + j2)
        return np.unique(nout)

    def get_asymmetric_unit_mapping_inversion_2D(self):
        """
        n: raveled model voxel location
        m: raveled asymmetric unit voxel location
        n = n_asy[m]
        m = M_n[n]
        """
        n_asy = self.get_asymmetric_unit()
        m_asy = np.arange(n_asy.size)
        M_n = np.zeros(self.n.size, dtype=int)
        i0 = self.i0
        N = self.N
        i = self.i[n_asy]
        j = self.j[n_asy]

        for (i2, j2) in [(i, j), (-i + 2 * i0, -j + 2 * i0)]:
            mask = (i2 < N) * (j2 < N)
            n = i2[mask] * N + j2[mask]
            M_n[n] = m_asy[mask]
        return M_n

    def get_asymmetric_unit_mapping_C6_3D(self):
        """
        n: raveled model voxel location
        m: raveled asymmetric unit voxel location
        n = n_asy[m]
        m = M_n[n]
        """
        n_asy = self.get_asymmetric_unit()
        m_asy = np.arange(n_asy.size)
        M_n = np.zeros(self.n.size, dtype=int)
        i0 = self.i0
        N = self.N
        i = self.i[n_asy]
        j = self.j[n_asy]
        k = self.k[n_asy]

        im = -i + 2 * self.i0
        jm = -j + 2 * self.i0
        km = -k + 2 * self.i0

        for (i2, j2, k2) in [(i, j, k), (im, jm, k), (im, jm, km), (i, j, km)]:
            mask = (i2 < N) * (j2 < N) * (k2 < N)
            n = i2[mask] * N**2 + j2[mask] * N + k2[mask]
            M_n[n] = m_asy[mask]
        return M_n

    def get_asymmetric_unit_mapping_D6_3D(self):
        """
        n: raveled model voxel location
        m: raveled asymmetric unit voxel location
        n = n_asy[m]
        m = M_n[n]
        """
        n_asy = self.get_asymmetric_unit()
        m_asy = np.arange(n_asy.size)
        M_n = np.zeros(self.n.size, dtype=int)
        i0 = self.i0
        N = self.N
        i = self.i[n_asy]
        j = self.j[n_asy]
        k = self.k[n_asy]

        for i2 in [i, -i + 2 * i0]:
            for j2 in [j, -j + 2 * i0]:
                for k2 in [k, -k + 2 * i0]:
                    mask = (i2 < N) * (j2 < N) * (k2 < N)
                    n = i2[mask] * N**2 + j2[mask] * N + k2[mask]
                    M_n[n] = m_asy[mask]
        return M_n

    def get_asymmetric_unit_mapping_inversion_3D(self):
        """
        n: raveled model voxel location
        m: raveled asymmetric unit voxel location
        n = n_asy[m]
        m = M_n[n]
        """
        n_asy = self.get_asymmetric_unit()
        m_asy = np.arange(n_asy.size)
        M_n = np.zeros(self.n.size, dtype=int)
        i0 = self.i0
        N = self.N
        i = self.i[n_asy]
        j = self.j[n_asy]
        k = self.k[n_asy]

        for (i2, j2, k2) in [(i, j, k), (-i + 2 * i0, -j + 2 * i0, -k + 2 * i0)]:
            mask = (i2 < N) * (j2 < N) * (k2 < N)
            n = i2[mask] * N**2 + j2[mask] * N + k2[mask]
            M_n[n] = m_asy[mask]
        return M_n


    def get_asymmetric_unit_mapping_P1(self):
        """
        n: raveled model voxel location
        m: raveled asymmetric unit voxel location
        n = n_asy[m]
        m = M_n[n]
        """
        return self.n


class Symmetry():
    def __init__(self, i0, shape, symmetry = 'inversion'):
        """
        symmetry = 'P1', 'inversion', 'C6' or 'D6'

        pixel coordinates: (i, j, k)
        real  coordinates: (x, y, z) = (i-i0, j-i0, k-i0)
        flattened coord  : (i, j, k) = (n / (N*N), n / N, n % N)
        inversion:
            assymetryic unit = x <= 0
                               i <= i0
                               n <= i0 * N^2
            mapping:           x2 = -x
                               i2 = -i + 2 i0
                               n2 = -n + 2 N^2 * i0
        """
        self.n = np.arange(np.prod(shape))

        self.N = N = shape[0]

        self.i0 = i0

        if len(shape) == 3:
            self.i = i = self.n // (shape[1] * shape[2])
            self.j = j = self.n // shape[2] % shape[1]
            self.k = k = self.n % shape[2]
            im = (-i + 2 * self.i0) % N
            jm = (-j + 2 * self.i0) % N
            km = (-k + 2 * self.i0) % N

            I = self.n.copy()
            inv = im * N**2 + jm * N + km
            P2z = im * N**2 + jm * N + k
            P2x = i * N**2 + jm * N + km
            P2y = im * N**2 + j * N + km

        elif len(shape) == 2:
            self.i = i = self.n // shape[1]
            self.j = j = self.n  % shape[1]
            im = (-i + 2 * self.i0) % N
            jm = (-j + 2 * self.i0) % N

            I = self.n.copy()
            inv = im * N + jm
            P2z = inv

        # only cubes for now
        assert(np.allclose(shape, self.N))
        assert(symmetry in ['P1', 'C6', 'D6', 'inversion'])
        assert(len(shape) in [2, 3])

        sym_ops = [I]
        if symmetry == 'inversion':
            sym_ops.append(inv)

        elif symmetry == 'D6':
            sym_ops.append(P2z)
            sym_ops.append(P2x)
            sym_ops.append(inv)

        elif symmetry == 'C6':
            sym_ops.append(P2z)
            sym_ops.append(inv)

        elif symmetry == 'P1':
            sym_ops.append(I)

        self.sym_ops = sym_ops

    def get_symmetry_partners(self, n):
        out = [self.sym_ops[0][n]]

        # apply cumulatively
        for op in self.sym_ops[1:]:
            N = len(out)
            for i in range(N):
                out.append(op[out[i]])

        return set(out)

    def get_asymmetric_unit(self):
        """
        loop over volume removing voxels
        related to others by symmetry
        """
        n = self.n
        m = np.ones(n.shape, dtype=bool)
        for i in range(n.size):
            if not m[i]: # not is much faster than ~m and m == False
                continue
            ns = self.get_symmetry_partners(i)
            ns.remove(i)
            if len(ns) > 0:
                m[list(ns)] = False
        return n[m]

    def get_asymmetric_unit_mapping(self):
        """
        n: raveled model voxel location
        m: raveled asymmetric unit voxel location
        n = n_asy[m]
        m = M_n[n]
        """
        N_m = self.get_asymmetric_unit()
        M_n = np.zeros(self.n.size, dtype=int)
        for m in range(N_m.size):
            n = N_m[m]
            ns = self.get_symmetry_partners(n)
            for n in ns:
                M_n[n] = m
        return M_n

    def apply_symmetry(self, ar, normalise=True):
        out = np.zeros(ar.size, dtype=ar.dtype)
        out[:] = ar.ravel()

        # apply cumulatively
        for op in self.sym_ops[1:]:
            out += out[op]

        return out.reshape(ar.shape)







def apply_symmetry_2D(ar, symmetry, i0):
    x, y = np.indices(ar.shape)
    x -= i0
    y -= i0
    if symmetry in ['D6', 'inversion'] :
        x2, y2 = -x + i0, -y + i0
        m  = (x2 >= 0) * (x2 < ar.shape[0])
        m *= (y2 >= 0) * (y2 < ar.shape[1])
        ar[m] += ar[x2[m], y2[m]]
    return ar

def apply_symmetry_3D(ar, symmetry, i0):
    x, y, z = np.indices(ar.shape)
    x -= i0
    y -= i0
    z -= i0
    # 2-fold axis about z: x -> -x, y -> -y
    if symmetry in ['D6', 'C6'] :
        x2, y2, z2 = -x + i0, -y + i0, z + i0
        m  = (x2 >= 0) * (x2 < ar.shape[0])
        m *= (y2 >= 0) * (y2 < ar.shape[1])
        m *= (z2 >= 0) * (z2 < ar.shape[2])
        ar[m] += ar[x2[m], y2[m], z2[m]]

    # 2-fold axis about x: y -> -y, z -> -z
    if symmetry in ['D6'] :
        x2, y2, z2 = x + i0, -y + i0, -z + i0
        m  = (x2 >= 0) * (x2 < ar.shape[0])
        m *= (y2 >= 0) * (y2 < ar.shape[1])
        m *= (z2 >= 0) * (z2 < ar.shape[2])
        ar[m] += ar[x2[m], y2[m], z2[m]]

    if symmetry in ['D6', 'C6', 'inversion'] :
        x2, y2, z2 = -x + i0, -y + i0, -z + i0
        m  = (x2 >= 0) * (x2 < ar.shape[0])
        m *= (y2 >= 0) * (y2 < ar.shape[1])
        m *= (z2 >= 0) * (z2 < ar.shape[2])
        ar[m] += ar[x2[m], y2[m], z2[m]]

    return ar


def apply_symmetry(ar, symmetry, i0):
    """
    here just apply the symmetry axes
    that exactly map pixels to pixels
    the rest are done in M_ri
    """
    if symmetry == 'P1':
        return ar

    if ar.ndim == 2 :
        ar = apply_symmetry_2D(ar, symmetry, i0)
    elif ar.ndim == 3 :
        ar = apply_symmetry_3D(ar, symmetry, i0)
    else :
        raise ValueError(f'dimension {ar.ndim} not supported')

    return ar

def get_non_voxel_operators(dimensions, symmetry):

    if symmetry == 'P1' or symmetry == 'inversion':
        if dimensions == 2 :
            return np.array([[[1, 0], [0, 1]]])
        elif dimensions == 3:
            return np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        else :
            raise ValueError(f'dimension {dimensions} not supported for symmetry {symmetry}')

    elif symmetry == 'D6' or symmetry == 'C6':
        if dimensions == 3:
            # 3 x pi / 3 rotations about z-axis
            #coord.x = x * c - y * s;
            #coord.y = x * s + y * c;
            c = 0.5;
            s = 0.8660254037844386;
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            return np.array([Rz, Rz.dot(Rz), Rz.dot(Rz.dot(Rz))])
        else :
            raise ValueError(f'dimension {dimensions} not supported for symmetry {symmetry}')
