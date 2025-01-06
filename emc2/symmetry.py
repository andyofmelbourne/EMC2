import numpy as np

class Symmetry():

    def __init__(self, i0, shape, symmetry = 'inversion'):
        """
        symmetry = 'inversion' or 'D6'
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
        assert(symmetry in ['inversion', 'D6'])
        assert(len(shape) in [2, 3])

        if   symmetry == 'inversion' and len(shape) == 3 :
            self.get_asymmetric_unit   = self.get_asymmetric_unit_inversion
            self.get_symmetry_partners = self.get_symmetry_partners_inversion_3D
            
        elif symmetry == 'inversion' and len(shape) == 2 :
            self.get_asymmetric_unit   = self.get_asymmetric_unit_inversion
            self.get_symmetry_partners = self.get_symmetry_partners_inversion_2D
            
        elif symmetry == 'D6' and len(shape) == 3:
            self.get_asymmetric_unit   = self.get_asymmetric_unit_D6_3D
            self.get_symmetry_partners = self.get_symmetry_partners_D6_3D
            
        elif symmetry == 'D6' and len(shape) == 2:
            self.get_asymmetric_unit   = self.get_asymmetric_unit_inversion
            self.get_symmetry_partners = self.get_symmetry_partners_inversion_2D
        
    def get_asymmetric_unit_inversion(self):
        return self.n[self.i <= self.i0]

    def get_asymmetric_unit_D6_3D(self):
        i0 = self.i0
        return self.n[(self.i <= i0) * (self.j <= i0) * (self.k <= i0)]
    
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
    if symmetry in ['D6'] :
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
    
    if symmetry in ['D6', 'inversion'] :
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
