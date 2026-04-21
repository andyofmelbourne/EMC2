import numpy as np
from scipy.spatial import ConvexHull

import pyopencl as cl

def right_handed_simplices(hull, float4 = True, double=False):
    """
    hull is a scipy.spatial ConvexHull object

    re-order hull.simplices (in place) to provide right handed vertex order

    and

    output a vertex matrix of the form:
        verts[f, j, d] = v0_fjd    for j = 0, 1 or 2
        verts[f, 3, d] = normal_fd

    where:
        f: face index
        j: face vertex index (<3 for simplices)
        d: dimension index (x, y or z)

    if float4 is True then a dummy "w" row is added to the output
    vertices for compatibility with opencl
    """
    if double:
        dtype = np.float64
    else:
        dtype = np.float32

    if float4:
        verts = np.empty((len(hull.simplices),) + (4, 4), dtype=dtype)
    else:
        verts = np.empty((len(hull.simplices),) + (4, 3), dtype=dtype)

    for i, simplex in enumerate(hull.simplices):
        # vertices of triangular face
        v1, v2, v3 = hull.points[simplex]

        # Face normal (n) and area (A)
        n0 = np.cross(v2-v1, v3-v1)
        A = np.linalg.norm(n0) / 2.
        unit_normal = hull.equations[i, :3]
        n = A * unit_normal

        verts[i, 0, :3] = v1

        # unit_normal away from object centre (good)
        # use it to re-order verts

        if np.dot(n0, unit_normal) < 0:
            #print()
            #print(f're-ordering simplex {i}: {v1, v2, v3} -> {v1, v3, v2}')
            #print(f'normal: {n0} --> {unit_normal}')
            verts[i, 1, :3] = v3
            verts[i, 2, :3] = v2

            # also re-order simplex indexing just in case hull.simplices
            # is used again
            simplex[1], simplex[2] = simplex[2], simplex[1]
        else:
            verts[i, 1, :3] = v2
            verts[i, 2, :3] = v3

        verts[i, 3, :3] = n
    return verts


class Fourier_transform_convex_polyhedron_cl():
    kernel_code = """
    // Helper function to sort 3 floats in-place
    void sort(float *a, float *b, float *c) {{
        // Using a simple bubble sort logic for 3 elements
        #define SWAP(x, y) {{ float t=x; x=y; y=t; }}

        if (*a > *b) SWAP(*a, *b);
        if (*a > *c) SWAP(*a, *c);
        if (*b > *c) SWAP(*b, *c);
    }}


    float2 fab_calc(
        private float a,
        private float b
    ) {{
        float step = fabs(b - a);
        float x, t, fac;
        float2 fab;

        if (step < 0.001f) {{
            x = (a + b) / 2.0f;
            fac = 1.0f + (a - x) * (b-x) / 6.0f;

            fab.x = - sin(x) * fac;
            fab.y = - cos(x) * fac;

        }} else {{
            // fab = (e^{{-i b}} - e^{{-ia}}) / (b-a)
            // fab = (cos(b) - i sin(b) - cos(a) + i sin(a)) / (b-a)
            fac = b - a;
            fab.x = (cos(b) - cos(a)) / fac;
            fab.y = (sin(a) - sin(b)) / fac;
        }}

        return fab;

    }}

    float2 fabc_calc(
        private float a,
        private float b,
        private float c
    ) {{
        sort(&a, &b, &c);

        //float step = c - a;
        float step = c - a;
        float x, fac;
        float2 fab, fbc, fabc;

        if (step < 0.001f) {{
            x = (a + b + c) / 3.0f;
            fac = (a-x) * (a-x) + (b-x) * (b-x) + (c-x) * (c-x)
                + (a-x) * (b-x) + (a-x) * (c-x) + (b-x) * (c-x);

            fac = -0.50f * (1.0f - fac / 12.0f);
            fabc.y = -fac * sin(x);
            fabc.x = fac * cos(x);

        }} else {{
            fab = fab_calc(a, b);
            fbc = fab_calc(b, c);

            fabc = (fbc - fab) / step;
        }}

        return fabc;
    }}

    // I = |F|^2
    // nabla I = 2 Re{{ (nabla F) F* }}
    __kernel void I_poly(
        __global float4 *qin,
        __global float4 *vin,
        __global float  *I,
        const int F // faces
    ) {{
        int gid = get_global_id(0);
        int gsize = get_global_size(0);

        float2 out = (float2)(0.0f);
        float2 fabc;
        float4 q = qin[gid];
        float a, b, c, qn;

        for (int f=0; f<F; f++) {{
            // vin = (F, 4, 4)
            // vin = (4 F, 4)
            // v1 = (f, 0, :3)
            // v2 = (f, 1, :3)
            // v3 = (f, 2, :3)
            // n  = (f, 3, :3)
            a = 2.0f * M_PI_F * dot(vin[4 * f + 0], q);
            b = 2.0f * M_PI_F * dot(vin[4 * f + 1], q);
            c = 2.0f * M_PI_F * dot(vin[4 * f + 2], q);
            qn = dot(vin[4 * f + 3], q);

            fabc = fabc_calc(a, b, c);
            out += qn * fabc;
        }}

        // factor of i is not important for intensities
        // also, forget about q==0

        // a = 1 / (pi q2)
        a = M_PI_F * dot(q, q);
        a = select(a, 1.0f, a == 0.0f);

        out    /= a;
        I[gid] = out.x * out.x + out.y * out.y;
    }}
    """

    def __init__(self, context, queue, shape, vertices, q, double=False, scale=False, apply_scale_to_output=False):
        """
        dverts_n = \sum_v M_nv \nabla_v I(q, verts_v)
        """
        if double:
            dtype = np.float64
        else:
            dtype = np.float32
        self.dtype = dtype
        self.double = double

        self.shape = shape
        self.size = np.prod(shape)
        self.vertices = vertices
        self.context = context
        self.queue = queue

        self.q = np.zeros(shape + (4,), dtype=dtype)

        mf = cl.mem_flags
        self.q_cl = cl.Buffer(context, mf.READ_ONLY, self.q.nbytes)

        self.I = np.zeros(shape, dtype=dtype)
        self.I_cl = cl.Buffer(context, mf.READ_WRITE, self.I.nbytes)

        self.prg = None

        self.verts_filled = False
        self.q_filled = False

        # scale q and vertices so that they are both closer to 1
        # q = 10^9
        # v = 10^-9
        # s = (v / q)**0.5 = 10^-9
        # q' = q s
        # v' = v/s
        # (q s) . (v/s) -> q v
        # 1/(q)^3 -> s^3 / (s q)^3
        if scale:
            qrange = q.max() - q.min()
            vrange = vertices.max() - vertices.min()
            scale = (vrange / qrange)**0.5
        else:
            scale = False

        self.scale = scale
        self.apply_scale_to_output = apply_scale_to_output

        self.fill_verts(vertices)
        self.fill_q(q)

        if double:
            code = self.kernel_code.replace('float', 'double')
            code = code.replace('a == 0.0f', '(long)(a == 0.0)')
            code = code.replace('.0f', '.0')
            code = code.replace('_F', '')
        else:
            code = self.kernel_code

        # print(code)

        self.prg = cl.Program(
            context,
            code
        ).build()


    def fill_q(self, q=None):
        if q is None:
            return

        self.q[..., :3] = q

        if self.scale:
            self.q *= self.scale

        self.q_filled = True
        cl.enqueue_copy(self.queue, self.q_cl, self.q)

    def fill_verts(self, vertices=None):
        """
        """
        if vertices is None:
            return

        mf = cl.mem_flags

        if self.scale:
            hull = ConvexHull(vertices / self.scale)
        else:
            hull = ConvexHull(vertices)

        self.verts = right_handed_simplices(hull, double=self.double)
        self.verts_cl = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.verts)

        self.vol = self.dtype(hull.volume)
        self.F = np.int32(len(hull.simplices))

        self.verts_filled = True

        self.vertices_shape = vertices.shape
        self.hull = hull

    def update_verts(self, vertices):
        if vertices is None:
            return

        assert(self.vertices_shape == vertices.shape)

        if self.scale:
            self.hull.points[:] = vertices / self.scale
        else:
            self.hull.points[:] = vertices

        # update normals
        # assume right handed
        for i, simplex in enumerate(self.hull.simplices):
            v1, v2, v3 = self.hull.points[simplex]

            # Face normal (n) and area (A)
            n = np.cross(v2-v1, v3-v1) / 2.

            self.verts[i, 0, :3] = v1
            self.verts[i, 1, :3] = v2
            self.verts[i, 2, :3] = v3
            self.verts[i, 3, :3] = n

        cl.enqueue_copy(self.queue, self.verts_cl, self.verts)

    def __call__(self, q=None, vertices=None):
        """
        out = int e^{} (1/m^3) dV
        unitless
        """
        self.fill_q(q)
        self.update_verts(vertices)

        cl.Kernel(self.prg, 'I_poly')(
            self.queue, (self.size,), None,
            self.q_cl, self.verts_cl, self.I_cl, self.F
        )
        cl.enqueue_copy(self.queue, self.I, self.I_cl)
        self.queue.finish()

        if self.apply_scale_to_output:
            self.I *= self.scale**6

        return self.I


def get_truncated_octahedron(scale, t):
    # s1 and s2 now correctly carry gradient information
    a = scale * (1.0 - t)
    b = scale * t

    verts = np.empty((24, 3), dtype=float)

    # square face at +x
    verts[0] = [ a,  b,  b]
    verts[1] = [ a, -b,  b]
    verts[2] = [ a,  b, -b]
    verts[3] = [ a, -b, -b]

    # square face at -x
    verts[4] = [-a,  b,  b]
    verts[5] = [-a, -b,  b]
    verts[6] = [-a,  b, -b]
    verts[7] = [-a, -b, -b]

    # square face at +y
    verts[8] = [ b,  a,  b]
    verts[9] = [-b,  a,  b]
    verts[10] = [ b,  a, -b]
    verts[11] = [-b,  a, -b]

    # square face at -y
    verts[12] = [ b, -a,  b]
    verts[13] = [-b, -a,  b]
    verts[14] = [ b, -a, -b]
    verts[15] = [-b, -a, -b]

    # square face at +z
    verts[16] = [ b,  b,  a]
    verts[17] = [-b,  b,  a]
    verts[18] = [ b, -b,  a]
    verts[19] = [-b, -b,  a]

    # square face at -z
    verts[20] = [ b,  b, -a]
    verts[21] = [-b,  b, -a]
    verts[22] = [ b, -b, -a]
    verts[23] = [-b, -b, -a]
    return verts


if __name__ == '__main__':
    # OpenCL stuff
    # ------------
    platforms = cl.get_platforms()
    my_platform = platforms[1]
    my_device = my_platform.get_devices()[0]
    ctx = cl.Context([my_device])
    queue = cl.CommandQueue(ctx)
    # ------------


    # set q-vectors and vertices
    # --------------------------
    a = 1.0 # distance from center to vertex
    vertices = np.array([
        [a, 0, 0], [-a, 0, 0],
        [0, a, 0], [0, -a, 0],
        [0, 0, a], [0, 0, -a]
    ])

    X = 50
    N = 1024
    qx = np.fft.fftfreq(N, X/N)
    q = np.zeros((N, N, 3), dtype=float)
    q[:, :, 0] = qx[:, None]
    q[:, :, 1] = qx[None, :]
    # --------------------------


    # calculate I
    # -----------
    poly_calc = Fourier_transform_convex_polyhedron_cl(
        ctx, queue, q.shape[:2], vertices, q
    )
    I = poly_calc()
    # -----------


    # display I
    # ---------
    import pyqtgraph as pg
    ims = np.fft.fftshift(I)
    pg.show(im**0.2)
    pg.exec()
