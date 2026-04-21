from .refdata import cromer_mann_params
import pyopencl as cl
import prody
import numpy as np


def parse_pdb(pdb_fnam, biomol=True):
    pdb = prody.parsePDB(pdb_fnam, biomol=biomol)
    # get the atomic coordinates (N, 3) --> (atom no., x/y/z in Ang)
    xyz = pdb.getCoords()
    occ = pdb.getOccupancies()
    B   = pdb.getBetas()
    names = pdb.getElements()
    return xyz, occ, B, names

def elements_to_cromer_mann_params(elements):
    crom_B = np.zeros((len(elements), 5), dtype=np.float32)
    crom_w = np.zeros((len(elements), 5), dtype=np.float32)
    for i, name in enumerate(elements):
        params = cromer_mann_params[name]
        crom_B[i, 1:] = params[4: -1]
        crom_w[i, 0] = params[8]
        crom_w[i, 1:] = params[:4]

    return crom_B, crom_w



# atomic coords + q -> Fourier amplitudes
class Density_atoms():
    def __init__(self, queue, context, q, xyz, occ=None, B_factors=None, elements=None):
        """
        xyz = list of atomic coordinates in Angstroms
        """
        self.context = context
        self.queue = queue
        N = len(xyz)
        mf = cl.mem_flags

        # q-vectors (float4)
        # ------------------
        assert (q.shape[1] == 3)
        self.q_cpu = np.zeros((len(q), 4), dtype=np.float32)
        self.q_cpu[:, :3] = q
        self.q = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.q_cpu)

        # output: fourier amplitudes (float2)
        # -----------------------------------
        self.F_cpu = np.empty((len(q), 2), dtype=np.float32)
        self.F = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.F_cpu)

        # atomic coordinates (float4)
        # ---------------------------
        assert (xyz.shape[1] == 3)
        t = np.zeros((N, 4), dtype=np.float32)
        t[:, :3] = xyz
        self.xyz = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = t)

        # occupancy factors
        # -----------------
        t = np.ones((N,), dtype=np.float32)
        if occ is not None:
            assert (len(occ) == N)
            t[:] = occ
        self.occ = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = t)

        # B-factors (temperature)
        # -----------------------
        B = np.ones((N,), dtype=np.float32)
        if B_factors is not None:
            assert (len(B_factors) == N)
            B[:] = B_factors
        self.B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = B)

        # Cromer Mann params
        # -----------------------
        if elements is None:
            print('Warning! elements is None, setting all atoms to Hydrogen')
            elements = ['H' for i in range(N)]

        crom_B, crom_w = elements_to_cromer_mann_params(elements)
        self.crom_B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = crom_B)
        self.crom_w = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = crom_w)

        # compile openCL kernel
        # ---------------------
        self.prg = cl.Program(
            context,
            self.code()
        ).build()
        self.queue = queue
        self.context = context
        self.N = np.int32(N)

    def fill_q(self, q):
        if q is not None:
            self.q_cpu[..., :3] = q
            cl.enqueue_copy(self.queue, self.q, self.q_cpu)

    def __call__(self, q=None):
        self.fill_q(q)

        cl.Kernel(self.prg, 'calculate_q_density')(
            self.queue, (len(self.q_cpu),), None,
            self.xyz, self.q, self.occ, self.B, self.crom_B, self.crom_w, self.F, self.N
        )
        cl.enqueue_copy(self.queue, self.F_cpu, self.F)
        self.queue.finish()

        F = self.F_cpu[:, 0] + 1j * self.F_cpu[:, 1]
        return F


    def code(self):
        render_kernel = \
        r"""
        // hat(phi_e)(q) = o e^{-2\pi i r_n \cdot q} \sum_{k=1}^5 a_k  e^{-(b_k + B) q^2 / 4}

        __kernel void calculate_q_density(
            __global const float4  *xyz,
            __global const float4  *qs,
            __global const float  *occupancy,
            __global const float  *bfactors,
            __global const float  *crom_B,
            __global const float  *crom_w,
            __global float2  *out,
            const int  atoms
            )
        {
            float rdotq;

            float occ, BB, B;
            int i, j;

            // assume one work item per work group
            int n = get_global_id(0);

            // get my q coordinate
            float4 q = qs[n];
            float q2 = dot(q, q);

            float2 ramp;

            // need double b/c lots of addition
            double2 F = (double2)(0.0);
            float t;

            // loop over atoms
            for (i=0; i<atoms; i++){

                // atom coordinate
                rdotq = dot(xyz[i], q);

                // occupancy
                occ = occupancy[i];

                // B-factor
                B = bfactors[i];

                // hat(phi_e)(q) = o e^{-2\pi i r_n \cdot q} \sum_{k=1}^5 a_k  e^{-(b_k + B) q^2 / 4}

                // phase factor
                ramp.x =  occ * cos(2 * M_PI_F * rdotq);
                ramp.y = -occ * sin(2 * M_PI_F * rdotq);

                // loop cromer man coefs
                t = 0.0f;
                for (j=0; j<5; j++){
                    // cromer mann coefs (gausian widths)
                    BB = B + crom_B[5 * i + j];
                    t += crom_w[5 * i + j] * exp(-BB * q2 / 4.);
                }

                F += convert_double2(t * ramp);
            }

            out[n] = convert_float2(F);
        }
        """
        return render_kernel


# pdb + q -> Fourier amplitudes
class Density_pdb():
    def __init__(self, queue, context, q, pdb, biomol=True):
        """
        q (m^-1)
        """
        # pdb -> xyz, occ, B_factors, elements
        xyz, occ, B, elements = parse_pdb(pdb, biomol)

        # q (m^-1 -> A^-1)
        qs = 1e-10 * q

        self.density_atoms = Density_atoms(
            queue, context, qs, xyz, occ, B, elements
        )

    def __call__(self, q=None):
        if q is not None:
            qs = 1e-10 * q
        else:
            qs = q

        F = self.density_atoms(qs)
        return F

