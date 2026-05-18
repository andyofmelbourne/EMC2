from .refdata import cromer_mann_params
import pyopencl as cl
import prody
import numpy as np
import h5py


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

        # elements (str)
        # ------------------
        if elements is None:
            print('Warning! elements is None, setting all atoms to Hydrogen')
            elements = ['H' for i in range(N)]

        # number of atomic species
        elements_unique, unique_inds, inds, element_counts = np.unique(elements, return_index=True, return_inverse=True, return_counts=True)
        Nelements = len(elements_unique)
        element_counts = np.array(element_counts, dtype=np.int32)
        self.element_counts = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = element_counts)

        # indices for coords etc
        inds_sorted = np.argsort(inds)

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

        t[:, :3] = xyz[inds_sorted]
        self.xyz = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = t)

        # occupancy factors
        # -----------------
        t = np.ones((N,), dtype=np.float32)
        if occ is not None:
            assert (len(occ) == N)
            t[:] = occ[inds_sorted]
        self.occ = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = t)

        # B-factors (temperature)
        # only one B-factor per element type
        # this is penalty for faster code
        # ----------------------------------
        B = np.ones((Nelements,), dtype=np.float32)
        if B_factors is not None:
            assert (len(B_factors) == N)
            B[:] = B_factors[unique_inds]
        self.B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = B)

        # Cromer Mann params
        # this is for unique elements only
        # not per atom
        # --------------------------------
        crom_B, crom_w = elements_to_cromer_mann_params(elements_unique)
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
        self.Nelements = np.int32(Nelements)

        print(f'\n {B=}')
        print(f'\n {xyz[inds_sorted]=}')
        print(f'\n {element_counts=}')
        print(f'\n {occ[inds_sorted]=}')

    def fill_q(self, q):
        if q is not None:
            self.q_cpu[..., :3] = q
            cl.enqueue_copy(self.queue, self.q, self.q_cpu)

    def __call__(self, q=None):
        self.fill_q(q)

        cl.Kernel(self.prg, 'calculate_q_density')(
            self.queue, (len(self.q_cpu),), None,
            self.xyz, self.q, self.occ, self.B, self.crom_B, self.crom_w,
            self.element_counts,
            self.F, self.Nelements
        )
        cl.enqueue_copy(self.queue, self.F_cpu, self.F)
        self.queue.finish()

        F = self.F_cpu[:, 0] + 1j * self.F_cpu[:, 1]
        return F


    def code(self):
        render_kernel = \
        r"""
        // hat(phi_e)(q) = o e^{-2\pi i r_n \cdot q} \sum_{k=1}^5 a_k  e^{-(b_k + B) q^2 / 4}

        // loop over element types
        // xyz -> sorted by atom type
        // same for others
        __kernel void calculate_q_density(
            __global const float4  *xyz,
            __global const float4  *qs,
            __global const float  *occupancy,
            __global const float  *bfactors,
            __global const float  *crom_B,
            __global const float  *crom_w,
            __global const int  *element_counts,
            __global float2  *out,
            const int  Nelements
            )
        {
            float rdotq;

            float occ, BB, B;
            int i, j, el_index;

            // assume one work item per work group
            int n = get_global_id(0);

            // get my q coordinate
            float4 q = qs[n];
            float q2 = dot(q, q);

            double2 ramp;

            // need double b/c lots of addition
            double2 F = (double2)(0.0);
            float t;

            int atom_index = 0;

            for (el_index=0; el_index<Nelements; el_index++){
                // B-factor
                // cannot vary with position
                B = bfactors[el_index];

                // calculate profile at q for this element type
                t = 0.0f;
                for (j=0; j<5; j++){
                    // cromer mann coefs (gausian widths)
                    BB = B + crom_B[5 * el_index + j];
                    t += crom_w[5 * el_index + j] * exp(-BB * q2 / 4.);
                }

                // loop over atoms
                ramp.x = 0.0;
                ramp.y = 0.0;
                for (i=0; i<element_counts[el_index]; i++){

                    // atom coordinate
                    rdotq = dot(xyz[atom_index], q);

                    // occupancy
                    occ = occupancy[atom_index];


                    // hat(phi_e)(q) = o e^{-2\pi i r_n \cdot q} \sum_{k=1}^5 a_k  e^{-(b_k + B) q^2 / 4}

                    // phase factor
                    ramp.x +=  occ * cos(2 * M_PI_F * rdotq);
                    ramp.y += -occ * sin(2 * M_PI_F * rdotq);

                    atom_index += 1;
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


if __name__ == '__main__':
    #from .. import utils_cl
    from emc3 import utils_cl
    import time
    import sys
    import pickle

    pdb = sys.argv[1]+'.pdb'

    cl_stuff = utils_cl.opencl_init()

    N = 192
    #N = 512
    i, j, k = np.indices((N, N, N))

    # inverse Angstroms
    dq = 0.000270731 * 1e10

    out = sys.argv[1]+f'_dq_{dq}_N_{N}.pickle'

    i0 = i.shape[0]//2
    i = dq * (i - i0)
    j = dq * (j - i0)
    k = dq * (k - i0)
    qs = np.zeros((i.size, 3), dtype=np.float32)
    qs[:, 0] = i.ravel()
    qs[:, 1] = j.ravel()
    qs[:, 2] = k.ravel()

    i_calc = Density_pdb(
        cl_stuff['queue'],
        cl_stuff['context'],
        qs,
        pdb,
        biomol=True
    )

    t0 = time.time()
    F = i_calc().reshape(i.shape)
    print(f'time 1: {time.time() - t0:.2e}')

    t = {
        'electron_density_fourier': F,
        'dq': dq,
    }
    #with h5py.File(out, 'w') as f:
    #    f.create_dataset('electron_density_fourier', data=F, chunks=F.shape, compression='gzip')
    #    f['dq'] = dq

    pickle.dump(t, open(out, 'wb'))


    """
    from emc3.fsim.density_pdb import Density_pdb as Density_pdb2
    i_calc2 = Density_pdb2(
        cl_stuff['queue'],
        cl_stuff['context'],
        qs,
        '2gtl',
        biomol=True
    )

    t0 = time.time()
    F = i_calc2().reshape(i.shape)
    print(f'time 2: {time.time() - t0:.2e}')
    """
