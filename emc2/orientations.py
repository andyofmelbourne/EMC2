import pyopencl as cl
import pyopencl.array 
import numpy as np
import tqdm

quiet = True

# perhaps opencl image has each pixel coordinate at 0.5, 1.5 ... (N-1) + 0.5
# yes thats it # the coordinates are also transposed

def get_rotation_matrices(queue = None, context = None, rotation_order = 10, dimensions = 3):
    if dimensions == 3 and rotation_order > 0 :
        R_cl = get_rotations_3D(rotation_order, queue, context)
    
    elif dimensions == 2 and rotation_order > 0 :
        R_cl = get_rotations_2D(rotation_order, queue, context)
    
    elif dimensions == 2 and rotation_order == 0 :
        R_cl = cl.array.empty(queue, (1, 2, 2), dtype = np.float32)
        R = np.array([[[1, 0], [0, 1]]], dtype = np.float32)
        cl.enqueue_copy(queue, R_cl.data, R)
    
    else :
        raise ValueError(f'could not reconsile dimension {dimensions} and rotation_order {rotation_order}')
    
    return R_cl

def calculate_rotation_matrices(Mrot, M_in_plane, M_sphere, queue, context):
    cl_code = cl.Program(context, r"""
        // R = Rz(theta).dot(Ry(phi).dot(Rz(phi2)))
        //[-sin(phi_2)*sin(theta) + cos(phi)*cos(phi_2)*cos(theta), -sin(phi_2)*cos(phi)*cos(theta) - sin(theta)*cos(phi_2), sin(phi)*cos(theta)],
        //[ sin(phi_2)*cos(theta) + sin(theta)*cos(phi)*cos(phi_2), -sin(phi_2)*sin(theta)*cos(phi) + cos(phi_2)*cos(theta), sin(phi)*sin(theta)],
        //[                                   -sin(phi)*cos(phi_2),                                     sin(phi)*sin(phi_2),            cos(phi)]])
        
        __kernel void calculate_rotation_matrix ( global float *R, const int M_in_plane, const int M_sphere)
        {
        // this is the rotation index r
        int r = get_global_id(0);
        
        int n = r / M_in_plane ;
        int m = r % M_in_plane ;
        
        float i = (float)n + 0.5;
        float j = (float)m + 0.5;
        float phi  = acos(1 - 2 * i / M_sphere);
        float phi_2 = 2. * M_PI * j / M_in_plane; 
        float goldenRatio = (1. + sqrt(5.)) / 2.;
        float theta = 2. * M_PI * i / goldenRatio;
        
        float Rl[9];
        
        Rl[0] = -sin(phi_2) * sin(theta) + cos(phi) * cos(phi_2) * cos(theta) ;
        Rl[1] = -sin(phi_2)*cos(phi)*cos(theta) - sin(theta)*cos(phi_2) ;
        Rl[2] = sin(phi)*cos(theta) ;
        
        Rl[3] = sin(phi_2)*cos(theta) + sin(theta)*cos(phi)*cos(phi_2) ;
        Rl[4] = -sin(phi_2)*sin(theta)*cos(phi) + cos(phi_2)*cos(theta) ;
        Rl[5] = sin(phi)*sin(theta) ;
        
        Rl[6] = -sin(phi)*cos(phi_2) ;
        Rl[7] = sin(phi)*sin(phi_2) ;
        Rl[8] = cos(phi) ;
        
        // apply orientation offset to keep the poles away from 
        // any symmetry axes that might be applied
        float Ro[9]; 
        phi   = 1.2 * M_PI / 4.;
        phi_2 = 1.3660358560403127 * M_PI;
        theta = 1.4316602255033495 * M_PI;
        
        Ro[0] = -sin(phi_2) * sin(theta) + cos(phi) * cos(phi_2) * cos(theta) ;
        Ro[1] = -sin(phi_2)*cos(phi)*cos(theta) - sin(theta)*cos(phi_2) ;
        Ro[2] = sin(phi)*cos(theta) ;
        
        Ro[3] = sin(phi_2)*cos(theta) + sin(theta)*cos(phi)*cos(phi_2) ;
        Ro[4] = -sin(phi_2)*sin(theta)*cos(phi) + cos(phi_2)*cos(theta) ;
        Ro[5] = sin(phi)*sin(theta) ;
        
        Ro[6] = -sin(phi)*cos(phi_2) ;
        Ro[7] = sin(phi)*sin(phi_2) ;
        Ro[8] = cos(phi) ;
        
        R[r * 9 + 0] = Rl[0] * Ro[0] + Rl[1] * Ro[3] + Rl[2] * Ro[6];
        R[r * 9 + 1] = Rl[0] * Ro[1] + Rl[1] * Ro[4] + Rl[2] * Ro[7];
        R[r * 9 + 2] = Rl[0] * Ro[2] + Rl[1] * Ro[5] + Rl[2] * Ro[8];
        
        R[r * 9 + 3] = Rl[3] * Ro[0] + Rl[4] * Ro[3] + Rl[5] * Ro[6];
        R[r * 9 + 4] = Rl[3] * Ro[1] + Rl[4] * Ro[4] + Rl[5] * Ro[7];
        R[r * 9 + 5] = Rl[3] * Ro[2] + Rl[4] * Ro[5] + Rl[5] * Ro[8];
        
        R[r * 9 + 6] = Rl[6] * Ro[0] + Rl[7] * Ro[3] + Rl[8] * Ro[6];
        R[r * 9 + 7] = Rl[6] * Ro[1] + Rl[7] * Ro[4] + Rl[8] * Ro[7];
        R[r * 9 + 8] = Rl[6] * Ro[2] + Rl[7] * Ro[5] + Rl[8] * Ro[8];
        }

    """).build()
    
    # pre-calculate rotations
    # -----------------------
    R_cl  = cl.array.empty(queue, (Mrot, 3, 3), dtype = np.float32)
    for t in tqdm.tqdm(range(1), desc='pre-calculating rotation matrices', disable = quiet):
        cl_code.calculate_rotation_matrix(queue, (Mrot,), None, R_cl.data, np.int32(M_in_plane), np.int32(M_sphere))
    
    #R = R_cl.get().reshape((-1, 3, 3)).astype(np.float32)
    return R_cl

def calculate_rotation_matrices_2D(Mrot, queue, context):
    cl_code = cl.Program(context, r"""
        // R = Rz(theta)
        // [c, -s]
        // [s,  c]
        
        __kernel void calculate_rotation_matrix ( global float *R, const int M_in_plane)
        {
        // this is the rotation index r
        int r = get_global_id(0);
        
        float theta = 2. * M_PI * r / M_in_plane;
        
        float c = cos(theta);
        float s = sin(theta);
        
        R[r * 4 + 0] = c;
        R[r * 4 + 1] = -s;
        R[r * 4 + 2] = s;
        R[r * 4 + 3] = c;
        }
    """).build()
    
    # pre-calculate rotations
    # -----------------------
    R_cl  = cl.array.empty(queue, (Mrot, 2, 2), dtype = np.float32)
    for t in tqdm.tqdm(range(1), desc='pre-calculating rotation matrices', disable = quiet):
        cl_code.calculate_rotation_matrix(queue, (Mrot,), None, R_cl.data, np.int32(Mrot))
    
    #R = R_cl.get().reshape((-1, 2, 2)).astype(np.float32)
    return R_cl


def calculate_rotation_matrices_2D_cpu(M):
    # theta[r] = 2 pi r / M_in_plane
    # R[r]     = [cos -sin]
    #            |sin  cos|
    M_in_plane = round_up_to_odd(np.pi * M)
        
    t = 2 * np.pi * np.arange(M_in_plane) / M_in_plane
    R = np.empty((M_in_plane, 2, 2), dtype = np.float32)
    R[:, 0, 0] =  np.cos(t)
    R[:, 0, 1] = -np.sin(t)
    R[:, 1, 0] =  np.sin(t)
    R[:, 1, 1] =  np.cos(t)
    return R


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def get_rotations_3D(M, queue, context):
    # we want this to be odd (so that inversion symmetry is more useful)
    #M_in_plane = int(np.pi * M)+1
    M_in_plane = round_up_to_odd(np.pi * M)
    
    M_sphere   = int(np.pi * M**2)+1
    Mrot = M_in_plane * M_sphere
        
    R_cl = calculate_rotation_matrices(Mrot, M_in_plane, M_sphere, queue, context)
    return R_cl

def get_rotations_2D(M, queue, context):
    # we want this to be odd (so that inversion symmetry is more useful)
    #M_in_plane = int(np.pi * M)+1
    M_in_plane = round_up_to_odd(np.pi * M)
    
    R_cl = calculate_rotation_matrices_2D(M_in_plane, queue, context)
    return R_cl

