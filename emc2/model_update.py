import numpy as np
import math
from . import utils 
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    quiet = False
else :
    quiet = True

class Model_update():
    """
    Wsum_r = sum_i C_i W_ri
    
    Basic:
        N_ri  = sum_d P_dr K_di
        D_ri  = C_i sum_d P_dr 
    
    Fluence:
        N_ri  = sum_d P_dr K_di
        D_r   = C_i sum_d w_d P_dr

        a_d   = sum_i K_di
        b_d   = sum_r P_dr Wsum_r
        w'_d  = a_d / b_d
    
    Fluence free:
        N_ri = Wsum_r sum_d P_dr K_di
        D_ri = C_i sum_d P_dr K_d
        
    merge tomograms (then I):
        W'_ri = N_ri / D_ri
     
        A_n = sum_ri M^-1(W'_ri, r, i)_n
        B_n = sum_ri M^-1(1,     r, i)_n
        I_n = A_n / B_n
    
    merge I:
        A_n = sum_ri M^-1(N_ri, r, i)_n
        B_n = sum_ri M^-1(D_ri, r, i)_n
        I_n = A_n / B_n
        
    Loop over model class (this reduces out-of-order memory operations)
    Would be nice to skip frames with low P_dr values but this complicates
    dot product which is about 40 times faster
    
    Perform coordinate mappings on gpu
    nearest: i --> n
        n0 = round(i0 + (R_r . q_i)_0 / dq)
        n1 = round(i0 + (R_r . q_i)_1 / dq)
        n2 = round(i0 + (R_r . q_i)_2 / dq)
        
        n = N^2 n0 + N n1 + n2
        
        M^-1(x, r, i)_m = x delta(n - m)
    
    Perform sum in I on cpu (out-of-order memory operations)
    """
    def __init__(
        self, 
        K_di, 
        M_ri, 
        P_dr, 
        Wsums_r, 
        models_I,
        update_fluence = False,
        likelihood     = 'Poisson',
        frame_model    = 'basic',
        maximise       = 'I',
        **config
    ):
        self.K_di    = K_di
        self.M_ri    = M_ri
        self.P_dr    = P_dr
        self.Wsums_r = Wsums_r
        self.w_d     = models_I.w
        
        # confusing having model intensities as I and number of pixels I
        self.Is      = models_I.I
        self.Os      = []
        # set to zero 
        for c in range(len(self.Is)):
            self.Is[c][:] = 0
            self.Os.append(np.zeros_like(self.Is[c]))

        if update_fluence :
            self.w_d[:] = 0.
        
        self.D, self.R = self.P_dr.shape
        self.I         = K_di.shape[1]
        
        self.C = config['C']
        self.models = config['models']
        
        self.r_chunk_size = 256
        self.d_chunk_size = 256
        
        self.update_fluence = update_fluence
        self.likelihood     = likelihood
        self.frame_model    = frame_model
        self.maximise       = maximise
    
    def calc(self):
        # split r over mpi and local chunks
        # every chunk is <= r_chunk_size
        r_iter = utils.chunker_mpi_local(
            self.r_chunk_size, 
            size, 
            self.R
        )[rank]
        Nr = len(list(r_iter))
        
        PK_ri = np.empty((self.r_chunk_size, self.I), dtype = float)
        
        for r0, r1, dr in tqdm(r_iter, desc = 'merging over I', total = Nr, disable = quiet):
            N_ri, D_ri, b_d = self.calc_N_D_rchunk(
                np.ascontiguousarray(self.P_dr[:, r0:r1]), 
                self.Wsums_r[r0:r1], 
                PK_ri[:dr]
            )
            
            if self.update_fluence :
                self.w_d += b_d
            
            for c in range(len(self.Is)):
                I = self.Is[c]
                O = self.Os[c]
                
                # get r --> class index mapping
                r = np.where(self.M_ri.class_r[r0:r1] == c)[0]
                
                if len(r) > 0 :
                    # calculate pixel mappings
                    n = self.M_ri[r0+r[0]: r0+1+r[-1], :]
                    # this is needed as there are duplicates along axis 0
                    # when symmetry mapping is enabled
                    # np.add.at allows for duplicates
                 
                    if self.maximise == 'I':
                        pass
                        
                    elif self.maximise == 'W':
                        D_ri[D_ri==0] = 1
                        N_ri         /= D_ri
                        D_ri[:]       = 1
                    else :
                        err = f'failed to parse maximise option: {self.maximise}'
                        raise ValueError(err)
                    
                    np.add.at(I.ravel(), n, N_ri[r])
                    np.add.at(O.ravel(), n, D_ri[r])
            
        # mpi reduce
        for c in range(len(self.Is)):
            if rank == 0 :
                I = np.empty_like(self.Is[c])
                O = np.empty_like(self.Os[c])

            else :
                I = None
                O = None
            
            comm.Reduce(self.Is[c], I, op = MPI.SUM, root = 0) 
            comm.Reduce(self.Os[c], O, op = MPI.SUM, root = 0) 
            comm.barrier()
            
            if rank == 0 :
                O[O==0] = 1
                I /= O
                self.Is[c] = I.copy()
        return self.Is
    
    
    def calc_N_D_rchunk(self, P_dr, Wsums_r, PK_ri):
        # split d over local chunks
        d_iter = utils.chunker(
            self.d_chunk_size, 
            self.K_di.shape[0]
        )
        Nd = len(list(d_iter))
        
        # PK = P^T . K = sum_d P_dr K_di
        # ------------------------------
        PK_ri[:] = 0
        for d0, d1, dd in tqdm(d_iter, desc = 'looping over d', total = Nd, leave = False, disable = quiet):
            K_di   = self.K_di[d0:d1, :]
            PK_ri += np.dot(P_dr[d0:d1].T, K_di[:dd])
        
        # N_ri  = sum_d P_dr K_di
        # D_ri  = C_i sum_d P_dr 
        if self.likelihood == 'Poisson' and self.frame_model == 'basic':
            N_ri = PK_ri
            D_ri = np.outer(np.sum(P_dr[d0:d1], axis=0), self.C)
        
        # N_ri = Wsum_r sum_d P_dr K_di
        # D_ri = C_i sum_d P_dr K_d
        elif self.likelihood == 'Poisson_fluence_free' and self.frame_model == 'basic':
            N_ri = (Wsums_r * PK_ri.T).T
            D_ri = np.outer(np.dot(self.K_di.photon_sums, P_dr), self.C)
        
        # N_ri  = sum_d P_dr K_di
        # D_r   = C_i sum_d w_d P_dr
        elif self.likelihood == 'Poisson' and self.frame_model == 'fluence':
            N_ri = PK_ri
            D_ri = np.outer(np.dot(self.w_d, P_dr), self.C)

        else :
            err = f'could not parse likelihood "{self.likelihood}" and frame_model "{self.frame_model}" combination'
            raise ValueError(err)
            
        # a_d   = sum_i K_di
        # b_d   = sum_r P_dr Wsum_r
        # w'_d  = a_d / b_d
        if self.update_fluence :
            b_d  = np.dot(P_dr, Wsums_r)
        else :
            b_d = None
        
        return N_ri, D_ri, b_d
        
