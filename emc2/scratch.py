

code = """
__kernel void update_O (
    global int *N_sri, 
    global int *O,
    const int S,
    const int R
) {
    int i = get_global_id(0);
    int I = get_global_size(0);
    int n;
    
    for (int s=0; s<S; s++) {
    for (int r=0; r<R; r++) {
        
        n = N_sri[s * R * I + r * I + i];
        
        atomic_inc(&O[n]);
    
    }}
}
"""

def calculate_overlap_on_gpu():
    # see how long it takes to calculate the overlap integral
    # to see how many photons arrive at each point in I
    O = np.zeros(Is[c].size, dtype = np.int32)
    O_cl = utils_cl.to_gpu(O, queue = queue)
    event = None
    for d in tqdm(range(D), desc = 'calculating O (looping over d)', disable = False):
        
        # get r with P_dr above threshold
        rs = rs_d[d]
        
        if len(rs) > 0 :       
            Kd_i, pixels = K_di.sparse(d)
            Bd_i         = B_di.sparse(d, pixels)
            Cd_i         = config['C'][pixels]
            
            # (sym, rs, pixels)
            if event: event.wait()
            
            N_sri  = M_ri[rs, pixels]
            #print(f'{type(N_sri)=} {N_sri.shape=}', file = sys.stderr)
            
            # gpu is much faster (x100) on intelHD vs 4 x cpu
            #event = cl_code.update_O(queue, (len(pixels),), None,  cl.SVM(N_sri), cl.SVM(O), np.int32(N_sri.shape[0]), np.int32(N_sri.shape[1]))
            event = cl_code.update_O(queue, (len(pixels),), None,  N_sri.data, O_cl.data, np.int32(1), np.int32(N_sri.shape[0]))
            
            #for s in range(N_sri.shape[0]):
            #    for r in range(N_sri.shape[1]):
            #        O += np.bincount(N_sri[s, r], Kd_i, minlength = O.size).astype(O.dtype)
    
    queue.finish()
    O = O_cl.get()
