#!/bin/bash

set -e 

# fails because of MPI msg should make utils independent of mpi
N=$(python scripts/get_number_of_iterations.py $1)
#mpirun -np 2 python scripts/update_I.py $1
for i in $(seq 1 $N); do
    mpirun -np 2 python scripts/calc_prob.py $1
    mpirun -np 2 python scripts/update_I.py $1
done
