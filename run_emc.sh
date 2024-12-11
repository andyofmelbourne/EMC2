#!/bin/bash

rm ~/Documents/2024/p7927/scratch/3D-EMC/Ery2/iteration_info.h5
#rm ~/Documents/2024/p7927/scratch/3D-EMC/Cube/iteration_info.h5

set -e 

N=$(python scripts/get_number_of_iterations.py $1)
for i in $(seq 1 $N); do
    mpirun -np 2 python scripts/calc_prob.py $1
    mpirun -np 2 python scripts/update_I.py $1
    #python scripts/calc_prob.py $1
    #python scripts/update_I.py $1
done
