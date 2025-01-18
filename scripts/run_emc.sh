#!/bin/bash

set -e

# should get these using python

config=/home/andyofmelbourne/Documents/2024/p7927/scratch/3D-EMC/Ery2/config.py
class_file="/home/andyofmelbourne/Documents/2024/p7927/scratch/3D-EMC/Ery2/class_"

# number of classes 
C=4

# number of iterations 
C=4

# class sequence
classes=$(seq 0 $(($C-1)))

# initialise classes 
time parallel -j 2 "python scripts/init.py $config --class_no {}" ::: $classes

# calculate wsums
time parallel -j 2 "python scripts/calculate_wsums.py ${class_file}{}.h5" ::: $classes

# calculate logR
time parallel --eta -j 2 "python scripts/calculate_logR.py ${class_file}{1}.h5 --data_chunk {2} --data_chunks 8" ::: $classes ::: $(seq 0 7)

# normalise 

# update I 
