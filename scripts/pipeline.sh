#!/bin/bash

DIR="$(dirname "$1")"

# can be optimised by determining the number of classes then
# running this for each class in parallel
# python scripts/init.py $1

# calculate tomogram sums
# parallel python scripts/calculate_wsums.py ::: ${DIR}/class_*.h5

# calculate logR matrix with 1 cpu per class
parallel python scripts/calculate_logR.py ::: ${DIR}/class_*.h5 | python emc2/pipe_to_h5.py

# calculate P matrix by normalising logR over classes with 8 cpus
parallel python scripts/calculate_probability.py ${DIR}/class_*.h5 --data_chunks 8 --data_chunk {} ::: $(seq 0 7) | python emc2/pipe_to_h5.py
