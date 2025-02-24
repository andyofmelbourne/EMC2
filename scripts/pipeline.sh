#!/bin/bash

set -e

# I like to have many scripts to be called from the command line
# This leads to more modularity

DIR="$(dirname "$1")"

iterations=$(python -c "
from emc2 import input_output
import sys
config = input_output.load_config(sys.argv[1])
print(config['iterations'])
" $1)

restart=$(python -c "
from emc2 import input_output
import sys
config = input_output.load_config(sys.argv[1])
print(config['restart'])
" $1)

# chunk calls to logR based on number of frames and orientations
logR_cmd() {
python -c "
import sys
import h5py
import math

# D x R chunksize
chunksize = 64 * 1024 * 1024

# for each class file get D and R
for fnam in sys.argv[1:]:
    with h5py.File(fnam, 'r') as f:
        D, R = f['probability_matrix'].shape
    # number of chunks
    chunks = math.ceil((D*R) / chunksize)
    for chunk in range(chunks):
        print(f'python scripts/calculate_logR.py {fnam} \
              --data_chunk {chunk} \
              --data_chunks {chunks}')
" ${DIR}/class_*.h5
}
export -f logR_cmd

# can be optimised by determining the number of classes then
# running this for each class in parallel
if [[ $restart == "True" ]]; then
	rm -f ${DIR}/iteration_info.h5
	rm -f ${DIR}/class_*.h5
	python scripts/init.py $1
fi

echo $iterations $restart

for (( iteration = 0; iteration < iterations; iteration++ )); do
	# calculate tomogram sums
	parallel --verbose --jobs 50% python scripts/calculate_wsums.py ::: ${DIR}/class_*.h5

	# calculate logR matrix with 1 cpu per chunk
	logR_cmd | parallel --delay 1 --verbose --jobs 50% | python emc2/pipe_to_h5.py

	# calculate P matrix by normalising logR over classes with 8 cpus
	# this doesn't need to use parallel, it helps a little with loading
	# data but then pickle piping is slow
	#parallel python scripts/calculate_probability.py ${DIR}/class_*.h5 --data_chunks 8 --data_chunk {} ::: $(seq 0 7) | python emc2/pipe_to_h5.py
	python scripts/calculate_probability.py $1 $iteration

	python scripts/update_w.py ${DIR}/class_*.h5

	parallel --delay 1 --verbose --jobs 50% python scripts/update_I_new.py --numpy --r_chunk_size 1024 ::: ${DIR}/class_*.h5
	python scripts/save_model_slices.py ${DIR}/class_*.h5
done
