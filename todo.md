# Pipeline

## Init
- [x] scripts/init.py: write class files class_<n>.h5 to working directory that contain all information needed for further processing of that class.
- [ ] make script faster (maybe we can run it with each script). 

a bit tricky since we need to check these files every time an option is changed.

```bash
h5ls ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/class_0.h5 
C                        Dataset {27100}
P_thresh                 Dataset {SCALAR}
beta                     Dataset {SCALAR}
cxi_file                 Dataset {SCALAR}
dq                       Dataset {SCALAR}
frame_model              Dataset {SCALAR}
frame_selection          Dataset {188557}
interpolation_forward    Dataset {SCALAR}
likelihood               Dataset {SCALAR}
mapping_matrix           Dataset {3, 93555, 5, 3}
mask                     Dataset {16, 512, 128}
maximise                 Dataset {SCALAR}
model                    Dataset {64, 64, 64}
pixels_per_voxel         Dataset {SCALAR}
pointing_fluctuations    Dataset {2}
polarisation             Dataset {SCALAR}
probability_matrix       Dataset {1025, 93555}
relative_fluence         Dataset {1025}
rotation_order           Dataset {SCALAR}
split_frames             Dataset {SCALAR}
symmetry                 Dataset {SCALAR}
update_fluence           Dataset {SCALAR}
update_model             Dataset {SCALAR}
wsums                    Dataset {93555}
xyz                      Dataset {3, 27100}
```

```bash
$ python scripts/init.py ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/config.py
```

## LogR
- [x] calculate logR matrix for each class file
- [x] chunked calculation to enable multiprocessing

```bash
$ python scripts/calculate_logR.py ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/class_0.h5 
$ python scripts/calculate_logR.py ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/class_1.h5 
$ python scripts/calculate_logR.py ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/class_2.h5 
$ python scripts/calculate_logR.py ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/class_3.h5 
```

## Probability Matrix
- [x] normalise LogR matrix over all classes

```bash
$ python scripts/calculate_probability.py ~/Documents/2024/p7927/scratch/2D-EMC/Ery_test/class_*.h5 
```
needs all class files

## Models
- [x] update models for each class file

## Iteration Info
- [x] write summery information to iteration info for viewing

## Utilities 
- [x] view iteration info script
- [x] script for executing pipeline
- [ ] GUI for executing pipeline
- [x] script for submitting script to maxwell over ssh connection
- [ ] add command line arguments to overide defaults in class files
- [ ] write generic maxwell run / get scripts with config as input
