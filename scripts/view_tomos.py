"""
parallelise over frames
trust that numpy already does multithreaded np.dot
mpi or multiprocessing?
I can't get opencl to behave with multiprocessing
so mpi it is
"""
from context import emc2

from emc2 import utils
from emc2 import utils_cl
from emc2 import tomograms
from emc2 import classes

import signal
import argparse
import pyqtgraph as pg
import numpy as np
from pathlib import Path

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        Calculate probability matrix
        """
    )
    parser.add_argument('class_file', type=str, help='class file name')
    parser.add_argument('--nproc', type=int, default=1,
                        help='number of proccessors to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    working_directory = Path(args.class_file).parent
    cachedir = working_directory.joinpath('cachdir')

    # load class file
    class_c = classes.Class()
    class_c.load(args.class_file, skip=['probability_matrix'])

    opencl_stuff = utils_cl.opencl_init()

    # initialise mapper: r,i -> s,n
    mapper = tomograms.Mapper(
        class_c.model.ndim,
        class_c.model.shape[0],
        class_c.xyz,
        class_c.mapping_matrix,
        opencl_stuff['context'],
        opencl_stuff['queue'],
        interpolation=class_c.interpolation_forward
    )

    W_ri_getter = tomograms.Tomograms(mapper, class_c.model)

    # view tomograms
    geom = utils.Geom_corr_masked(class_c.mask)

    rs = np.sort(np.random.randint(0, class_c.wsums.shape[0], 20))

    # W_ri = W_ri_getter[rs, :]
    W_ri = mapper[0, rs, :][0]
    print(f'{W_ri.min()=} {W_ri.max()=} {W_ri.dtype=}')

    ims = geom.apply(W_ri)

    pg.setConfigOption('background', 0.1)
    pg.setConfigOption('foreground', 'w')
    pg.setConfigOptions(antialias=True)
    pg.setConfigOptions(imageAxisOrder='row-major')
    pg.show(ims)
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow Control-C
    pg.exec()
