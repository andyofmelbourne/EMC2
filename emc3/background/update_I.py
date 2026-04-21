from . import update_I_cn
from .update_I_buffer_size import update_buffer_size
from .update_I_fill_buffer import fill_buffer
from .update_w_d import w_update
import sys


def update_I(config_file, config, iters=1, update_w_first=False, update_w=True, update_b=False, cids=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()

    # get classes to process
    if cids is None:
        cids = list(range(len(config['classes'])))

    if update_w_first:
        w_update(config, config_file, update_b=update_b)

        comm.Barrier()

    for i in range(iters):
        update_I_cn.calculate_c_n(config, config_file, cids=cids)

        update_buffer_size(config, config_file, cids=cids)

        fill_buffer(config, config_file, cids=cids)

        print(f'\n{rank=} waiting at barrier\n')
        sys.stdout.flush()
        comm.Barrier()
        print(f'\n{rank=} passing barrier\n')
        sys.stdout.flush()

        if update_w:
            w_update(config, config_file, update_b=update_b)
