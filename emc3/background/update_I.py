from . import update_I_cn
from .update_I_buffer_size import update_buffer_size
from .update_I_fill_buffer import fill_buffer
from .update_w_d import w_update

def update_I(config, config_file, iters=1, update_w_first=False, update_w=True, update_b=False):
    if update_w_first:
        w_update(config, config_file, update_b=update_b)

    for i in range(iters):
        update_I_cn.calculate_c_n(config, config_file)

        update_buffer_size(config, config_file)

        fill_buffer(config, config_file)

        if update_w:
            w_update(config, config_file, update_b=update_b)
