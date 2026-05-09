"""
Dispatch calculate_logR and update_models to the correct implementation
based on each class's frame_model ('basic' or 'background').
"""

def cids_background_logR(cids, config):
    out_back = []
    out_norm = []
    for ci in cids:
        if config['classes'][ci]['update_logR']:
            if config['classes'][ci]['frame_model'] == 'background':
                out_back.append(ci)
            else:
                out_norm.append(ci)
    return out_norm, out_back


def cids_background_models(cids, config):
    out_back = []
    out_norm = []
    force_model_background = config.get('force_model_background', False)
    for ci in cids:
        # actually we need to call update_models so we get wsums_r
        # even for models with 'update_model'=False
        if (config['classes'][ci]['frame_model'] == 'background' \
                and config['classes'][ci]['likelihood'] == 'Poisson') \
                or force_model_background:
            out_back.append(ci)
        else:
            out_norm.append(ci)
    return out_norm, out_back


def calculate_logR(config_file, config, p_per_device=2, cids=None):
    if cids is None:
        cids = list(range(len(config['classes'])))

    normal, background = cids_background_logR(cids, config)

    if normal:
        from .calculations import calculate_logR_subprocess
        calculate_logR_subprocess(config_file, config, p_per_device=p_per_device, cids=normal)

    if background:
        from .background.calculate_logR_single import calculate_logR_subprocess
        calculate_logR_subprocess(config_file, config, p_per_device=p_per_device, cids=background)


def update_models(config_file, config, p_per_device=2, cids=None, **kwargs):
    if cids is None:
        cids = list(range(len(config['classes'])))

    normal, background = cids_background_models(cids, config)

    if normal:
        from .update_models import update_model_subprocess
        update_model_subprocess(config_file, config, p_per_device=p_per_device, cids=normal, **kwargs)

    if background:
        from .background.update_I import update_I
        update_I(config_file, config, cids=background, **kwargs)
