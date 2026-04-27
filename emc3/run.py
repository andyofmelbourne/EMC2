"""
Dispatch calculate_logR and update_models to the correct implementation
based on each class's frame_model ('basic' or 'background').
"""


def calculate_logR(config_file, config, p_per_device=2, cids=None):
    if cids is None:
        cids = list(range(len(config['classes'])))

    basic      = [ci for ci in cids if config['classes'][ci]['frame_model'] == 'basic']
    background = [ci for ci in cids if config['classes'][ci]['frame_model'] == 'background']

    if basic:
        from .calculations import calculate_logR_subprocess
        calculate_logR_subprocess(config_file, config, p_per_device=p_per_device, cids=basic)

    if background:
        from .background.calculate_logR_single import calculate_logR_subprocess
        calculate_logR_subprocess(config_file, config, p_per_device=p_per_device, cids=background)


def update_models(config_file, config, p_per_device=2, cids=None):
    if cids is None:
        cids = list(range(len(config['classes'])))

    basic      = [ci for ci in cids if config['classes'][ci]['frame_model'] == 'basic']
    background = [ci for ci in cids if config['classes'][ci]['frame_model'] == 'background']

    if basic:
        from .update_models import update_model_subprocess
        update_model_subprocess(config_file, config, p_per_device=p_per_device, cids=basic)

    if background:
        from .background.update_I import update_I
        update_I(config_file, config, cids=background)
