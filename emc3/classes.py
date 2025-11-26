from pathlib import Path

default_class = {
    'cxi_file': None,
    'class_id': None,
    'P_data': 0,
    'model': None,
    'mapper': None,
    'interpolation_forward': 'linear',
    'likelihood': 'Poisson_fluence_free',
    'frame_model': 'basic',
    'maximise': 'W',
    'P_thresh': 0,
    'beta': 1.,
    'update_fluence': True,
    'update_logR': True,
    'update_probability': True,
    'update_model': True,
    'filter_model': None,
    'polarisation': 'x',
    'r_offset': 0,
    'P_dr': None,
    'probability_matrix_file': None,
    'logR_file': None,
    'model_file': None,
    'P_wsums_file': None,
    'wsums_file': None,
}


def make_config(config):
    classes = []

    # global r index offset
    r_offset = 0

    wd = Path(config['working_directory'])
    if not wd.is_dir():
        raise ValueError('must set working_directory in config')

    # point to same file for now
    k = 'fluence_file'
    if k not in config or config[k] is None:
        fnam = wd / f"fluence.h5"
        config[k] = fnam

    k = 'update_fluence'
    if k not in config or config[k] is None:
        config[k] = True

    for i, c in enumerate(config['classes']):
        d = dict(default_class)
        d.update(c)

        if d['class_id'] is None:
            d['class_id'] = i

        d['r_offset'] = r_offset
        r_offset += d['mapper'].shape[1]

        d['fluence_file'] = config['fluence_file']

        if d['probability_matrix_file'] is None:
            fnam = wd / f"class_probability_matrix_{d['class_id']}.h5"
            d['probability_matrix_file'] = fnam

        if d['model_file'] is None:
            fnam = wd / f"class_model_{d['class_id']}.h5"
            d['model_file'] = fnam

        if d['logR_file'] is None:
            fnam = wd / f"class_logR_{d['class_id']}.h5"
            d['logR_file'] = fnam

        if d['P_wsums_file'] is None:
            fnam = wd / f"class_P_wsums_{d['class_id']}.h5"
            d['P_wsums_file'] = fnam

        if d['wsums_file'] is None:
            fnam = wd / f"class_wsums_{d['class_id']}.h5"
            d['wsums_file'] = fnam

        classes.append(d)

    config['R'] = r_offset
    config['classes'] = classes
    return config
