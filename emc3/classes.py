from pathlib import Path

default_class = {
    'cxi_file': None,
    'class_id': None,
    'P_data': 0,
    'K_data': 0,
    'model': None,
    'relative_fluence': None,
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
    'polarisation': 'x',
    'r_offset': 0,
    'probability_matrix': None,
    'probability_matrix_file': None,
    'wsums': None,
    'P_wsums': None,
    'P_ksums': None,
    'ksums': None,
}


def make_config(config):
    classes = []

    # global r index offset
    r_offset = 0

    wd = Path(config['working_directory'])
    if not wd.is_dir():
        raise ValueError('must set working_directory in config')

    for i, c in enumerate(config['classes']):
        d = dict(default_class)
        d.update(c)

        if d['class_id'] is None:
            d['class_id'] = i

        d['r_offset'] = r_offset
        r_offset += d['mapper'].shape[1]

        if d['probability_matrix_file'] is None:
            fnam = wd / f"probability_matrix_{d['class_id']}.h5"
            d['probability_matrix_file'] = fnam

        classes.append(d)

    config['R'] = r_offset
    config['classes'] = classes
    return config
