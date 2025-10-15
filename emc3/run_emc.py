import numpy as np
import pickle
import emc3
import time
import sys

from emc3 import calculations

config_file = sys.argv[1]
iters = int(sys.argv[2])
beta = float(sys.argv[3])

config = pickle.load(open(config_file, 'rb'))

# load data
for c in config['classes']:
    c['data'].load_from_file()
    c['P_data'].load_from_file()


for i in range(iters):
    t0 = time.time()
    calculations.calculate_logR_cl(config)
    time_logR = time.time() - t0

    t0 = time.time()
    emc3.probability.calculate_P(config, beta)
    time_prob = time.time() - t0

    t0 = time.time()
    emc3.update_models.update_model_basic(config)
    time_I = time.time() - t0

    # save model slices
    # -----------------
    dq = config['classes'][0]['model'].dq
    models = [c['model'].data for c in config['classes']]
    emc3.utils.save_model_slices(
            models,
            dq,
            config['working_directory']
            )

    # save extra info
    # ---------------
    # in principle different classes can have different data
    # in which case this will break
    frames_d = config['classes'][0]['data'].frames
    D = config['classes'][0]['data'].source_shape[0]

    # save DOS
    # need to clean all this iteration_info stuff up
    k = 'frame_labels'
    if k in config:
        m_d = config['most_likely_model_d']
        A_d, B_d, *_ = config[k].values()
        A_d = A_d[frames_d]
        B_d = B_d[frames_d]
        DOS = emc3.calculate_DOS(A_d, B_d, m_d)
        frame_labels = config[k]
    else:
        DOS = None
        frame_labels = None

    emc3.utils.save_data_info(
            frames_d,
            frame_labels,
            DOS,
            D,
            config['working_directory']
            )

    print(f'{time_logR=}')
    print(f'{time_prob=}')
    print(f'{time_I=}')


# un-load data
for c in config['classes']:
    c['data'].unload()
    c['P_data'].unload()
    c['logR_dr'] = None
    c['P_dr'] = None

# save
pickle.dump(config, open(config_file, 'wb'))

