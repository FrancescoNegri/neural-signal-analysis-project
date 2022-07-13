import json
import os
import shutil
import numpy as np
import neurospyke as ns
from scipy.io import loadmat

f = open('settings.json')
settings = json.load(f)
f.close()

sampling_frequency = settings['sampling_frequency']
sampling_time = 1 / sampling_frequency
trial_duration_view = settings['trial_duration_view']
trial_samples = np.floor(trial_duration_view * sampling_frequency).astype(np.int_)

group = settings['group']
subject = settings['subject']
conditions = settings['conditions']
areas = list(settings['areas'].keys())
n_channels = settings['n_channels']
n_stimuli = settings['n_stimuli']

output_dir = './output'

if not os.path.isdir(output_dir):
    print('output not found')
else:
    spike_trains = loadmat(os.path.join(output_dir, group, subject, subject + '.mat'))
    spike_trains = spike_trains['spike_trains']

    for conditions_idx in np.arange(np.size(conditions)):
        for areas_idx in np.arange(np.size(areas)):
            ns.visualization.plot_raster(spike_trains[conditions_idx, areas_idx, :, :, 0:trial_samples], sampling_time=sampling_time, n_cols=4, is_train=True, figsize=[10, 10])
            ns.visualization.pyplot.savefig('./temp.png')
            ns.visualization.pyplot.close()