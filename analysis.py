import json
import os
import numpy as np
import neurospyke as ns
from scipy.io import loadmat
from tqdm import tqdm
import utils

f = open('settings.json')
settings = json.load(f)
f.close()

sampling_frequency = settings['sampling_frequency']
sampling_time = 1 / sampling_frequency

resampling_frequency = settings['resampling_frequency']
resampling_time = 1 / resampling_frequency

trial_duration = settings['trial_duration_view']
trial_samples = np.floor(trial_duration * sampling_frequency).astype(np.int_)

group = settings['group']
subject = settings['subject']
conditions = settings['conditions']
areas = list(settings['areas'].keys())
areas_labels = [settings['areas'][area]['label'] for area in areas]
n_channels = settings['n_channels']
n_stimuli = settings['n_stimuli']

output_dir = './output'
subject_dir = os.path.join(output_dir, group, subject)
subject_data = os.path.join(subject_dir, subject + '.mat')

if not os.path.isdir(output_dir):
    raise Exception('Output directory not found.')
elif not os.path.isdir(subject_dir):
    raise Exception(subject_dir + ' directory not found.')
elif not os.path.isfile(subject_data):
    raise Exception('No ' + subject + '.mat' + ' file found in ' + subject_dir)
else:
    spike_trains = loadmat(subject_data)
    spike_trains = spike_trains['spike_trains']

    for conditions_idx in tqdm(np.arange(np.size(conditions)), desc='Conditions', leave=False):
        condition = conditions[conditions_idx]

        for areas_idx in tqdm(np.arange(np.size(areas)), desc='Areas', leave=False):
            area = areas[areas_idx]
            subject_information = utils.get_subject_information(group, subject, condition, area)
            raster_plot_filename = 'Raster_Plot_' + subject_information
            raster_plot_title = 'Stimulus-Related Raster Plot\n' + subject_information

            ns.visualization.plot_raster(spike_trains[conditions_idx, areas_idx, :, :, 0:trial_samples], sampling_time=sampling_time, n_cols=4, is_train=True, figsize=[15, 15], title=raster_plot_title, xlim=[0, trial_duration*1.01])
            ns.visualization.pyplot.savefig(os.path.join(subject_dir, raster_plot_filename + '.png'))
            ns.visualization.pyplot.close()

print('Done.')