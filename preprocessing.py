import json
import os
import shutil
import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import neurospyke as ns
import preprocessing

f = open('settings.json')
settings = json.load(f)
f.close()

sampling_frequency = settings['sampling_frequency']
sampling_time = 1 / sampling_frequency
resampling_frequency = settings['resampling_frequency']
resampling_time = 1 / resampling_frequency
signal_duration = settings['signal_duration']
trial_duration = settings['trial_duration']

group = settings['group']
subject = settings['subject']
conditions = settings['conditions']
areas = list(settings['areas'].keys())
n_channels = settings['n_channels']
n_stimuli = settings['n_stimuli']

output_dir = './output'
preprocessed_data = np.ndarray([np.size(conditions), np.size(areas), n_channels, n_stimuli, trial_duration * resampling_frequency])
print(np.shape(preprocessed_data))

def get_raw_data_paths(group, subject, conditions, areas):
    subject_path = os.path.join('./data', group, subject)
    raw_data_paths = {}
    for condition in conditions:
        files = [os.path.join(subject_path, condition, f) for f in os.listdir(os.path.join(subject_path, condition)) if (os.path.isfile(os.path.join(subject_path, condition, f)) and f != 'Raw-Info.mat')]
        raw_data_paths[condition] = {}
        for area in areas:
            raw_data_paths[condition][area] = [file for file in files if file.split('\\')[-1].find('_' + area + '_') != -1 ]

    return raw_data_paths
   
raw_data_paths = get_raw_data_paths(group, subject, conditions, areas)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for condition in tqdm(conditions, desc='Conditions'):
    for area in tqdm(areas, desc='Brain Areas', leave=False):
        output_path = os.path.join(output_dir, group, subject, condition, area)
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)

        os.makedirs(output_path)

        for path in tqdm(raw_data_paths[condition][area], desc='Channels', leave=False):
            channel = int(path.split('\\')[-1].split('_Ch_')[-1].split('.mat')[0])
            f = h5py.File(path)
            data = f['data'][:]
            data = np.reshape(data, np.size(data))
            data = data[np.arange(0, signal_duration * sampling_frequency)]
            raw_data = np.copy(data)

            data, stimulus_idxs = preprocessing.run_SALPA(data, sampling_time)

            ns.visualization.plot_spikes(raw_data, stimulus_idxs, sampling_time=sampling_time, title='Raw ' + str(len(stimulus_idxs)), dpi=200)
            plt.savefig(os.path.join(output_path, str(channel) + 'a.png'))
            plt.close()

            ns.visualization.plot_spikes(data, stimulus_idxs, sampling_time=sampling_time, title='SALPA ' + str(len(stimulus_idxs)), dpi=200)
            plt.savefig(os.path.join(output_path, str(channel) + 'z.png'))
            plt.close()