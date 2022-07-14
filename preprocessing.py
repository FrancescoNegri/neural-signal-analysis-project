import json
import os
import shutil
import h5py
import numpy as np
import neurospyke as ns
from tqdm import tqdm
from scipy.signal import butter, lfilter, resample
from scipy.stats import median_abs_deviation
from scipy.io import savemat
from scipy.ndimage import gaussian_filter1d
import preprocessing
import utils

f = open('settings.json')
settings = json.load(f)
f.close()

sampling_frequency = settings['sampling_frequency']
sampling_time = 1 / sampling_frequency
resampling_frequency = settings['resampling_frequency']
resampling_time = 1 / resampling_frequency
frequency_ratio = sampling_frequency / resampling_frequency

signal_duration = settings['signal_duration']
trial_duration = settings['trial_duration']

group = settings['group']
subject = settings['subject']
conditions = settings['conditions']
areas = list(settings['areas'].keys())
n_channels = settings['n_channels']
n_stimuli = settings['n_stimuli']

output_dir = './output'

# Output vars
spike_trains = np.ndarray([np.size(conditions), np.size(areas), n_channels, n_stimuli, trial_duration * sampling_frequency])
preprocessed_data = np.ndarray([np.size(conditions), np.size(areas), n_channels, n_stimuli, trial_duration * resampling_frequency])

raw_data_paths = utils.get_raw_data_paths(group, subject, conditions, areas)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

subject_dir = os.path.join(output_dir, group, subject)
if os.path.isdir(subject_dir):
    print('The ' + subject_dir + ' path already exists.')
    ans = input('Do you want to delete all the content and execute a new preprocessing? (y/[n])\n')
    if ans == 'y':
        shutil.rmtree(subject_dir)
    else:
        print('Abort.')
        exit()
os.makedirs(subject_dir)

for conditions_idx in tqdm(np.arange(np.size(conditions)), desc='Conditions'):
    condition = conditions[conditions_idx]

    # Get stimulus idxs for the considered condition
    stimulus_idxs_matrix = preprocessing.get_stimulus_idxs_matrix(raw_data_paths, condition, settings)
    stimulus_idxs = preprocessing.get_median_stimulus_idxs(stimulus_idxs_matrix, tolerance_duration=0.1, settings=settings)

    for areas_idx in tqdm(np.arange(np.size(areas)), desc='Brain Areas', leave=False):
        area = areas[areas_idx]

        # output_path = os.path.join(subject_dir, condition, area)
        # if os.path.isdir(output_path):
        #     shutil.rmtree(output_path)

        # os.makedirs(output_path)
        
        for channel_path in tqdm(raw_data_paths[condition][area], desc='Channels', leave=False):
            channel = int(channel_path.split('\\')[-1].split('_Ch_')[-1].split('.mat')[0])
            f = h5py.File(channel_path)
            data = f['data'][:]
            data = np.reshape(data, np.size(data))
            data = data[np.arange(0, signal_duration * sampling_frequency)]

            # Clean data by suppressing stimulus artifacts
            data = preprocessing.suppress_stimulus_artifacts(data, stimulus_idxs, sampling_time)

            # Apply a 300-7000 Hz band-pass filter
            num, den = butter(2, [300, 7000], btype='bandpass', fs=sampling_frequency)
            data = lfilter(num, den, data)

            # Spike Detection: Hard Threshold Local Maxima
            threshold = 5*1.4824*median_abs_deviation(data)
            spikes_idxs, _ = ns.spikes.hard_threshold_local_maxima(data, threshold, refractory_period=0.001, use_abs=True, sampling_time=sampling_time)
            
            # Get spike train and divide it in trials
            spike_train = ns.utils.convert_spikes_idxs_to_spike_train(spikes_idxs, sampling_time=sampling_time)
            spike_train_trials = preprocessing.shape_trials(spike_train, stimulus_idxs, trial_duration, sampling_time)
            spike_trains[conditions_idx][areas_idx][channel] = spike_train_trials

            # Get Istantaneous Firing Rate (IFR)
            IFR, bin_samples = preprocessing.get_IFR(spike_train, bin_duration=0.05, sampling_time=sampling_time)

            # FIlter IFR with a Gaussian
            IFR = gaussian_filter1d(IFR, 0.95 * bin_samples)

            # Resample IFR
            IFR = resample(IFR, int(np.floor(np.size(spike_train, axis=0) / frequency_ratio)))

            # Resample the stimulus indexes
            resampled_stimulus_idxs = np.round(stimulus_idxs / frequency_ratio).astype(np.int_)

            # Get IFR trials
            IFR_trials = preprocessing.shape_trials(IFR, resampled_stimulus_idxs, trial_duration, sampling_time=resampling_time)
            preprocessed_data[conditions_idx][areas_idx][channel] = IFR_trials
            

print('Saving ...')
savemat(os.path.join(subject_dir, subject + '.mat'), {'spike_trains': spike_trains, 'data': preprocessed_data})
print('Done.')
