import h5py
import numpy as np
import neurospyke as ns
from scipy.signal import savgol_filter

def get_stimulus_idxs_matrix(raw_data_paths, condition, settings):
    sampling_frequency = settings['sampling_frequency']
    n_channels = settings['n_channels']
    n_stimuli = settings['n_stimuli']
    signal_duration = settings['signal_duration']

    areas = list(raw_data_paths[condition].keys())

    stimulus_idxs_matrix = np.ndarray([np.size(areas) * n_channels, n_stimuli])
    stimulus_idxs_matrix[:, :] = np.nan

    for area_idx in np.arange(np.size(areas)):
        area = areas[area_idx]
        
        for channel_path_idx in np.arange(np.size(raw_data_paths[condition][area])):
            channel_path = raw_data_paths[condition][area][channel_path_idx]
            f = h5py.File(channel_path)

            data = f['data'][:]
            data = np.reshape(data, np.size(data))
            data = data[np.arange(0, signal_duration * sampling_frequency)]

            channel_stimulus_idxs, _ = ns.spikes.differential_threshold(data, threshold=4000, window_length=160, refractory_period=4*sampling_frequency)

            channel_idx = area_idx * n_channels + channel_path_idx
            if np.size(channel_stimulus_idxs) > n_stimuli:
                channel_stimulus_idxs = channel_stimulus_idxs[0:n_stimuli]
            stimulus_idxs_matrix[channel_idx, 0:np.size(channel_stimulus_idxs)] = channel_stimulus_idxs

    return stimulus_idxs_matrix

def get_median_stimulus_idxs(stimulus_idxs_matrix, tolerance_duration, settings):
    sampling_frequency = settings['sampling_frequency']
    n_channels = settings['n_channels']
    n_stimuli = settings['n_stimuli']

    tolerance_samples = tolerance_duration * sampling_frequency
    stimulus_idxs = np.ndarray(n_stimuli)

    for idx in np.arange(n_stimuli):
        stimulus_idxs[idx] = int(np.round(np.nanmedian(stimulus_idxs_matrix[:, idx])))
        
        for channel_idx in np.arange(n_channels):
            if np.abs(stimulus_idxs_matrix[channel_idx, idx] - stimulus_idxs[idx]) <= tolerance_samples:
                pass
            else:
                remaining_stimulus_idxs = np.copy(stimulus_idxs_matrix[channel_idx, idx:(n_stimuli - 1)])
                stimulus_idxs_matrix[channel_idx, idx] = stimulus_idxs[idx]
                stimulus_idxs_matrix[channel_idx, (idx + 1):n_stimuli] = np.copy(remaining_stimulus_idxs)

    stimulus_idxs = stimulus_idxs.astype(np.int_)
    return stimulus_idxs

def suppress_stimulus_artifacts(data, stimulus_idxs, sampling_time, stimulus_duration=0.002, transient_duration=0.75):
    stimulus_samples = int(np.round(stimulus_duration / sampling_time))
    stimulus_half_samples = int(np.ceil(stimulus_samples / 2))
    stimulus_range = np.arange(-stimulus_half_samples, stimulus_half_samples)

    transient_samples = int(np.ceil(transient_duration / sampling_time))
    transient_range = np.arange(stimulus_half_samples, stimulus_half_samples + transient_samples)

    for idx in np.arange(np.size(stimulus_idxs)):
        stimulus_idx = stimulus_idxs[idx]
        data[stimulus_idx + stimulus_range] = 0
        transient = data[stimulus_idx + transient_range]

        max_idx = np.argmax(transient)
        # Ensure a meaningful max_idx, otherwhise set it null
        max_idx = max_idx[0] if type(max_idx) == np.ndarray else max_idx
        max_idx = 0 if max_idx < 100 else max_idx
        max_idx = np.size(transient) if np.abs(np.size(transient) - max_idx) < 100 else max_idx

        transient_smoothed = np.zeros(np.shape(transient))

        if max_idx != 0:
            left_transient_range = np.arange(stimulus_half_samples, stimulus_half_samples + max_idx) - stimulus_half_samples
            window_length =  int(np.floor(np.size(left_transient_range) / 5) * 2 + 1) # force odd length
            transient_smoothed[left_transient_range] = savgol_filter(transient[left_transient_range], window_length, 3)

        if max_idx != np.size(transient):
            right_transient_range = transient_range[max_idx:] - stimulus_half_samples
            window_length =  int(np.floor(np.size(right_transient_range) / 5) * 2 + 1) # force odd length
            transient_smoothed[right_transient_range] = savgol_filter(transient[right_transient_range], window_length, 3)

        transient_smoothed = savgol_filter(transient_smoothed, 501, 3) # remove left-right discontinuity

        data[stimulus_idx + transient_range] = data[stimulus_idx + transient_range] - transient_smoothed
        
    return data