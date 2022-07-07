import h5py
import numpy as np
import neurospyke as ns

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