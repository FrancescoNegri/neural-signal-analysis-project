from turtle import st
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def run_SALPA(data, sampling_time, stimulus_samples = 100, transient_duration = 0.7, peaks_height = 2000, peaks_distance = 4):
    sampling_frequency = 1 / sampling_time
    
    stimulus_idxs, _ = find_peaks(np.abs(data), height=peaks_height, distance=peaks_distance * sampling_frequency)

    stimulus_half_samples = int(np.ceil(stimulus_samples / 2))
    stimulus_range = np.arange(-stimulus_half_samples, stimulus_half_samples)

    transient_samples = int(np.ceil(transient_duration * sampling_frequency))
    transient_range = np.arange(stimulus_half_samples, stimulus_half_samples + transient_samples)
    transients = []

    for stimulus_idx in stimulus_idxs:
        data[stimulus_idx + stimulus_range] = 0
        transients.append(data[stimulus_idx + transient_range])

    mean_transient = np.mean(transients, 0)
    
    zero_idx = np.where(mean_transient >= 0)[0][0]
    extended_stimulus_range = np.arange(stimulus_half_samples, stimulus_half_samples + zero_idx)
    transients = []

    for stimulus_idx in stimulus_idxs:
        data[stimulus_idx + extended_stimulus_range] = 0
        transients.append(data[stimulus_idx + transient_range])

    mean_transient = np.mean(transients, 0)
    mean_transient_smoothed = savgol_filter(mean_transient, 1001, 3)

    for stimulus_idx in stimulus_idxs:
        data[stimulus_idx + transient_range] = data[stimulus_idx + transient_range] - mean_transient_smoothed

    remaining_stimulus_idxs, _ = find_peaks(np.abs(data), height=peaks_height, distance=peaks_distance * sampling_frequency)
    if len(remaining_stimulus_idxs) > 0:
        data, _ = run_SALPA(data, sampling_time, stimulus_samples, transient_duration, peaks_height, peaks_distance)

    return data, stimulus_idxs
 