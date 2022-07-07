import numpy as np

def shape_trials(data, stimulus_idxs, trial_duration, sampling_time):
    trial_samples = int(np.ceil(trial_duration / sampling_time))

    trials = np.ndarray([np.size(stimulus_idxs), trial_samples])
    trial_range = np.arange(0, trial_samples)

    for idx in range(np.size(stimulus_idxs)):
        stimulus_idx = stimulus_idxs[idx]
        trials[idx] = data[stimulus_idx + trial_range]

    return trials