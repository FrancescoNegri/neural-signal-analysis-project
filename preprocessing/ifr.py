import numpy as np

def get_IFR(spike_train, bin_duration, sampling_time):
    bin_samples = int(np.ceil(bin_duration / sampling_time))

    n_bins = int(np.floor(np.size(spike_train, axis=0) / bin_samples))
    IFR = np.zeros(np.shape(spike_train))

    for bin_idx in range(n_bins):
        bin_range = np.arange(bin_idx * bin_samples, bin_idx * bin_samples + bin_samples)
        bin = spike_train[bin_range]
        bin_spike_count = np.sum(bin)
        IFR[bin_range] = bin_spike_count / bin_duration

    return IFR, bin_samples