import numpy as np
from scipy.signal import welch, spectrogram

def detect_outliers_rms(trials_2d, z_thresh=3.0):
        # RMS for each trial
        rms = np.sqrt(np.mean(trials_2d**2, axis=1))  # (n_trials,)

        # z-score 
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        z = (rms - mean_rms) / (std_rms + 1e-12)

        keep_mask = np.abs(z) <= z_thresh

        return keep_mask, rms, z

def avg_psd(trials, fs, nperseg=1024):
        psd_acc = None
        for tr in trials:
            f, Pxx = welch(tr, fs=fs, nperseg=nperseg)
            if psd_acc is None:
                psd_acc = Pxx
            else:
                psd_acc += Pxx
        psd_avg = psd_acc / trials.shape[0]
        return f, psd_avg

def avg_spectrogram(trials, fs, nperseg, noverlap, nfft):
        S_acc = None
        for tr in trials:
            f, t_spec, Sxx = spectrogram(tr, fs=fs, nperseg=nperseg,
                                        noverlap=noverlap, nfft=nfft,
                                        scaling='density', mode='psd')
            if S_acc is None:
                S_acc = Sxx
            else:
                S_acc += Sxx
        S_avg = S_acc / trials.shape[0]
        # converting to dB scale
        S_db = 10 * np.log10(S_avg + 1e-20)
        return f, t_spec, S_db