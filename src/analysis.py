import numpy as np
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
from src.utils import detect_outliers_rms, avg_psd, avg_spectrogram

class Analysis():
    def __init__(self, num_sessions, DATA, fs, stim_onset_ms, stim_offset_ms, config):
        
        self.num_sessions = num_sessions
        self.DATA = DATA
        self.fs = fs
        self.stim_onset_ms = stim_onset_ms
        self.stim_offset_ms = stim_offset_ms

        self.cutoff_hz = config['cutoff_hz']# 1000.0
        self.nyquist = config['nyquist']# fs / 2.0
        self.order = config['order']# 10

        self.low_pass_data = self.filtering(self.num_sessions, self.DATA)
        
        self.clean_low, self.clean_high = self.preprocess(self.num_sessions, self.low_pass_data, self.DATA)

    def filtering(self, num_sessions, DATA):
        Wn = self.cutoff_hz / self.nyquist
        b, a = butter(self.order, Wn, btype='low', analog=False)

        low_pass_data = []

        for session_idx in range(num_sessions):
            session_signals = DATA[session_idx, 0]  # shape: (n_trials, n_samples)
            n_trials, n_samples = session_signals.shape

            filtered_session = np.zeros_like(session_signals)

            for trial_idx in range(n_trials):
                raw_trial = session_signals[trial_idx, :]   # one trial (1D)
                filtered_trial = filtfilt(b, a, raw_trial)  # zero-phase filtering
                filtered_session[trial_idx, :] = filtered_trial

            low_pass_data.append(filtered_session)
        return low_pass_data

    def preprocess(self, num_sessions, low_pass_data, DATA):
        clean_low = []   # clean_low[session] -> (n_trials_clean, n_samples)
        clean_high = []

        for session_idx in range(num_sessions):
            session_signals = low_pass_data[session_idx]    # filtered signals
            tone_labels = DATA[session_idx, 4]

            # low / high tone 구분
            unique_tones = np.unique(tone_labels)
            low_tone_code = np.min(unique_tones)
            high_tone_code = np.max(unique_tones)

            low_idx = np.where(tone_labels == low_tone_code)[0]
            high_idx = np.where(tone_labels == high_tone_code)[0]

            low_trials = session_signals[low_idx, :]
            high_trials = session_signals[high_idx, :]

            # removing outliers
            low_keep_mask, _, _ = detect_outliers_rms(low_trials, z_thresh=3.0)
            high_keep_mask, _, _ = detect_outliers_rms(high_trials, z_thresh=3.0)

            clean_low.append(low_trials[low_keep_mask, :])
            clean_high.append(high_trials[high_keep_mask, :])

        return clean_low, clean_high
    
    def psd(self, ):

        for session_idx in range(self.num_sessions):
            low_trials = self.clean_low[session_idx]
            high_trials = self.clean_high[session_idx]

            n_low = low_trials.shape[0]
            n_high = high_trials.shape[0]
            n_samples = low_trials.shape[1]

            time_ms = np.arange(n_samples) / self.fs * 1000.0

            # avg and SEM
            mean_low = np.mean(low_trials, axis=0)
            mean_high = np.mean(high_trials, axis=0)

            sem_low = np.std(low_trials, axis=0, ddof=1) / np.sqrt(n_low)
            sem_high = np.std(high_trials, axis=0, ddof=1) / np.sqrt(n_high)

            f_low, psd_low = avg_psd(low_trials, self.fs)
            f_high, psd_high = avg_psd(high_trials, self.fs)

            # plotting time area (ERP) + freq area (PSD)
            plt.figure(figsize=(12, 6))
            plt.suptitle(f"Session {session_idx + 1} - Method 1: ERP & PSD", fontsize=14)

            # time area
            plt.subplot(1, 2, 1)
            plt.fill_between(time_ms, mean_low - sem_low, mean_low + sem_low,
                            alpha=0.3, label=f"Low tone (n={n_low})")
            plt.plot(time_ms, mean_low, label="Low tone mean")
            plt.fill_between(time_ms, mean_high - sem_high, mean_high + sem_high,
                            alpha=0.3, label=f"High tone (n={n_high})")
            plt.plot(time_ms, mean_high, label="High tone mean")

            # (100–150 ms)
            plt.axvspan(self.stim_onset_ms, self.stim_offset_ms, color="gray", alpha=0.1)

            plt.xlabel("Time (ms)")
            plt.ylabel("LFP (a.u.)")
            plt.title("Average LFP (ERP) with SEM")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # freq area
            plt.subplot(1, 2, 2)
            plt.semilogy(f_low, psd_low, label="Low tone")
            plt.semilogy(f_high, psd_high, label="High tone")
            plt.xlim(0, 200) 
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (power/Hz)")
            plt.title("Welch Power Spectrum")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    
    def lfp(self, ):

        for session_idx in range(self.num_sessions):
            low_trials = self.clean_low[session_idx]
            high_trials = self.clean_high[session_idx]

            nperseg = 256
            noverlap = 255
            nfft = np.arange(0, 200+5, 5)   # 0~200 Hz, 5 Hz step 

            nfft = 512

            f_low, t_low, S_low = avg_spectrogram(low_trials, self.fs, nperseg, noverlap, nfft)
            f_high, t_high, S_high = avg_spectrogram(high_trials, self.fs, nperseg, noverlap, nfft)

            # second -> ms
            t_low_ms = t_low * 1000.0
            t_high_ms = t_high * 1000.0

            # low / high tone spectrogram
            plt.figure(figsize=(10, 8))
            plt.suptitle(f"Session {session_idx + 1} - Method 2: Time-Frequency (Spectrogram)", fontsize=14)

            plt.subplot(2, 1, 1)
            plt.pcolormesh(t_low_ms, f_low, S_low, shading='gouraud')
            plt.colorbar(label='Power (dB)')
            plt.axvspan(self.stim_onset_ms, self.stim_offset_ms, color="white", alpha=0.2)
            plt.ylim(0, 200)
            plt.xlabel("Time (ms)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Low tone - average spectrogram")

            plt.subplot(2, 1, 2)
            plt.pcolormesh(t_high_ms, f_high, S_high, shading='gouraud')
            plt.colorbar(label='Power (dB)')
            plt.axvspan(self.stim_onset_ms, self.stim_offset_ms, color="white", alpha=0.2)
            plt.ylim(0, 200)
            plt.xlabel("Time (ms)")
            plt.ylabel("Frequency (Hz)")
            plt.title("High tone - average spectrogram")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def combined(self, ):
        all_low = np.concatenate(self.clean_low, axis=0)    # shape: (N_low_total, n_samples)
        all_high = np.concatenate(self.clean_high, axis=0)  # shape: (N_high_total, n_samples)

        n_low_all, n_samples = all_low.shape
        n_high_all, _ = all_high.shape

        time_ms = np.arange(n_samples) / self.fs * 1000.0

        # ERP + PSD (for all 4 session data)
        # avg and SEM
        mean_low_all = np.mean(all_low, axis=0)
        mean_high_all = np.mean(all_high, axis=0)

        sem_low_all = np.std(all_low, axis=0, ddof=1) / np.sqrt(n_low_all)
        sem_high_all = np.std(all_high, axis=0, ddof=1) / np.sqrt(n_high_all)

        f_low_all, psd_low_all = avg_psd(all_low, self.fs)
        f_high_all, psd_high_all = avg_psd(all_high, self.fs)


        # 그림: time area ERP + freq area PSD
        plt.figure(figsize=(12, 6))
        plt.suptitle("All Sessions Combined - Method 1: ERP & PSD", fontsize=14)

        # time area (ERP)
        plt.subplot(1, 2, 1)
        plt.fill_between(time_ms,
                        mean_low_all - sem_low_all,
                        mean_low_all + sem_low_all,
                        alpha=0.3,
                        label=f"Low tone (N={n_low_all})")
        plt.plot(time_ms, mean_low_all, label="Low tone mean")

        plt.fill_between(time_ms,
                        mean_high_all - sem_high_all,
                        mean_high_all + sem_high_all,
                        alpha=0.3,
                        label=f"High tone (N={n_high_all})")
        plt.plot(time_ms, mean_high_all, label="High tone mean")

        plt.axvspan(self.stim_onset_ms, self.stim_offset_ms, color="gray", alpha=0.1)
        plt.xlabel("Time (ms)")
        plt.ylabel("LFP (a.u.)")
        plt.title("Average LFP (ERP) with SEM - All Sessions")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # freq area (PSD)
        plt.subplot(1, 2, 2)
        plt.semilogy(f_low_all, psd_low_all, label="Low tone")
        plt.semilogy(f_high_all, psd_high_all, label="High tone")
        plt.xlim(0, 200)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (power/Hz)")
        plt.title("Welch Power Spectrum - All Sessions")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def save(self, file_name):
        raw_data_3d = np.stack([self.DATA[i, 0] for i in range(self.num_sessions)], axis=0)
        low_pass_3d = np.stack(self.low_pass_data, axis=0)

        n_sessions, n_trials, n_samples = raw_data_3d.shape
        time_ms = np.arange(n_samples) / self.fs * 1000.0

        # ERP per session (clean_low / clean_high)
        erp_mean_low_sessions = np.zeros((self.num_sessions, n_samples))
        erp_mean_high_sessions = np.zeros((self.num_sessions, n_samples))

        psd_f = None
        psd_low_sessions = []
        psd_high_sessions = []

        for session_idx in range(self.num_sessions):
            low_trials = self.clean_low[session_idx]   # (n_low_clean, n_samples)
            high_trials = self.clean_high[session_idx] # (n_high_clean, n_samples)

            # ERP (mean)
            erp_mean_low_sessions[session_idx, :] = np.mean(low_trials, axis=0)
            erp_mean_high_sessions[session_idx, :] = np.mean(high_trials, axis=0)

            # PSD
            f_l, psd_l = avg_psd(low_trials, self.fs)
            f_h, psd_h = avg_psd(high_trials, self.fs)

            if psd_f is None:
                psd_f = f_l  

            psd_low_sessions.append(psd_l)
            psd_high_sessions.append(psd_h)

        psd_low_sessions = np.stack(psd_low_sessions, axis=0)   # (session, freq)
        psd_high_sessions = np.stack(psd_high_sessions, axis=0) # (session, freq)

        # computing ERP, PSD for total 4 session trials
        # ------------------------------------------------
        all_low = np.concatenate(self.clean_low, axis=0)    # (N_low_total, n_samples)
        all_high = np.concatenate(self.clean_high, axis=0)  # (N_high_total, n_samples)

        erp_mean_low_all = np.mean(all_low, axis=0)
        erp_mean_high_all = np.mean(all_high, axis=0)

        f_low_all, psd_low_all = avg_psd(all_low, self.fs)
        f_high_all, psd_high_all = avg_psd(all_high, self.fs)


        # dictionary for saving the results

        results_dict = {
            "fs": self.fs,
            "stim_onset_ms": self.stim_onset_ms,
            "stim_offset_ms": self.stim_offset_ms,

            "raw_data": raw_data_3d,         # (session, trial, time)
            "low_pass_data": low_pass_3d,    # (session, trial, time)

            "clean_low": np.array(self.clean_low, dtype=object),
            "clean_high": np.array(self.clean_high, dtype=object),

            # time-area ERP
            "time_ms": time_ms,
            "erp_mean_low_sessions": erp_mean_low_sessions,   # (session, time)
            "erp_mean_high_sessions": erp_mean_high_sessions,  # (session, time)
            "erp_mean_low_all": erp_mean_low_all,             # (time,)
            "erp_mean_high_all": erp_mean_high_all,           # (time,)

            # freq area PSD
            "psd_f": psd_f,                         # (freq,)
            "psd_low_sessions": psd_low_sessions,   # (session, freq)
            "psd_high_sessions": psd_high_sessions,  # (session, freq)
            "psd_low_all": psd_low_all,             # (freq,)
            "psd_high_all": psd_high_all,           # (freq,)
        }

        #saving 
        np.savez(file_name, **results_dict)

        print(f"Saved: {file_name}")
