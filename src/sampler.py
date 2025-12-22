import numpy as np
from src.utils import detect_outliers_rms


class outlier_sample_rejection():

    def __init__(self, num_sessions, DATA):
        self.num_sessions = num_sessions
        self.DATA = DATA
    
    
    def analyze(self, ):
        per_session_counts = [] 
        total_low_after = 0
        total_high_after = 0


        for session_idx in range(self.num_sessions):
            # LFP data for session (trial x time)
            session_signals = self.DATA[session_idx, 0]
            # tone 정보 (low / high tone)
            tone_labels = self.DATA[session_idx, 4]

            # tone
            unique_tones = np.unique(tone_labels)
            low_tone_code = np.min(unique_tones)
            high_tone_code = np.max(unique_tones)

            # trial index correspoing to the tone
            low_indices = np.where(tone_labels == low_tone_code)[0]
            high_indices = np.where(tone_labels == high_tone_code)[0]

            # extracting raw data for each toen
            low_trials = session_signals[low_indices, :]   # shape: (n_low_trials, n_samples)
            high_trials = session_signals[high_indices, :] # shape: (n_high_trials, n_samples)

            # low tone
            low_keep_mask, low_rms, low_z = detect_outliers_rms(low_trials, z_thresh=3.0)
            low_clean = low_trials[low_keep_mask, :]

            n_low_before = low_trials.shape[0]
            n_low_after = low_clean.shape[0]
            n_low_rejected = n_low_before - n_low_after

            # high tone
            high_keep_mask, high_rms, high_z = detect_outliers_rms(high_trials, z_thresh=3.0)
            high_clean = high_trials[high_keep_mask, :]

            n_high_before = high_trials.shape[0]
            n_high_after = high_clean.shape[0]
            n_high_rejected = n_high_before - n_high_after

            per_session_counts.append({
                "session": session_idx + 1,
                "low_before": n_low_before,
                "low_after": n_low_after,
                "low_rejected": n_low_rejected,
                "high_before": n_high_before,
                "high_after": n_high_after,
                "high_rejected": n_high_rejected
            })

            total_low_after += n_low_after
            total_high_after += n_high_after

        return per_session_counts, total_low_after, total_high_after
