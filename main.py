import numpy as np
from scipy.io import loadmat
import os 
from src.sampler import outlier_sample_rejection
from src.analysis import Analysis

######################
# Setting parameters #zZ
######################
cutoff_frequency = 1e3
bin_width = 5
max_freq = 200
num_trials = 200
num_sessions = 4
fs = 1e4

# 100, 150 ms
stim_onset_ms = 100
stim_offset_ms = 150

stim_onset_sample = int(stim_onset_ms / 1000 * fs)
stim_offset_sample = int(stim_offset_ms / 1000 * fs)

#######################
# loading the dataset #
#######################
mat = loadmat("data/mouseLFP.mat", squeeze_me=True, struct_as_record=False)
DATA = mat["DATA"]

# assigning appropriate variables

num_sessions = DATA.shape[0]

example_session_signals = DATA[0, 0]   # shape: (n_trials, n_samples)
n_trials_per_session = example_session_signals.shape[0]
n_samples = example_session_signals.shape[1]


############################
# outlier sample rejection #
############################

outlier_sampler = outlier_sample_rejection(num_sessions, DATA)

per_session_counts, total_low_after, total_high_after = outlier_sampler.analyze()

print("=== # trial per session (per-tone, berfore/after rejecting outliers) ===")
for summary in per_session_counts:
    s = summary["session"]
    print(f"[Session {s}]")
    print(f"  Low tone  : before = {summary['low_before']:3d}, "
          f"after = {summary['low_after']:3d}, "
          f"rejected = {summary['low_rejected']:3d}")
    print(f"  High tone : before = {summary['high_before']:3d}, "
          f"after = {summary['high_after']:3d}, "
          f"rejected = {summary['high_rejected']:3d}")
    print()

print()
print(f"The number of trials of low tone: {total_low_after}")
print(f"The number of trials of high tone: {total_high_after}")


#############
# filtering #
#############

config = {
    "cutoff_hz": 1000.0,
    "nyquist": fs / 2.0,
    "order": 10
}

anal = Analysis(num_sessions, DATA, fs, stim_onset_ms, stim_offset_ms, config)

##################
# method 1 - PSD #
##################
anal.psd()

##################
# method 1 - LFP #
##################
anal.lfp()

##################
# method 1 - LFP #
##################
save_foldername = "output/"
save_filename = "lfp_analysis_results.npz"
save_pth = os.path.join(save_foldername, save_filename)
if not os.path.exists(save_foldername):
    os.mkdir(save_foldername)
anal.save(save_pth)

