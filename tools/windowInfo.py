import numpy as np
from scipy.signal import welch
from .reflectData import reflectData
from typing import List, Tuple
from tools import function_register
from .registerData import getRegisteredData
from .polar import index2name, name2index
from skimage.metrics import structural_similarity as ssim



@function_register.register(
    description="need end - start <= 60. Compute amplitude features (mean absolute, RMS, peak values) of specified EEG channels in a given time window.",
    parameters=[
        {
            "name": "name",
            "type": "List[str]",
            "description": "List of channel names to analyze, e.g., ['FP1-F7', 'F7-T3']"
        },
        {
            "name": "start",
            "type": "int",
            "description": "Start time in seconds."
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time in seconds."
        }
    ],
    returns={
        "type": "Dict[str, Dict[str, float]]",
        "description": "A dictionary where keys are channel names and values are dictionaries of amplitude features including: mean_abs, rms, max_val, min_val"
    }
)
def compute_amplitude(name: List[str], start: int, end: int, config):
    if end - start > 60:
        raise ValueError("The time interval between start and end should not exceed 60 seconds.")
    data = getRegisteredData(start, end, config)
    ids = sorted([name2index[i] for i in name])
    data = data[ids]
    chs = [index2name[i] for i in ids]
    amplitude_features = {}
    for ch_name, ch_data in zip(chs, data):
        amplitude_features[ch_name] = {
            "mean_abs": "{:.2e}".format(np.mean(np.abs(ch_data))),
            "rms": "{:.2e}".format(np.sqrt(np.mean(np.square(ch_data)))),
            "max_val": "{:.2e}".format(np.max(ch_data)),
            "min_val": "{:.2e}".format(np.min(ch_data))
        }
    return amplitude_features


@function_register.register(
    description="need end - start <= 60. Compute power spectral density (PSD) of specified EEG channels in a given time window and integrate by frequency bands.",
    parameters=[
        {
            "name": "name",
            "type": "List[str]",
            "description": "List of channel names to analyze, e.g., ['FP1-F7', 'F7-T3']"
        },
        {
            "name": "start",
            "type": "int",
            "description": "Start time in seconds."
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time in seconds."
        }
    ],
    returns={
        "type": "Dict[str, Dict[str, float]]",
        "description": "A dictionary where keys are channel names and values are dictionaries of power values for each frequency band, unit: V²"
    }
)
def compute_psd(name: List[str], start: int, end: int, config):
    if end - start > 60:
        raise ValueError("The time interval between start and end should not exceed 60 seconds.")
    data = getRegisteredData(start, end, config)
    ids = sorted([name2index[i] for i in name])
    data = data[ids]
    chs = [index2name[i] for i in ids]
    fs = config['fs']
    freq_bands = config.get('prior knowledge')['freq_bands']
    psds = {}
    for ch_name, ch_data in zip(chs, data):
        f, Pxx = welch(ch_data, fs=fs, nperseg=fs)
        band_power = {}
        for band_name, (low, high) in freq_bands.items():
            idx_band = np.logical_and(f >= low, f <= high)
            band_power[band_name] = "{:.2e}".format(np.trapz(Pxx[idx_band], f[idx_band])) # 积分
        psds[ch_name] = band_power
    return psds


@function_register.register(
    description="need end - start <= 60. Calculate the Pearson correlation coefficient between specified left and right channel pairs within a given time window to assess EEG signal symmetry.",
    parameters=[
        {
            "name": "channel_pairs",
            "type": "List[Tuple[str, str]]",
            "description": "List of left-right channel pairs, e.g., ['FP1-F7', 'FP2-F8')]. ",
        },
        {
            "name": "start",
            "type": "int",
            "description": "Start time (in seconds)."
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time (in seconds)."
        }
    ],
    returns={
        "type": "object",
        "description": "A dictionary with channel pairs as keys and Pearson correlation coefficients, Structural Similarity Index as values."
    }
)
def compute_symmetry(channel_pairs: List[Tuple[str, str]], start: int, end: int, config):
    if end - start > 60:
        raise ValueError("The time interval between start and end should not exceed 60 seconds.")
    if channel_pairs is None:
        channel_pairs = config.get("symmetry_channel", [])
    
    symmetry_scores = {}
    data = getRegisteredData(start, end, config)

    for left_name, right_name in channel_pairs:
        left_data = data[name2index[left_name]]
        right_data = data[name2index[right_name]]
        corr_coef = np.corrcoef(left_data, right_data)[0, 1]
        ssim_score = ssim(left_data, right_data, data_range=right_data.max() - right_data.min())
        pair_key = f"{left_name} ↔ {right_name}"
        symmetry_scores[pair_key] = {
            "corr_coef": "{:.2e}".format(corr_coef),
            "ssim": "{:.2e}".format(ssim_score)
        }
    
    return symmetry_scores
