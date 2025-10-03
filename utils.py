import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, linregress, entropy
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

# ==============================
# 1) estatisticas normais
# ==============================
def extract_statistical_features(df: pd.DataFrame, column: str) -> dict:
    values = df[column].dropna().values
    
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values, ddof=1),
        "var": np.var(values, ddof=1),
        "skewness": skew(values),
        "kurtosis": kurtosis(values),
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75),
    }


# ==============================
# 2) Coisas de serie temporal
# ==============================
def extract_time_features(df: pd.DataFrame, column: str) -> dict:
    values = df[column].dropna().values
    n = len(values)
    time_idx = np.arange(n)

    slope, intercept, r_value, p_value, std_err = linregress(time_idx, values)

    def autocorr(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag < len(x) else np.nan

    ac1 = autocorr(values, 1)
    ac2 = autocorr(values, 2)
    ac7 = autocorr(values, 7)
    ac15 = autocorr(values, 15)
    ac30 = autocorr(values, 30)

    peaks, _ = find_peaks(values)
    valleys, _ = find_peaks(-values)

    frac_above_mean = np.mean(values > np.mean(values))

    q_idx = [int(n * frac) for frac in [0.25, 0.5, 0.75]]
    timeline_vals = [values[i] for i in q_idx if i < n]

    return {
        "trend_slope": slope,
        "trend_r2": r_value**2,
        "autocorr_lag1": ac1,
        "autocorr_lag2": ac2,
        "autocorr_lag7": ac7,
        "autocorr_lag15": ac15,
        "autocorr_lag30": ac30,
        "num_peaks": len(peaks),
        "num_valleys": len(valleys),
        "frac_above_mean": frac_above_mean,
        "value_at_25pct": timeline_vals[0] if len(timeline_vals) > 0 else np.nan,
        "value_at_50pct": timeline_vals[1] if len(timeline_vals) > 1 else np.nan,
        "value_at_75pct": timeline_vals[2] if len(timeline_vals) > 2 else np.nan,
    }


# ==============================
# 3) Features espectrais
# ==============================
def extract_frequency_features(df: pd.DataFrame, column: str, sampling_rate: float = 1.0) -> dict:
    values = df[column].dropna().values
    n = len(values)

    yf = np.abs(rfft(values))
    xf = rfftfreq(n, d=1/sampling_rate)

    psd = yf**2
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)

    dominant_freq = xf[np.argmax(psd)] if n > 0 else np.nan
    spectral_entropy = entropy(psd_norm) if np.sum(psd_norm) > 0 else np.nan

    cutoff = len(psd) // 4
    low_power = np.sum(psd[:cutoff])
    high_power = np.sum(psd[cutoff:])

    return {
        "dominant_frequency": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "low_freq_power": low_power,
        "high_freq_power": high_power,
    }


# ==============================
# 4) Forma
# ==============================
def extract_shape_features(df: pd.DataFrame, column: str) -> dict:
    values = df[column].dropna().values
    n = len(values)

    auc = np.trapz(values)

    first_diff = np.diff(values)
    second_diff = np.diff(first_diff)
    inflections = np.sum(np.diff(np.sign(second_diff)) != 0)

    if np.ptp(values) > 0:
        normalized = (values - np.min(values)) / np.ptp(values)
    else:
        normalized = values

    return {
        "auc": auc,
        "num_inflections": inflections,
        "normalized_mean": np.mean(normalized),
        "normalized_std": np.std(normalized),
    }

def extract_all_features(df: pd.DataFrame, column: str) -> dict:
    features = {}
    features.update(extract_statistical_features(df, column))
    features.update(extract_time_features(df, column))
    features.update(extract_frequency_features(df, column))
    features.update(extract_shape_features(df, column))
    return features