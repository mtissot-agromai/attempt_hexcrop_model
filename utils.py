import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, linregress, entropy
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq, fft
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List, Optional


def double_logistic(t: np.ndarray, c1: float, c2: float, c3: float, c4: float, c5: float, c6: float, c7: float) -> np.ndarray:
    """
    The 7-parameter double logistic function used to model vegetation index curves.
    
    Parameters:
        t: Time in Days Since Sowing (DSS).
        c1: Baseline value (minimum).
        c2: Amplitude of growth.
        c3: Growth rate (steepness of the rising edge).
        c4: Inflection point of growth (Start of Season, SOS).
        c5: Amplitude of decline (usually near c2).
        c6: Decline rate (steepness of the falling edge).
        c7: Inflection point of decline (End of Season, EOS).
    """
    
    # Growth curve:
    growth = c2 / (1 + np.exp(-c3 * (t - c4)))
    
    # Decline curve (inverted logistic):
    decline = c5 / (1 + np.exp(c6 * (t - c7))) # Note: c6 is positive here for the falling curve
    
    return c1 + growth - decline

# ==============================
# 1) estatisticas normais
# ==============================
# def extract_statistical_features(df: pd.DataFrame, column: str) -> dict:
#     values = df[column].dropna().values
    
#     return {
#         "mean": np.mean(values),
#         "median": np.median(values),
#         "std": np.std(values, ddof=1),
#         "var": np.var(values, ddof=1),
#         "skewness": skew(values),
#         "kurtosis": kurtosis(values),
#         "min": np.min(values),
#         "max": np.max(values),
#         "range": np.max(values) - np.min(values),
#         "q25": np.percentile(values, 25),
#         "q75": np.percentile(values, 75),
#     }
def extract_statistical_features(
    df: pd.DataFrame, 
    column: str, 
    date_column: str = 'date'
) -> dict:
    df = df.dropna(subset=[column, date_column])
    if df.empty:
        return {}
    
    peak_date_idx = df['EVI2'].idxmax()
    df[date_column] = pd.to_datetime(df[date_column])
    peak_date = df['date'].iloc[int(peak_date_idx)].strftime("%Y-%m-%d")

    peak_dt = pd.to_datetime(peak_date)

    data_full = df[column].values
    data_growth = df[df[date_column] <= peak_dt][column].values
    data_decline = df[df[date_column] > peak_dt][column].values

    sowing_val = data_full[0]
    harvest_val = data_full[-1]
    peak_val = data_decline[0]

    results = {}

    def calculate_stats(values: np.ndarray, prefix: str) -> dict:
        if len(values) < 2:
            return {f'{prefix}_{k}': np.nan for k in ["mean", "median", "std", "var", "skewness", "kurtosis", "min", "max", "range", "q25", "q75"]}

        return {
            f'{prefix}_mean': np.mean(values),
            f'{prefix}_median': np.median(values),
            f'{prefix}_std': np.std(values, ddof=1),
            f'{prefix}_var': np.var(values, ddof=1),
            f'{prefix}_skewness': skew(values),
            f'{prefix}_kurtosis': kurtosis(values),
            f'{prefix}_min': np.min(values),
            f'{prefix}_max': np.max(values),
            f'{prefix}_range': np.max(values) - np.min(values),
            f'{prefix}_q25': np.percentile(values, 25),
            f'{prefix}_q75': np.percentile(values, 75),
        }

    results.update(calculate_stats(data_full, 'full'))

    results.update(calculate_stats(data_growth, 'growth'))

    results.update(calculate_stats(data_decline, 'decline'))

    results.update({f"{column}_sowing": sowing_val,
                    f"{column}_peak": peak_val,
                    f"{column}_harvest": harvest_val})
    
    return results


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

    # ac1 = autocorr(values, 1)
    # ac2 = autocorr(values, 2)
    # ac7 = autocorr(values, 7)
    # ac15 = autocorr(values, 15)
    # ac30 = autocorr(values, 30)

    peaks, _ = find_peaks(values)
    valleys, _ = find_peaks(-values)

    frac_above_mean = np.mean(values > np.mean(values))

    q_idx = [int(n * frac) for frac in [0.25, 0.5, 0.75]]
    timeline_vals = [values[i] for i in q_idx if i < n]

    return {
        "trend_slope": slope,
        # "trend_r2": r_value**2,
        # "autocorr_lag1": ac1,
        # "autocorr_lag2": ac2,
        # "autocorr_lag7": ac7,
        # "autocorr_lag15": ac15,
        # "autocorr_lag30": ac30,
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

# ==============================
# 5) "Phenological" features
# ==============================
def extract_phenological_rate_features(
    df: pd.DataFrame, 
    column: str,
    date_column: str = 'date'
) -> dict:
    peak_date_idx = df['EVI2'].idxmax()
    df['date'] = pd.to_datetime(df['date'])
    peak_date = df['date'].iloc[int(peak_date_idx)].strftime("%Y-%m-%d")
    df[date_column] = pd.to_datetime(df[date_column])
    peak_dt = pd.to_datetime(peak_date)

    df_clean = df.dropna(subset=[column, date_column]).sort_values(date_column)
    
    if df_clean.empty:
        return {}
        
    sowing_date = df_clean[date_column].min()
    df_clean['DSS'] = (df_clean[date_column] - sowing_date).dt.days
    
    data_growth = df_clean[df_clean[date_column] <= peak_dt]
    data_decline = df_clean[df_clean[date_column] > peak_dt]
    
    results = {}

    def calculate_rate(data_phase: pd.DataFrame, phase_name: str) -> dict:
        if len(data_phase) < 2:
            return {f'{phase_name}_slope': np.nan, f'{phase_name}_duration': np.nan}
        
        y = data_phase[column].values
        x = data_phase['DSS'].values
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except np.linalg.LinAlgError:
            slope = np.nan
            
        duration = x[-1] - x[0]
        
        return {
            f'{phase_name}_slope': slope,
            f'{phase_name}_duration': duration,
        }

    results.update(calculate_rate(data_growth, 'growth'))
    
    results.update(calculate_rate(data_decline, 'decline'))
    
    overall_amplitude = df_clean[column].max() - df_clean[column].min()
    results['overall_amplitude'] = overall_amplitude
    
    return results

def extract_phenological_features_logistic(
    df: pd.DataFrame, 
    column: str,
    date_column: str = 'date'
) -> dict:
    
    # Use the column data itself to find the approximate peak for initial parameter guessing
    # NOTE: The EVI2 check is removed as we now fit the specified 'column'
    df_clean = df.dropna(subset=[column, date_column]).sort_values(date_column) # THIS CHANGED
    
    if len(df_clean) < 7: # Need at least 7 points for 7 parameters
        return {f'DL_c{i}': np.nan for i in range(1, 8)}

    df_clean[date_column] = pd.to_datetime(df_clean[date_column])
    
    sowing_date = df_clean[date_column].min()
    df_clean['DSS'] = (df_clean[date_column] - sowing_date).dt.days
    
    t = df_clean['DSS'].values
    y = df_clean[column].values
    
    y_min = y.min()
    y_max = y.max()
    t_mid = (t.max() + t.min()) / 2
    
    # Initial Parameter Guesses (p0)
    # Good initial guesses are crucial for stable non-linear fitting.
    c1_init = y_min * 0.95  # Baseline (95% of min)
    c2_init = y_max - y_min # Growth Amplitude
    c3_init = 0.1          # Growth Rate (a guess)
    c4_init = t_mid * 0.5  # Growth Inflection (Start of Season, guessed at 50% through first half)
    c5_init = c2_init      # Decline Amplitude (same as growth amplitude)
    c6_init = 0.1          # Decline Rate (same as growth rate)
    c7_init = t_mid * 1.5  # Decline Inflection (End of Season, guessed at 50% through second half)
    
    p0 = [c1_init, c2_init, c3_init, c4_init, c5_init, c6_init, c7_init] # THIS CHANGED
    
    results = {}
    
    try:
        # Fit the Double Logistic function to the time series
        popt, pcov = curve_fit(double_logistic, t, y, p0=p0, maxfev=5000) # THIS CHANGED
        
        # Store the fitted parameters as features
        for i, param in enumerate(popt):
            results[f'DL_c{i+1}'] = param
            
    except RuntimeError:
        # If the curve fitting fails (often due to non-convergence)
        print(f"Warning: Double Logistic fit failed for column {column}. Returning NaN coefficients.")
        results = {f'DL_c{i}': np.nan for i in range(1, 8)}
        
    # Old simple features are now removed:
    # overall_amplitude = df_clean[column].max() - df_clean[column].min()
    # results['overall_amplitude'] = overall_amplitude
    
    # Clean up any residual keys if the original function had them
    # Note: If you still need simple min/max/amplitude, you should use the general statistical function.
    
    return results # THIS CHANGED

# ==============================
# 6) Harmonic features
# ==============================
def extract_harmonic_features(df: pd.DataFrame,
                              column: str,
                              num_harmonics: int = 3) -> dict:
    values = df[column].dropna().sort_values().values
    N = len(values)
    
    if N < num_harmonics * 2:
        return {f'harmonic_amplitude_{i}': np.nan for i in range(1, num_harmonics + 1)}

    yf = fft(values)
    
    amplitudes = np.abs(yf[1:N//2]) * 2 / N
    
    results = {}
    for i in range(min(num_harmonics, len(amplitudes))):
        results[f'harmonic_amplitude_{i+1}'] = amplitudes[i]
        
    return results

# ==============================================
# 7) Features de correlação cruzada (DTW de pobre)
# ==============================================
def extract_cross_correlation_features(
    df: pd.DataFrame, 
    column: str, 
    prototype_curves: Dict[str, List[float]], 
    date_column: str = 'date'
) -> dict:
    target_prototypes = prototype_curves.get(column, {})

    values = df[column].dropna().values
    N = len(values)
    
    if N == 0:
        return {}

    # Normalize the input time series
    if np.std(values) != 0:
        input_series = (values - np.mean(values)) / np.std(values)
    else:
        input_series = values - np.mean(values) # Just centering
        
    results = {}

    for name, prototype in target_prototypes.items():
        if len(prototype) != N:
            results[f'corr_max_{name}'] = np.nan
            results[f'corr_lag_{name}'] = np.nan
            continue

        if np.std(prototype) != 0:
            proto_series = (np.array(prototype) - np.mean(prototype)) / np.std(prototype)
        else:
            proto_series = np.array(prototype) - np.mean(prototype)

        correlation = np.correlate(input_series, proto_series, mode='full')
        
        max_corr_index = np.argmax(correlation)
        max_corr = correlation[max_corr_index]
        lag = max_corr_index - (N - 1)
        
        results[f'corr_max_{name}'] = max_corr
        results[f'corr_lag_{name}'] = lag
        
    return results

def extract_all_features(df: pd.DataFrame, column: str, master_prototypes = None) -> dict:
    PHENOLOGICAL_COLUMNS = ['B6', 'B7', 'B8', 'B8A', 'B9', 'NDVI', 'EVI2', 'NDMI']

    features = {}
    features.update(extract_statistical_features(df, column))
    # features.update(extract_time_features(df, column))
    # features.update(extract_frequency_features(df, column))
    features.update(extract_shape_features(df, column))
    if column in PHENOLOGICAL_COLUMNS:
        features.update(extract_phenological_features_logistic(df, column))
    features.update(extract_phenological_rate_features(df, column))
    features.update(extract_harmonic_features(df, column))
    # features.update(extract_cross_correlation_features(df, column, master_prototypes))
    return features