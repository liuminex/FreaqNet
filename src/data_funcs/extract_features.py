import os
import sys
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # root of IntermitNet

# frequency -> seasonal period (None = no seasonality)
PERIODS = {
    "year": None,
    "quarter": 4,
    "month": 12,
    "day": 7,       # weekly seasonality for daily data
    "hour": 24,     # daily seasonality for hourly data
    "15 min": 96,   # daily seasonality for 15-min data
    "30 min": 48,   # daily seasonality for 30-min data
}
SEASONAL_VECTOR_LEN = 12
MAX_HISTORY_PERIODS = 10   # cap history so per-window extraction stays cheap and identical in train/val/test



FREQ_TO_ID = {"year": 0, "quarter": 1, "month": 2, "day": 3, "hour": 4, "15 min": 5, "30 min": 6}


EXOG_DIM = 2 + SEASONAL_VECTOR_LEN   # [freq_id, period, sidx_1..sidx_12] = 14


def extract_all_features(series_values, frequency):
    """
    Strictly causal exogs for ONE forecast origin.
    Returns float array: [freq_id, period, sidx_1, ..., sidx_12]  (length EXOG_DIM)
    """
    if frequency not in FREQ_TO_ID:
        raise ValueError(f"Unknown frequency '{frequency}'.")
    x = np.asarray(series_values, dtype=np.float64)
    seasonal = get_seasonal_indices(x, frequency, SEASONAL_VECTOR_LEN)
    return np.concatenate([[float(FREQ_TO_ID[frequency])],
                           [float(PERIODS[frequency] or 0)],
                           seasonal])



def get_seasonal_indices(series_values, frequency, vector_len=12):
    """
    Seasonal indices for the NEXT `vector_len` steps after the end of the series.
    Entry h is the (scaled) seasonal effect of forecast step h+1, so the vector is
    phase-aligned to the forecast origin regardless of where the window starts.
    """
    if frequency not in PERIODS:
        raise ValueError(f"Unknown frequency '{frequency}' for seasonal index extraction.")

    period = PERIODS[frequency]
    zeros = np.zeros(vector_len)
    if period is None:
        return zeros

    x = np.asarray(series_values, dtype=np.float64)[-MAX_HISTORY_PERIODS * period:]

    if len(x) < 2 * period or not np.all(np.isfinite(x)):
        return zeros

    std = np.std(x)
    if std < 1e-12:  # constant series
        return zeros

    try:
        result = seasonal_decompose(x, period=period, model="additive", extrapolate_trend="freq")
    except Exception:
        return zeros

    cycle = np.asarray(result.seasonal[:period])        # index = phase relative to start of x
    cycle = (cycle - cycle.mean()) / std                 # scale by series std (keeps seasonality strength)

    # phase of the next vector_len points after the end of x
    steps = (len(x) + np.arange(vector_len)) % period
    return cycle[steps]                                  # tiles when period < vector_len, truncates when larger


def calculate_acf_features(series, nlags=24):
    """ACF for lags 1..nlags (zero-padded / NaN-safe)."""
    x = np.asarray(series, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12:
        return np.zeros(nlags)
    vals = acf(x, nlags=nlags, fft=True, missing="conservative")[1:nlags + 1]
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(nlags)
    out[:len(vals)] = vals
    return out


def calculate_advanced_features(series_values):
    """
    Returns np.array([entropy, hurst, nonlinearity, lumpiness, stability]).
    Defaults are returned for short/constant series or on numerical failure.
    """
    default = np.array([0.0, 0.5, 0.0, 0.0, 0.0])
    x = np.asarray(series_values, dtype=np.float64)
    n = len(x)
    if n < 10 or not np.all(np.isfinite(x)) or np.all(x == x[0]):
        return default

    try:
        # 1. Spectral entropy (demeaned so the DC term doesn't dominate; normalised to [0, 1])
        power = np.abs(np.fft.rfft(x - x.mean())) ** 2
        power = power / (power.sum() + 1e-12)
        entropy = -np.sum(power * np.log2(power + 1e-12)) / np.log2(len(power))

        # 2. Hurst proxy (variogram method: std of lag-differences ~ lag^H; this is NOT R/S)
        lags = np.arange(2, min(n // 2, 20))
        tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        if np.any(tau <= 0):
            hurst = 0.5
        else:
            hurst = float(np.clip(np.polyfit(np.log(lags), np.log(tau), 1)[0], 0.0, 1.0))

        # 3. Nonlinearity proxy
        d1 = np.diff(x)
        d2 = np.diff(x, n=2)
        nonlinearity = np.std(d2) / (np.std(d1) + 1e-6)

        # 4. Stability / lumpiness on non-overlapping chunks
        width = 4 if n < 24 else 12
        num_chunks = n // width
        if num_chunks > 1:
            chunks = x[:num_chunks * width].reshape(num_chunks, width)
            global_var = np.var(x) + 1e-6
            stability = np.var(chunks.mean(axis=1)) / global_var
            lumpiness = np.var(chunks.var(axis=1)) / (global_var ** 2)
        else:
            stability = lumpiness = 0.0

        out = np.array([entropy, hurst, nonlinearity, lumpiness, stability], dtype=np.float64)
        return out if np.all(np.isfinite(out)) else default
    except Exception:
        return default