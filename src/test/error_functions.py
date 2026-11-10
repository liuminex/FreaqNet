import numpy as np
import os


os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp

# error formulas from the paper: https://www.sciencedirect.com/science/article/pii/S0169207019301128


def smape_series(y, y_hat):
    y_hat = np.array(y_hat).flatten()
    y = np.array(y).flatten()
    denominator = (np.abs(y) + np.abs(y_hat)) / 2.0
    diff = np.abs(y - y_hat) / denominator
    smape_value = np.mean(diff) * 100
    return smape_value

def smape(y,y_hat):

    """
    Note that sMAPE and MASE are first estimated for each series by averaging the error computed for
    each forecasting horizon, then averaged again across all time series to compute
    the average value for the entire dataset.
    """
    smape_values = []
    for y_series, y_hat_series in zip(y, y_hat):
        y_series = np.array(y_series).flatten()
        y_hat_series = np.array(y_hat_series).flatten()

        smape_value = smape_series(y_series, y_hat_series)
        smape_values.append(smape_value)

    return np.mean(smape_values), smape_values


import numpy as np

# M4 horizon -> seasonality
H_TO_M = {
    18: 12,  # monthly
    8: 4,    # quarterly
    48: 24,  # hourly
    6: 1,    # yearly
    13: 1,   # weekly
    14: 1,   # daily
    30: 1,   # weather daily
}

def mase(set_name, ts_generator, y_hat, h, m=None):
    """
    MASE: Mean Absolute Scaled Error, averaged over all series.

    set_name:     passed to ts_generator to select the split
    ts_generator: callable yielding full series (context + last h points as pandas Series)
    y_hat:        iterable of forecasts, one per series, each of length h
    h:            forecast horizon (e.g. 6 yearly, 8 quarterly, 18 monthly)
    m:            optional seasonality override; if None, it is looked up from h

    Returns (mean MASE, list of per-series MASE).

    All series must be included: a length mismatch between series and
    forecasts, or a degenerate series, raises instead of being skipped.
    """
    if m is None:
        if h not in H_TO_M:
            m = 1
            cp('yellow', f"Warning: unknown forecast horizon {h}, assuming seasonality m=1.", 2)
        else:
            m = H_TO_M[h]

    mase_values = []

    # strict=True raises if the generator and y_hat have different lengths
    for idx, (y_series, y_hat_series) in enumerate(
        zip(ts_generator(set_name), y_hat, strict=True)
    ):
        y_series = np.asarray(y_series.values, dtype=float)
        n = len(y_series) - h  # context length (C)

        if n <= 0 or m >= n:
            raise ValueError(
                f"Series {idx}: insufficient context (n={n}) for h={h}, m={m}."
            )

        y_true = y_series[-h:]
        y_pred = np.asarray(y_hat_series, dtype=float).flatten()

        if len(y_pred) != h:
            raise ValueError(
                f"Series {idx}: got {len(y_pred)} predictions, expected {h}."
            )

        numerator = np.mean(np.abs(y_true - y_pred))

        # in-sample seasonal-naive error over the context window only
        denominator = np.mean(np.abs(y_series[m:n] - y_series[:n - m]))

        if denominator == 0 or not np.isfinite(denominator):
            raise ValueError(f"Series {idx}: invalid scaling denominator ({denominator}).")

        mase_values.append(numerator / denominator)

    return np.mean(mase_values), mase_values


def maseOLD(set_name, ts_generator, y_hat, h):
    """
    MASE: Mean Absolute Scaled Error (not scaled)

    time_series_gen: generator of time series (each one a list of actual values)
    y_hat: List of predicted values (forecasted values for each time series)
    h: Forecast horizon of the series (e.g. 6 for yearly, 8 for quarterly, 18 for monthly)

    Returns the average MASE across all time series.

    n: number of time points in the series (excluding the forecast horizon)
    m: seasonality of the series (e.g. 12 for monthly, 4 for quarterly, 1 for yearly)

    WE DONT SKIP ANY SERIES IN TEST SET! ALL X_TEST AND Y_TEST MUST BE INCLUDED
    """
    h_to_m = {
        18: 12,  # monthly
        8: 4,    # quarterly
        48: 24,  # hourly
        6: 1,    # yearly
        13: 1,   # weekly
        14: 1    # daily
    }
    m = h_to_m.get(h, 1)
    mase_values = []

    for idx, (y_series, y_hat_series) in enumerate(zip(ts_generator(set_name), y_hat)):

        y_series = y_series.values

        n = len(y_series) - h

        if m > n or n <= 0:
            cp('yellow', f"Skipping series {idx} for MASE calculation due to insufficient length (n={n}) or m={m} greater than n.", 2)
            print(f"Series={y_series}, y_hat={y_hat_series}, h={h}, m={m}")
            #continue
            raise ValueError(f"Invalid number of time points (n={n}) or forecast horizon (h={h}) for series {idx}.")

        y_last_h = np.array(y_series[-h:])
        y_hat_idx = np.array(y_hat_series).flatten()

        nominator = np.mean(np.abs(y_last_h - y_hat_idx))

        denominator = np.mean([abs(y_series[t] - y_series[t - m]) for t in range(m, n)])

        this_mase = nominator / denominator
        mase_values.append(this_mase)

    return np.mean(mase_values), mase_values

def owa(model_smape, model_mase, naive_smape, naive_mase):
    """
    OWA: Overall Weighted Average
    model_smape: sMAPE of the model
    model_mase: MASE of the model
    naive_smape: sMAPE of the naive method
    naive_mase: MASE of the naive method
    
    Returns the OWA value.
    """
    relative_smape = model_smape / naive_smape
    relative_mase = model_mase / naive_mase

    owa = (relative_smape + relative_mase) / 2

    return owa



# NAIVE 1 is the method used in the paper
def get_naive1_predictions(X_test, h):
    """
    Naive1 predictions: repeat the last value of each series as the prediction for the next h steps.
    """
    forecasts = []
    for x_test_row in X_test:
        if len(x_test_row) < 1:
            raise ValueError("Each time series must have at least one data point.")
        forecasts.append(np.repeat(x_test_row[-1], h))
    return forecasts


def get_naive2_predictions(X_test, h):
    """
    Naive2 predictions: repeat the mean of the last h values of each series as the prediction for the next h steps.
    """
    forecasts = []
    for x_test_row in X_test:
        if len(x_test_row) < h:
            raise ValueError("Each time series must have at least h data points.")
        mean_last_h = np.mean(x_test_row[-h:])
        forecasts.append(np.repeat(mean_last_h, h))
    return forecasts

def get_naiveS_predictions(X_test, h):
    forecasts = []

    for x_test_row in X_test:
        if len(x_test_row) < h:
            raise ValueError("Each time series must be at least as long as the forecast horizon.")
        last_season = x_test_row[-h:]
        forecasts.append(last_season)  # Repeat last season as prediction

    return forecasts


def get_naive_errors(fh):
    # returns smape, mase
    if fh == 6:

        #return 16.342, 3.974, [] # my calculated
        return 16.342, 3.974, [] # by paper

    elif fh == 8:

        #return 11.610, 1.477, [] # my calculated
        return 11.012, 1.371, [] # by paper
    
    elif fh == 18:

        #return 14.384, 1.209, [] # my calculated
        return 14.427, 1.063, [] # by paper
    else:
        raise ValueError(f"[get_naive_errors] Unsupported forecasting horizon: {fh}")
    

if __name__ == "__main__":
    
    get_naive_errors(6)
    get_naive_errors(8)
    get_naive_errors(18)



