import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # root of IntermitNet

from src.utils import cp
from data.src.read_datasets import stream_dataset_series
from src.data_funcs.extract_features import extract_all_features
from src.data_funcs.scaling import scale_X_y_windows

OUTLIER_THRESHOLD = 5.0


def _scale_and_validate(X_raw, y_raw):
    """Scale a window; return (X, y) or None if invalid (NaN/Inf/constant/outlier)."""
    scaled_X, scaled_y = scale_X_y_windows(X_raw, y_raw)
    if scaled_X is None or scaled_y is None:
        return None
    scaled_X = np.asarray(scaled_X, dtype=np.float64)
    scaled_y = np.asarray(scaled_y, dtype=np.float64)
    if not (np.all(np.isfinite(scaled_X)) and np.all(np.isfinite(scaled_y))):
        return None
    if np.max(np.abs(scaled_X)) > OUTLIER_THRESHOLD:
        return None
    return scaled_X, scaled_y


def _check_fh(model_fh):
    if model_fh is None or model_fh <= 0:
        raise ValueError(f"model_fh must be a positive integer, got {model_fh}.")


# =============
# TRAIN SET (in batches of series)
# =============
def get_train_batch(dataset_names_list, model_window_size, model_fh, n_series):
    """
    Generator. Yields (X_train, X_train_exogs, y_train) built from `n_series`
    series at a time. The last batch may contain fewer series.
    """
    _check_fh(model_fh)
    if n_series is None or n_series <= 0:
        raise ValueError(f"n_series must be a positive integer, got {n_series}.")

    X_train, X_train_exogs, y_train = [], [], []
    series_in_batch = 0

    progress = 0

    for dataset_name in dataset_names_list:

        for row in stream_dataset_series(dataset_name):
            ds_fh = row.prediction_length
            if ds_fh is None or ds_fh <= 0:
                raise ValueError(f"Invalid forecasting horizon {ds_fh} for this series.")

            # remove test + val region to avoid leakage
            values = np.asarray(row.values, dtype=np.float64)[:-(ds_fh + model_fh)]

            progress += 1
            if progress % 1000 == 0:
                cp("gray", f"Loaded {progress} series (dataset={dataset_name})")

            if len(values) < model_window_size + model_fh:
                #cp("yellow", "Not enough values for training set. Skipping...", 0)
                pass
            else:
                for start in range(len(values) - model_window_size - model_fh + 1):
                    origin = start + model_window_size
                    X_raw = values[start:origin]
                    y_raw = values[origin:origin + model_fh]

                    scaled = _scale_and_validate(X_raw, y_raw)
                    if scaled is None:
                        cp("yellow", "Skipping training window (NaN/Inf, constant, or outlier).", 3)
                        continue

                    # strictly causal: only data up to the forecast origin
                    exogs = extract_all_features(values[:origin], row.frequency)

                    X_train.append(scaled[0])
                    X_train_exogs.append(exogs)
                    y_train.append(scaled[1])

            series_in_batch += 1
            if series_in_batch == n_series:
                if X_train:  # skip batches where every series was filtered out
                    yield X_train, X_train_exogs, y_train
                X_train, X_train_exogs, y_train = [], [], []
                series_in_batch = 0

        if X_train:  # remaining partial batch
            yield X_train, X_train_exogs, y_train


# =============
# VAL SET
# =============
def get_val_set(dataset_name, model_window_size, model_fh):
    """Returns SCALED x-y pairs for validation."""
    _check_fh(model_fh)
    X_val, X_val_exogs, y_val = [], [], []

    for row in stream_dataset_series(dataset_name):
        ds_fh = row.prediction_length
        if ds_fh is None or ds_fh <= 0:
            raise ValueError(f"Invalid forecasting horizon {ds_fh} for this series.")

        values = np.asarray(row.values, dtype=np.float64)[:-ds_fh]  # remove test set

        if len(values) < model_window_size + model_fh:
            #cp("yellow", "Not enough values for validation set. Skipping...", 0)
            continue

        y_raw = values[-model_fh:]
        history = values[:-model_fh]          # everything before the val targets
        X_raw = history[-model_window_size:]

        scaled = _scale_and_validate(X_raw, y_raw)
        if scaled is None:
            cp("yellow", "Skipping validation series (NaN/Inf, constant, or outlier).", 3)
            continue

        X_val.append(scaled[0])
        X_val_exogs.append(extract_all_features(history, row.frequency))
        y_val.append(scaled[1])

    return X_val, X_val_exogs, y_val


# =============
# TEST SET
# =============
def get_test_set(dataset_name, model_window_size):
    """Returns UNSCALED x-y pairs for evaluating. y has length ds_fh (series-specific)."""
    X_test, X_test_exogs, y_test = [], [], []

    for row in stream_dataset_series(dataset_name):
        ds_fh = row.prediction_length
        if ds_fh is None or ds_fh <= 0:
            raise ValueError(f"Invalid forecasting horizon {ds_fh} for this series.")

        all_values = np.asarray(row.values, dtype=np.float64)
        if len(all_values) <= ds_fh:
            raise ValueError("Not enough values to extract a test set and keep any history.")

        y_test_series = all_values[-ds_fh:]
        history = all_values[:-ds_fh]

        # exogs from the REAL history only (padding must not distort the features)
        exogs = extract_all_features(history, row.frequency) # fixed size for any window size

        # left-pad short histories with the first observed value (not 0.0)
        n_missing = model_window_size - len(history)
        X_series = history
        if n_missing > 0:
            X_series = np.concatenate([np.full(n_missing, history[0]), history])
        X_series = X_series[-model_window_size:]

        X_test.append(X_series)
        X_test_exogs.append(exogs)
        y_test.append(y_test_series)

    return X_test, X_test_exogs, y_test