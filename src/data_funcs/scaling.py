import numpy as np

import os
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) # append root of IntermitNet
from src.utils import cp


## =======================================================
## For Testing (scale X window, inverse scale on y window)
## =======================================================
def minmax_inverse_scale(arr, min_val, max_val):
    arr = np.array(arr)
    return arr * (max_val - min_val) + min_val

def minmax_scale_X_set(X_set):

    X_set_scaled = []
    minmaxes = []

    for arr in X_set:

        min_val = min(arr)
        max_val = max(arr)
            
            
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"[minmax_scale_X_set] NaN or Inf values found in array: {arr}")

        if min_val == max_val:
            # return zeros
            scaled_arr = [0.0] * len(arr)
        else:
            scaled_arr = [(val - min_val) / (max_val - min_val) for val in arr]
        
        #scaled_arr = [round(val, 7) for val in scaled_arr]
        X_set_scaled.append(scaled_arr)
        minmaxes.append((min_val, max_val))
    
    return X_set_scaled, minmaxes


## =======================================================
## For Training (scale X and y windows, do not need min maxes
## =======================================================
def scale_X_y_windows(X_window, y_window):
    """
    Scales the input window (X) and forecasting horizon (y) using min-max scaling based on the input window.
    
    Args:
        X_window (list): The input window values.
        y_window (list): The forecasting horizon values.
    """
    
    min_val = min(X_window)
    max_val = max(X_window)

    if np.any(np.isnan(X_window)) or np.any(np.isnan(y_window)):
        raise ValueError(f"Error scaling because of NaN values. X_window: {X_window}, y_window: {y_window}")
    if np.any(np.isinf(X_window)) or np.any(np.isinf(y_window)):
        raise ValueError(f"Error scaling because of Inf values. X_window: {X_window}, y_window: {y_window}")
    if min_val == max_val:
        cp("yellow", f"Error scaling because of min_val == max_val. X_window: {X_window}, y_window: {y_window}. Skipping...", 3)
        return None, None

    scaled_X = [(val - min_val) / (max_val - min_val) for val in X_window]
    scaled_y = [(val - min_val) / (max_val - min_val) for val in y_window]

    return scaled_X, scaled_y



















def scale_X_y_row(row, window_size, forecasting_horizon):
    print(f"DEPRECATED FUNCTION: scale_X_y_row. Use scale_X_y_windows instead.")
    raise DeprecationWarning("scale_X_y_row is deprecated. Use scale_X_y_windows instead.")

    # Split the row into input window (X) and forecasting horizon (y)
    X = row[:window_size]

    # Scale the row using min-max scaling based on the input window
    min_val = min(X)
    max_val = max(X)

    if np.any(np.isnan(row)):
        # return empty to declare error
        cp("yellow", f"Error scaling because of NaN values. Row: {row}", 1)
        return None, None, None, (None, None)
    if np.any(np.isinf(row)):
        # return empty to declare error
        cp("yellow", f"Error scaling because of Inf values. Row: {row}", 1)
        return None, None, None, (None, None)
    if min_val == max_val:
        # return empty to declare error
        cp("yellow", f"Error scaling because of min_val == max_val. Row: {row}", 3)
        return None, None, None, (None, None)

    scaled_row = [(val - min_val) / (max_val - min_val) for val in row]

    X = scaled_row[:window_size]
    y = scaled_row[window_size:window_size + forecasting_horizon]

    return scaled_row, X, y, (min_val, max_val)


