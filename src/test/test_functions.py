import numpy as np
import os
import torch

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data_funcs.scaling import minmax_inverse_scale, minmax_scale_X_set
from src.utils import cp


def predict_ensemble(models_ensemble, X_test, X_test_exogs, frequency, CONFIG):

    X_test_scaled, minmaxes = minmax_scale_X_set(X_test)
    if X_test_scaled is None:
        cp('yellow', f"Scaling failed. X_test sample: {X_test[:2]}", 2)
        raise ValueError("Scaling failed")

    X_test_scaled = torch.as_tensor(X_test_scaled, dtype=torch.float32, device=CONFIG['device'])
    X_test_exogs = torch.as_tensor(X_test_exogs, dtype=torch.float32, device=CONFIG['device'])

    cp('gray', f"X_test_scaled shape: {X_test_scaled.shape}, X_test_exogs shape: {len(X_test_exogs)}", 2, CONFIG)

    return np.median(np.array([predict_different_horizon(m, X_test_scaled, X_test_exogs, minmaxes, frequency, CONFIG['window_size'], CONFIG['fh'], CONFIG) for m in models_ensemble], dtype=np.float64), axis=0)


def predict_different_horizon(model_obj, X_test_scaled, X_test_exogs, minmaxes, series_fh, window_size, forecasting_horizon, CONFIG):

    model = model_obj['model'].to(CONFIG['device'])
    model.eval()

    series_fh_counter = series_fh
    all_predictions = []

    with torch.no_grad():
        while series_fh_counter > 0:

            predictions = model(X_test_scaled, X_test_exogs) if CONFIG['model'].get('n_exogs', 0) > 0 else model(X_test_scaled)

            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(1)

            if series_fh_counter < forecasting_horizon:
                # cut off excess predictions
                predictions = predictions[:, :series_fh_counter]

            step = predictions.size(1)
            all_predictions.append(predictions)

            # ---- 2D rolling window ----
            if window_size > step:
                past = X_test_scaled[:, step:]
            else:
                past = X_test_scaled[:, :0]

            X_test_scaled = torch.cat([past, predictions], dim=1)
            series_fh_counter -= step

    all_predictions = torch.cat(all_predictions, dim=1)[:, :series_fh]
    all_predictions = all_predictions.cpu().numpy()

    final_predictions = []
    for i in range(len(all_predictions)):
        pred = minmax_inverse_scale(np.array(all_predictions[i], dtype=np.float64), *minmaxes[i])

        # also round to 2 decimals:
        pred = np.round(pred, 2)

        if not np.all(np.isfinite(pred)):
            raise ValueError(f"Non-finite values in series {i}")
        final_predictions.append(pred)

    return final_predictions










