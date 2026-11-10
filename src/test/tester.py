import os
import pandas as pd

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, get_tmp_path
from src.data_funcs.get_train_test_val import get_test_set
from src.test.error_functions import smape, mase
from src.test.test_functions import predict_ensemble
from data.src.read_datasets import stream_dataset_series
from data.src.config import get_datasets_metadata

def test(CONFIG, models):

    cp('cyan', f"Testing with {len(models)} models.")
    
    results = []

    datasets_metadata = get_datasets_metadata()

    for set_name in CONFIG['test_datasets']:

        cp('cyan', f"\n>>> Inferring for {set_name}")

        
        this_dataset_meta = datasets_metadata[set_name]
        ds_fh = this_dataset_meta['prediction_length']

        ## ================
        ## INFER
        ## ================

        out_path = get_tmp_path(f"{CONFIG['id']}/predictions_{set_name}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # if file exists already, do not test again
        if not os.path.exists(out_path):

            X_test, X_test_exogs, y_test = get_test_set(set_name, CONFIG['window_size']) # not scaled, just lists and lists of lists
            

            cp('gray', f"\tTest set size: {len(y_test)} series")
            cp('gray', f"\tForecasting horizon: {ds_fh}")

            cp('gray', f"\tInferring with ensemble of {len(models)} models...")

            cp('gray', f"\tX_test len: {len(X_test)}, y_test len: {len(y_test)}")
            predictions = predict_ensemble(models, X_test, X_test_exogs, ds_fh, CONFIG)

            
            cp('gray', f"\tSaving predictions to {out_path}")
            pd.DataFrame(predictions).to_csv(out_path, index=False, header=False)

        else:

            cp('yellow', f"\tPredictions file already exists. Skipping inferring for {set_name}.")
            # read from file
            """
            csv is like:
            6750.8,6551.775,6487.325,6565.23,6581.585,6518.88,6585.975,6683.79,6881.905000000001,6793.34,6826.505,6966.89,7141.775,7164.42,7227.27,7346.545,7514.155000000001,7528.1849999999995
            2161.0699999999997,2013.6999999999998,2056.415,2183.925,2101.34,1977.21,2064.795,2180.8199999999997,2128.675,2051.6949999999997,2124.425,2246.6800000000003,2197.5699999999997,2125.855,2238.675,2272.205,2276.0950000000003,2210.435
            """
            predictions = pd.read_csv(out_path, header=None).values

        cp('gray', f"\n>>> Predictions shape: {predictions.shape}")




        ## ================
        ## EVALUATE
        ## ================

        out_path = get_tmp_path(f"{CONFIG['id']}/metrics_{set_name}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cp('gray', f"\tEvaluating predictions for {set_name}...")

        if not os.path.exists(out_path):

            X_test, X_test_exogs, y_test = get_test_set(set_name, CONFIG['window_size']) # not scaled, just lists and lists of lists

            if len(predictions) != len(y_test):
                raise ValueError(f"Number of predictions {len(predictions)} does not match number of test series {len(y_test)}")

            smape_model, all_smapes = smape(y_test, predictions)
            mase_model, all_mases = mase(set_name, stream_dataset_series, predictions, ds_fh)

            # save all smapes to csv: smape,mase | 
            pd.DataFrame({
                'smape': all_smapes,
                'mase': all_mases
            }).to_csv(out_path, index=False)

            cp('gray', f"\tEnsemble sMAPE: {smape_model:.3f} | MASE: {mase_model:.3f}")

        else:

            cp('yellow', f"\tMetrics file already exists. Skipping evaluation for {set_name}.")
            # read from file
            df = pd.read_csv(out_path)
            smape_model = df['smape'].mean()
            mase_model = df['mase'].mean()



        out_path_summary = get_tmp_path(f"{CONFIG['id']}/summary_metrics_{set_name}.csv")
        df_summary = pd.DataFrame({
            'dataset': [set_name],
            'smape': [smape_model],
            'mase': [mase_model],
        })
        df_summary.to_csv(out_path_summary, index=False)


        results.append({
            'dataset': set_name,
            'smape': smape_model,
            'mase': mase_model,
            'owa': 0
        })

    return results


