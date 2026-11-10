import os
import sys
import torch
from datetime import datetime
import random
import numpy as np
import shutil

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.train.train import train
from src.test.tester import test
from src.utils import cp, get_tmp_path
from src.train.resuming import load_best_models
from src.train.best_models_tracker import read_all_best_models, read_all_latest_models


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config_id(CONFIG):

    weights = CONFIG['balancing']['weights']
    weights_str = '_'.join(map(str, weights)) if weights is not None else 'None'
    balancing = f"{CONFIG['balancing']['method']}_{weights_str}_{CONFIG['balancing']['noise_pct']}"
    cws = 'None'
    limit_obs = CONFIG.get('max_obs', 'None')

    config_id = f"model\
        _M_{CONFIG['model']['id']}\
        _OPT_{CONFIG['optimizer']}\
        _CRIT_{CONFIG['criterion']}\
        _LR_{CONFIG['init_lr']}\
        _TSET_{'_'.join(CONFIG['train_datasets'])}\
        _LIMOBS_{limit_obs}\
        _BAL_{balancing}\
        _VALSETS_{'_'.join(CONFIG['validation_sets'])}\
        _PAD_None\
        _WS_{CONFIG['window_size']}\
        _CUSTOM_WS_{cws}\
        _FH_{CONFIG['fh']}\
        _ENSCNT_{CONFIG['models_count']}\
        _BS_{CONFIG['batch_size']}\
        _SEED_{CONFIG['seed']}"

    config_id = config_id.replace(" ", "").replace("\n", "")
    return config_id

def display_config(CONFIG):
    for key in sorted(CONFIG.keys()):
        cp('gray', f"CONFIG | {key}: {CONFIG[key]}")

def run(CONFIG):

    # required paramaters:
    assert 'train_datasets' in CONFIG, "train_datasets is required in CONFIG"
    assert 'validation_sets' in CONFIG, "validation_sets is required in CONFIG"
    assert 'model' in CONFIG, "model is required in CONFIG"
    assert 'optimizer' in CONFIG, "optimizer is required in CONFIG"
    assert 'criterion' in CONFIG, "criterion is required in CONFIG"
    assert 'init_lr' in CONFIG, "init_lr is required in CONFIG"
    assert 'batch_size' in CONFIG, "batch_size is required in CONFIG"
    assert 'models_count' in CONFIG, "models_count is required in CONFIG"
    assert 'test_datasets' in CONFIG, "test_datasets is required in CONFIG"

    if CONFIG['validation_sets'] == "train":
        CONFIG['validation_sets'] = CONFIG['train_datasets']
        cp('yellow', "Using training datasets as validation datasets, since validation_sets is set to 'train'...")

    # set default parameters if not defined:
    CONFIG['initial_epoch'] = CONFIG.get('initial_epoch', 0)
    CONFIG['seed'] = CONFIG.get('seed', 42)
    CONFIG['max_epochs'] = CONFIG.get('max_epochs', 1)
    CONFIG['train_batch_n_series'] = CONFIG.get('train_batch_n_series', 50000)
    CONFIG['verbose'] = CONFIG.get('verbose', 5)
    CONFIG['out_sample_test'] = CONFIG.get('out_sample_test', False)
    CONFIG['save_intermediate_models'] = CONFIG.get('save_intermediate_models', True)
    CONFIG['save_results'] = CONFIG.get('save_results', True)
    CONFIG['max_obs'] = CONFIG.get('max_obs', None) # if not None, stop training after this many observations (time series) have been trained on (only counts new obs, so if resuming and training on the same data again, those obs will not be counted again)
    CONFIG['id'] = CONFIG.get('id', 'id-not-set')
    CONFIG['fresh'] = CONFIG.get('fresh', False) # if True, delete previous models and results for this config id (if exists)
    CONFIG['balancing'] = CONFIG.get('balancing', {
        'method': None, # None | 'undersampling' | 'oversampling' | 'smote' | 'temporal_augmentation'
        'weights': None, # None (=equal) | [a,b,c] (=biased, len(weights) should be equal to number of classes)
        'noise_pct': 0 # percentage of noise to add in oversampling (only if method is oversampling)
    })

    if CONFIG['max_epochs'] > 1 and CONFIG['max_obs'] is not None:
        cp('red', "ERROR: if max obs != None, mach epochs must be 1 to ensure consistent stopping based on obs count. Please set max_epochs to 1.")
        raise ValueError("if max obs != None, mach epochs must be 1 to ensure consistent stopping based on obs count. Please set max_epochs to 1.")


    set_seed(CONFIG['seed'])

    CONFIG['window_size'] = CONFIG['model']['window_size']
    CONFIG['fh'] = CONFIG['model']['forecasting_horizon']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #device = "cpu" # TODO: only for lstm

    cp('cyan', f"Using device: {device}")
    CONFIG['device'] = device

    CONFIG['id'] = get_config_id(CONFIG)

    # if fresh, delete the model output path
    if CONFIG['fresh']:
        tmp_path = get_tmp_path(CONFIG['id'])
        if os.path.exists(tmp_path):
            cp('yellow', f"Fresh run requested. Deleting previous output for config id {CONFIG['id']} at {tmp_path}")
            shutil.rmtree(tmp_path)

    display_config(CONFIG)

    if len(CONFIG['train_datasets']) > 0:

        meta_df = train(CONFIG)
        total_time = (meta_df['time_loading_train_data'].sum() + meta_df['time_training'].sum())
        n_obs = int(meta_df['n_obs'].sum())

        best_models = read_all_best_models(CONFIG)
        #best_models = read_all_latest_models(CONFIG)
        test_models = load_best_models(best_models, CONFIG)

        # print best models:
        cp('cyan', f"Best models for testing:")
        for idx, model in enumerate(best_models):
            cp('cyan', f"\tEnsemble index: {idx}, Epoch: {model['epoch']}, Val loss: {model['metric']:.4f}")

    elif CONFIG['model']['id'].startswith('naive'):
        cp('yellow', "No training datasets specified, skipping training and directly testing the naive model...")
        total_time = 0
        n_obs = 0
        test_models = [CONFIG['model']]

    else:
        raise ValueError("train_datasets is empty but model is not naive. Please specify train_datasets or use a naive model.")

    
    cp('cyan', f"Testing with {len(test_models)} models.")

    results = test(CONFIG, test_models)

    results_to_save = []

    for result in results:
        cp('green', f"Dataset: {result['dataset']}, sMAPE: {result['smape']:.3f}, MASE: {result['mase']:.3f}, OWA: {result['owa']:.3f}")
        results_to_save.append({
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_obs': n_obs,
            'time': total_time,
            'config_id': CONFIG['id'],
            'dataset': result['dataset'],
            'smape': result['smape'],
            'mase': result['mase'],
        })

    # Write (save) the results
    results_csv = get_tmp_path(f"results.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    if not os.path.exists(results_csv):
        with open(results_csv, 'w') as f:
            f.write("datetime,n_obs,time,config_id,dataset,smape,mase")

    # append to the results file
    with open(results_csv, 'a') as f:
        for result in results_to_save:
            f.write(f"\n{result['datetime']},{result['n_obs']},{result['time']},{result['config_id']},{result['dataset']},{result['smape']},{result['mase']}")




if __name__ == "__main__":

    from src.models.architectures.naive import model_naive
    from src.models.architectures.simple_dense import model_simple_dense
    from src.models.architectures.ef_sidx_lstm import model_sidx_lstm
    

    CONFIGS = [
        {
            #'train_datasets': ['M4yqm Shuffled'],
            'train_datasets': [],
            #'model': model_sidx_lstm(24, 6, 128, 3, 3),
            'model': model_simple_dense(24, 6, 500, 3),
            #'model': model_naive(24, 6, naive_type='1'),
            'models_count': 10,
            'optimizer': "adam", # adam | sgd
            'criterion': "mae", # mse | smoothL1
            'init_lr': 0.001,
            'batch_size': 512,
            'max_epochs': 2,
            'test_datasets': ['M4 Yearly','M4 Quarterly','M4 Monthly', 'M3 Yearly','M3 Quarterly','M3 Monthly', 'Tourism Yearly','Tourism Quarterly','Tourism Monthly', 'Weather', 'M5', 'ETT Hourly', 'ETT 15Mins', 'Australian'],
        },
    ]

    for CONFIG in CONFIGS:
        CONFIG['validation_sets'] = CONFIG['train_datasets'] # use training datasets as validation datasets
    

    if '-q' in sys.argv:
        # very quick test

        cp('yellow', "RUNNING QUICK TEST...", 0, None)

        for CONFIG in CONFIGS:

            #CONFIG['train_batch_n_series'] = 30
            CONFIG['max_obs'] = 30
            CONFIG['max_epochs'] = 1
            CONFIG['models_count'] = 1
            CONFIG['validation_sets'] = []
            CONFIG['test_datasets'] = ["'M4 Yearly'"]
            CONFIG['fresh'] = True
            CONFIG['save_results'] = False
            CONFIG['save_intermediate_models'] = False
            CONFIG['train_batch_n_series'] = 5



    
    for CONFIG in CONFIGS:
        run(CONFIG)

    os.startfile(get_tmp_path(f"results.csv"))

    """
    example runs:

    python run.py
    python run.py -q
    """
