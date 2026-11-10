import numpy as np
import torch
import time
import os
from torch.utils.data import TensorDataset, DataLoader


os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, format_header, format_time, format_number
from src.train.metadata_manager import plot_metadata, update_last_metadata_row, write_metadata_df, add_metadata_row
from src.train.resuming import check_metadata_for_resuming, get_model_path, load_ensemble_safe
from src.train.best_models_tracker import read_all_best_models, update_best_model
from src.test.tester import test
from src.data_funcs.get_train_test_val import get_train_batch, get_val_set



def train(CONFIG):

    cp('cyan', "Training started. CONFIG: " + CONFIG['id'])
    
    # df for metadata
    meta_df, CONFIG = check_metadata_for_resuming(CONFIG)

    # if training datasets is empty, return
    if len(CONFIG['train_datasets']) == 0:
        cp('yellow', "No training datasets specified. Skipping training.")
        return

    device = CONFIG['device'] if 'device' in CONFIG else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cp('gray', f"Using device: {device}")

    # Load Validation Set if specified
    if CONFIG.get('validation_sets', None) is not None and len(CONFIG.get('validation_sets', [])) > 0:

        cp('gray', f"Using validation sets: {CONFIG['validation_sets']}")

        X_val, X_val_exogs, y_val = [], [], []
        for val_dataset_name in CONFIG['validation_sets']:
            xval, xvale, yval = get_val_set(val_dataset_name, CONFIG['window_size'], CONFIG['fh'])
            cp("gray", f"loaded validation set {val_dataset_name}")
            X_val.extend(xval)
            X_val_exogs.extend(xvale)
            y_val.extend(yval)

        # trim exogs (keep only the exogs the model needs for optimization)
        X_val_exogs = [exog[:CONFIG['model'].get('n_exogs', 0)] for exog in X_val_exogs]

        cp('magenta', f"(val set) Exogs example: {X_val_exogs[0] if len(X_val_exogs) > 0 else 'N/A'}")

        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_val_exogs = torch.tensor(X_val_exogs, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
    
        cp('gray', f"Validation set shapes:\n\t\tX_val:\t\t{X_val.shape}\n\t\tX_val_exogs:\t{X_val_exogs.shape}\n\t\ty_val:\t\t{y_val.shape}")
        
    else:
        cp('yellow', "No validation sets specified. Skipping validation.")
       
    # load models here (one for every ensemble)
    models = load_ensemble_safe(CONFIG, meta_df, device)

    # check
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            if next(models[i]['model'].parameters()).data_ptr() == next(models[j]['model'].parameters()).data_ptr():
                cp('red', f"Error: Models {i} and {j} are the same instance. This should not happen. Please check the model loading function.")
                raise ValueError(f"Models {i} and {j} are the same instance. This should not happen. Please check the model loading function.")
            
    best_models = read_all_best_models(CONFIG)

    stop_training = False

    cp('cyan', f"Starting training from epoch {CONFIG['initial_epoch']+1} to {CONFIG['max_epochs']} with max_obs={CONFIG.get('max_obs', 'None')}")

    for epoch in range(CONFIG['initial_epoch'], CONFIG['max_epochs']):

        cp('magenta', format_header(f"Epoch {epoch+1}/{CONFIG['max_epochs']}", "-"))

        batch_idx = 0

        for X_train, X_train_exogs, y_train in get_train_batch(CONFIG['train_datasets'], CONFIG['window_size'], CONFIG['fh'], CONFIG['train_batch_n_series']):

            t_start = time.time()

            cp('gray', f"(train set) Exogs example: {X_train_exogs[0] if len(X_train_exogs) > 0 else 'N/A'}")
            meta_df = add_metadata_row(meta_df, epoch, batch_idx) # add new row to metadata df, with epoch

            # move datasets to device
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            X_train_exogs = torch.tensor(X_train_exogs, dtype=torch.float32).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

            train_dataset = \
                TensorDataset(X_train, X_train_exogs, y_train) \
                if CONFIG['model'].get('n_exogs', 0) > 0 \
                else TensorDataset(X_train, y_train)

            train_loader = DataLoader(
                train_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                drop_last=False
            )
            meta_df = update_last_metadata_row(meta_df, 'time_loading_train_data', time.time() - t_start)

            cp('gray', f"\n\t\tX_train:\t{X_train.shape}\n\t\tX_train_exogs:\t{X_train_exogs.shape}\n\t\ty_train:\t{y_train.shape}")

            n_new_obs_current = len(X_train) if epoch == CONFIG['initial_epoch'] else 0 # only count new obs for the first epoch, after that we are resuming and we do not want to count the same obs again
          
            # if exceeds max_obs, set n_new_obs_current to the remaining obs until max_obs and then stop training after this epoch
            if CONFIG['max_obs'] is not None and int(meta_df['n_obs'].sum()) + n_new_obs_current > CONFIG['max_obs']:
                max_n = CONFIG['max_obs'] - int(meta_df['n_obs'].sum())
                cp('yellow', f"Reached max_obs limit of {CONFIG['max_obs']} with current obs {n_new_obs_current} and total obs {int(meta_df['n_obs'].sum())}. Limiting to {max_n} new obs for this epoch.")
                X_train = X_train[:max_n]
                X_train_exogs = X_train_exogs[:max_n]
                y_train = y_train[:max_n]
                n_new_obs_current = len(X_train)
                stop_training = True

            meta_df = update_last_metadata_row(meta_df, 'n_obs', n_new_obs_current)
            # training starts

            t_start = time.time()

            ensemble_train_losses = []
            ensemble_val_losses = []
            ensemble_lrs = []
            ensemble_out_sample_owas = []

            for ens_idx in range(CONFIG['models_count']):

                cp('green', f"E {epoch+1}/{CONFIG['max_epochs']} | ENS_IDX {ens_idx+1}/{CONFIG['models_count']} | Batch {batch_idx}", 0, CONFIG)

                if models[ens_idx]['model'] is None:
                    raise ValueError(f"Model for ensemble index {ens_idx} is None. This should not happen. Please check the model loading function.")
                
                #if CONFIG['verbose'] > 2: print(model) # print model summary

                if len(X_train) > 0:

                    cp('gray', f"\tTraining...", end="\r")

                    # --- TRAIN ---

                    models[ens_idx]['model'].train()

                    batch_losses = []

                    for batch in train_loader:

                        models[ens_idx]['optimizer'].zero_grad()

                        if CONFIG['model'].get('n_exogs', 0) > 0:
                            xb, exb, yb = batch
                            outputs = models[ens_idx]['model'](xb, exb)


                        else:
                            xb, yb = batch
                            outputs = models[ens_idx]['model'](xb)
                        
                        
                        loss = models[ens_idx]['criterion'](outputs, yb)
                        
                        loss.backward()
                        models[ens_idx]['optimizer'].step()

                        batch_losses.append(loss.item())


                    ensemble_train_losses.append(np.mean(batch_losses))

                    current_lr = models[ens_idx]['optimizer'].param_groups[0]['lr']
                    ensemble_lrs.append(current_lr)


                    # ---- VALIDATION ----

                    # ==========================================
                    # 2. BATCHED ITERATION FOR EVALUATION (Lines 140-161)
                    # ==========================================
                    # ---- FIX: BATCHED MEMORY-SAFE VALIDATION ----
                    avg_val_loss = -1
                    if CONFIG['validation_sets'] is not None and len(CONFIG['validation_sets']) > 0 and len(X_val) > 0:
                        models[ens_idx]['model'].eval()
                        
                        val_batch_losses = []
                        val_batch_size = CONFIG['batch_size']
                        
                        with torch.no_grad(): 
                            for i in range(0, X_val.size(0), val_batch_size):
                                # Slice mini-batches from CPU, then move just the tiny slice to GPU
                                xb_val = X_val[i : i + val_batch_size].to(device)
                                yb_val = y_val[i : i + val_batch_size].to(device)
                                
                                if CONFIG['model'].get('n_exogs', 0) > 0:
                                    exb_val = X_val_exogs[i : i + val_batch_size].to(device)
                                    val_outputs = models[ens_idx]['model'](xb_val, exb_val)
                                else:
                                    val_outputs = models[ens_idx]['model'](xb_val)
                                    
                                val_loss = models[ens_idx]['criterion'](val_outputs, yb_val)
                                val_batch_losses.append(val_loss.item())
                        
                        avg_val_loss = np.mean(val_batch_losses)
                        cp('gray', f"\tVal loss: {avg_val_loss:.4f}", 4, CONFIG)
                        ensemble_val_losses.append(avg_val_loss)



                    cp('gray', f"\tval loss: {avg_val_loss:.4f}, best val loss: {best_models[ens_idx]['metric']:.4f}", 4, CONFIG)
                    
                    if not (CONFIG.get('validation_sets', None) is not None and len(CONFIG.get('validation_sets', [])) > 0) or best_models[ens_idx]['metric'] == 0 or best_models[ens_idx]['metric'] > avg_val_loss: # if current val loss is better than best val loss for this model, update best model
                        update_best_model(CONFIG, ens_idx, epoch, batch_idx, avg_val_loss)
                        best_models[ens_idx]['metric'] = avg_val_loss


                else:
                    cp('gray', f"\tNo training data for this file.")


                


                if CONFIG['save_intermediate_models']:
                    intermediate_model_path = get_model_path(epoch, ens_idx, CONFIG)
                    cp('gray', f"\tSaving intermediate model:\t{intermediate_model_path}", 5, CONFIG)
                    torch.save({
                        'model': models[ens_idx]['model'].state_dict(),
                        'optimizer': models[ens_idx]['optimizer'].state_dict()
                    }, intermediate_model_path)

                # EOF ensemble model loop

            # After ensemble train end:

            

            if CONFIG['out_sample_test']:
                cp('gray', f"\tTesting out-of-sample OWA...")
                results = test(CONFIG, [{'model': model['model']} for model in models])

                owas = 0
                owas_count = 0
                
                for res in results:
                    if res['dataset'] in CONFIG['test_datasets']:
                        cp('gray', f"\t\tIn-sample OWA for {res['dataset']}: {res['owa']:.4f}", 4, CONFIG)
                        if res['dataset'] == "M4Y":
                            owas += res['owa']
                            owas_count += 1
                        if res['dataset'] == "M4Q":
                            owas += res['owa']
                            owas_count += 1
                        if res['dataset'] == "M4M":
                            owas += 2*res['owa']
                            owas_count += 2

                wowa = owas / owas_count if owas_count > 0 else -1
                
                ensemble_out_sample_owas.append(wowa)
                cp('gray', f"\tOut-of-sample OWA: {wowa:.4f}", 4, CONFIG)
                


            meta_df = update_last_metadata_row(meta_df, 'time_training', time.time() - t_start)
            meta_df = update_last_metadata_row(meta_df, 'train_loss', np.mean(ensemble_train_losses) if len(ensemble_train_losses) > 0 else None)
            meta_df = update_last_metadata_row(meta_df, 'val_loss', np.mean(ensemble_val_losses) if len(ensemble_val_losses) > 0 else None)
            meta_df = update_last_metadata_row(meta_df, 'lr', np.mean(ensemble_lrs) if len(ensemble_lrs) > 0 else None)
            meta_df = update_last_metadata_row(meta_df, 'out_sample_owa', np.mean(ensemble_out_sample_owas) if len(ensemble_out_sample_owas) > 0 else None)

            # save metadata df after each data file
            write_metadata_df(CONFIG['id'], meta_df)

            if CONFIG['verbose'] > 6: plot_metadata(meta_df, CONFIG['id'])


            batch_idx += 1
            if stop_training:
                cp('yellow', f"Reached max_obs limit of {CONFIG['max_obs']}. Stopping training after this epoch.")
                break
            # EOF batch loop


        



        

        if stop_training:
            cp('yellow', f"Reached max_obs limit of {CONFIG['max_obs']}. Stopping training after this epoch.")
            break

        # EOF epoch

        

    # get total time and number of obs from metadata (meta_df) and print them
    total_time_accurate = 0
    total_time_measured = (meta_df['time_loading_train_data'].sum() + meta_df['time_training'].sum())

    # time training:
    cp('green', f"total time data loading: {format_time(meta_df['time_loading_train_data'].sum())}", 0, CONFIG)
    cp('green', f"total time training: {format_time(meta_df['time_training'].sum())}", 0, CONFIG)

    cp('green', format_header("Training finished.", "+"), 0, CONFIG)
    cp('gray', f"Total time (session): {format_time(total_time_accurate)}", 0, CONFIG)
    cp('green', f"Total time (measured): {format_time(total_time_measured)}", 0, CONFIG)
    cp('green', f"Total observations used for training: {format_number(int(meta_df['n_obs'].sum()))}", 0, CONFIG)

    return meta_df

