import torch
import os
import copy

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import ask_yes_no, cp, get_tmp_path
from src.train.metadata_manager import create_new_metadata_df, read_metadata_df
from src.models.criterion import get_criterion
from src.models.optimzer import get_optimizer

def check_metadata_for_resuming(CONFIG):
    
    meta_df = read_metadata_df(CONFIG['id'])

    # if metadata is not none and has at least one full row of data (if the row has value in time_training it means the row is complete)
    if meta_df is None or meta_df.empty or meta_df['time_training'].isnull().all():
        cp('cyan', f"Metadata file not found. Creating new metadata dataframe.", 2, CONFIG)
        meta_df = create_new_metadata_df(CONFIG)
    else:
        # show message that metdata exists and get last epoch and data file from the metadata and ask user if they want ti continue training from there
        # (use input and check y/n)
        if True or ask_yes_no("Existing metadata found. Do you want to resume training from the last saved epoch and data file?: "):

            last_row = meta_df.iloc[-1]
            CONFIG['initial_epoch'] = int(last_row['epoch']) + 1 # start from next epoch

            cp('yellow', f"Resuming training from epoch {CONFIG['initial_epoch']+1}", 2, CONFIG)

        else:
            cp('cyan', f"Starting training from scratch. Previous metadata will be overwritten.", 2, CONFIG)
            meta_df = create_new_metadata_df(CONFIG)

            # cleanup previous models
            training_path = get_tmp_path()
            models_path = f"{training_path}/{CONFIG['id']}/models"
            if os.path.exists(models_path):
                for file in os.listdir(models_path):
                    if file.startswith(CONFIG['model']['id']):
                        file_path = os.path.join(models_path, file)
                        os.remove(file_path)
                        cp('yellow', f"Removed previous model file: {file_path}", 2, CONFIG)


    cp('cyan', f"Training from epoch {CONFIG['initial_epoch']+1}", 2, CONFIG)

    return meta_df, CONFIG



def get_model_path(epoch, ens_idx, CONFIG):
    return f"{get_tmp_path()}/{CONFIG['id']}/models/{CONFIG['model']['id']}_E{epoch+1}_ENS{ens_idx+1}.pth"


def load_ensemble_safe(CONFIG, meta_df, device):
    # based on the metadata, find lastest model and load all models in the ensemble (if they exist) and return them in a list. If any model is missing, raise an error.
    
    # find latest metadata row with time_training not null
    if meta_df is None or meta_df.empty or meta_df['time_training'].isnull().all():
        cp('cyan', f"No trained model found in metadata. Starting training from scratch.", 2, CONFIG)
        last_epoch = 0
    else:    

        last_trained_row = meta_df[meta_df['time_training'].notnull()].iloc[-1]
        cp('cyan', f"Last trained model found in metadata: Epoch {int(last_trained_row['epoch'])+1}", 2, CONFIG)

        last_epoch = int(last_trained_row['epoch'])

    models = []

    for ens_idx in range(CONFIG['models_count']):
        m = load_model_safe(CONFIG, last_epoch, ens_idx, device)
        if m[0] is not None:
            models.append({
                'model': m[0],
                'optimizer': m[1],
                'criterion': m[2],
                'model_path': m[3]
            })
            cp('green', f"Model for ENS_IDX {ens_idx+1} loaded successfully.", 2, CONFIG)
        else:
            cp('yellow', f"Model for ENS_IDX {ens_idx+1} not found. This model will be trained from scratch.", 2, CONFIG)

    return models










def load_model_safe(CONFIG, epoch, ens_idx, device):
    #cp('gray', f"Attempting to load model for Epoch {epoch+1}, Data File {file_idx+1}, ENS_IDX {ens_idx+1}...", 2, CONFIG)
    prev_model_path = get_model_path(epoch, ens_idx, CONFIG)
    #cp('gray', f"\tChecking for model at {prev_model_path}...", 2, CONFIG)

    # Instantiate model
    model = copy.deepcopy(CONFIG['model']['model'])
    #model = CONFIG['model']['model'] # this creates a reference to the same model, which causes issues when loading multiple models in the ensemble, as they will overwrite each other. We need to create a copy of the model for each ensemble member.
    model.to(device)  # move model to device **before** creating optimizer

    optimizer = get_optimizer(CONFIG, model)

    if os.path.exists(prev_model_path):
        cp('gray', f"\tLoading existing model: {prev_model_path}", 2, CONFIG)
        ckpt = torch.load(prev_model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    elif epoch != 0:
        # raise error if model missing
        raise FileNotFoundError(f"Could not resume training for model at E{epoch+1} - model not found at {prev_model_path}.")
    
    criterion = get_criterion(CONFIG)

    return model, optimizer, criterion, prev_model_path




# i think this function is not used anymore:
def load_best_models(best_models, CONFIG):

    test_list = []
    for ens_idx, bm in enumerate(best_models):

        model, optimizer, criterion, prev_model_path = load_model_safe(CONFIG, bm['epoch'], ens_idx, CONFIG['device'])

        test_list.append({
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion
        })
    return test_list