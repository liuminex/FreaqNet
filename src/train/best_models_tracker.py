import os

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, get_tmp_path



def new_best_models_tracker(CONFIG):

    files_path = get_tmp_path(f"{CONFIG['id']}/best_models_meta/")

    # for every model in ensemble create a file named: {ens_idx}.txt and leave empty
    for ens_idx in range(CONFIG['models_count']):
        file_path = f"{files_path}/{ens_idx}.txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write("0;0;9999999999")

def update_best_model(CONFIG, ens_idx, epoch, batch, metric):

    cp('magenta', f"Updating best model for ensemble index {ens_idx} to epoch {epoch}, batch {batch}, metric {metric}", 2, CONFIG)

    file_path = get_tmp_path(f"{CONFIG['id']}/best_models_meta/{ens_idx}.txt")

    with open(file_path, 'w') as f:
        f.write(f"{epoch};{batch};{metric}")


def read_all_best_models(CONFIG):

    best_models = []
    
    files_path = get_tmp_path(f"{CONFIG['id']}/best_models_meta/")

    for ens_idx in range(CONFIG['models_count']):
        file_path = f"{files_path}/{ens_idx}.txt"
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            if line:
                epoch, batch, metric = line.split(';')
                best_models.append({
                    'epoch': int(epoch),
                    'batch': int(batch),
                    'metric': float(metric)
                })
            else:
                best_models.append({
                    'epoch': 0,
                    'batch': 0,
                    'metric': 9999999999
                })


    return best_models


def read_all_latest_models(CONFIG):

    latest_models = []
    
    files_path = get_tmp_path(f"{CONFIG['id']}/best_models_meta/")

    """
    find max epoch and max batch indices and return the same for all models
    """
    max_epoch = 0
    max_batch = 0

    for ens_idx in range(CONFIG['models_count']):
        file_path = f"{files_path}/{ens_idx}.txt"
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            if line:
                epoch, batch, metric = line.split(';')
                epoch = int(epoch)
                batch = int(batch)
                if epoch > max_epoch:
                    max_epoch = epoch
                if batch > max_batch:
                    max_batch = batch

    for ens_idx in range(CONFIG['models_count']):
        latest_models.append({
            'epoch': max_epoch,
            'batch': max_batch,
            'metric': 9999999999
        })

    return latest_models



