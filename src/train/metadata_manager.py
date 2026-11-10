import os
import pandas as pd
import matplotlib.pyplot as plt

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.train.best_models_tracker import new_best_models_tracker
from src.utils import cp, get_tmp_path

"""
Metadata are stored in /tmp/{config_id}/metadata.csv

They are read/written using pandas DataFrame

The csv has a header and delimiter is ','.

Each row corresponds to one training step.

The fields are:
"""
COLUMNS = [

    # step tracker
    'epoch',                        # int, current epoch number
    'batch',                        # int, current batch number

    # time measurements
    'time_loading_train_data',      # float, time taken to load THIS training data file
    'time_training',                # float, time taken to train on THIS training data file (total of all ensemble models)

    # data measurements
    'n_obs',                        # int, number of observations (1 training window == 1 observation) used for training in all files (total so far)

    # error measurements
    'train_loss',                   # float, training loss achieved in this step
    'val_loss',                     # float, validation loss achieved after this step
    'lr',                           # float, learning rate AFTER this step
    'out_sample_owa',               # float, out sample OWA achieved AFTER this step (if calculated)
]

def read_metadata_df(config_id):

    metadata_path = get_tmp_path(f"{config_id}/metadata.csv")

    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path, header=0, delimiter=',')
        return df
    else:
        return None

def create_new_metadata_df(CONFIG):

    df = pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        df[col] = df[col].astype('float64')
    
    write_metadata_df(CONFIG['id'], df)

    # also create empty plots and models directory
    os.makedirs(get_tmp_path(f"{CONFIG['id']}/plots"), exist_ok=True)
    os.makedirs(get_tmp_path(f"{CONFIG['id']}/models"), exist_ok=True)

    # new best models tracker
    new_best_models_tracker(CONFIG)

    return df

def write_metadata_df(config_id, df):

    metadata_path = get_tmp_path(f"{config_id}/metadata.csv")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    df.to_csv(metadata_path, index=False, header=True, sep=',')

def add_metadata_row(df, epoch, batch):

    new_row = COLUMNS.copy()
    new_row[0] = float(epoch)
    new_row[1] = float(batch)
    for i in range(2, len(COLUMNS)):
        new_row[i] = -1

    df.loc[len(df)] = new_row

    return df

def update_last_metadata_row(df, field, value):

    if value is None:
        cp('yellow', f"[update_last_metadata_row] Value for field {field} is None. Skipping update.", 2)
        return df

    if field not in COLUMNS:
        raise ValueError(f"Field {field} not in metadata columns")

    df.at[df.index[-1], field] = float(value)


    return df






def plot_metadata(meta_df, config_id):
    # numeric x axis
    x = list(range(len(meta_df)))

    # labels like: E0 FI0
    labels = [
        f"E{int(e)} Batch{int(b)}" for e, b in zip(meta_df["epoch"], meta_df["batch"])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # ---- 1) Train vs Val loss ----
    ax = axes[0, 0]
    ax.plot(x, meta_df["val_loss"], label="Validation Loss")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch/Batch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    # ---- 2) Learning rate ----
    ax = axes[0, 1]
    ax.plot(x, meta_df["lr"])
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch/Batch")
    ax.set_ylabel("LR")
    ax.grid(True)

    # ---- 3) Out-of-sample OWA ----
    ax = axes[1, 0]
    ax.plot(x, meta_df["out_sample_owa"])
    ax.set_title("Out-of-sample OWA")
    ax.set_xlabel("Epoch/Batch")
    ax.set_ylabel("OWA")
    ax.grid(True)

    # ---- 4) Train vs Val (log scale helps with large gaps) ----
    ax = axes[1, 1]
    ax.plot(x, meta_df["train_loss"], label="Train Loss")
    ax.set_yscale("log")
    ax.set_title("Training Loss (Log Scale)")
    ax.set_xlabel("Epoch/Batch")
    ax.set_ylabel("Loss (log)")
    ax.grid(True)
    ax.legend()

    # reduce tick density across all plots
    step = max(1, len(labels) // 15)
    for ax in axes.ravel():
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right")

    fig.tight_layout()

    plot_path = get_tmp_path(f"{config_id}/plots/metadata_plot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
