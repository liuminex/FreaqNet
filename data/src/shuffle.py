import os
import pandas as pd
import random

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, get_abs_path
from data.src.config import get_datasets_metadata
from data.src.read_datasets import stream_dataset_series

def shuffle_datasets(dataset_names_list, output_dirname, n_files=3):
    """
    Use stream_dataset_series to read the datasets in dataset_names_list,
    add each series to a random file of the n_files in output_dirname,
    shuffle the series within each output file, and write to Parquet.
    """
    os.makedirs(output_dirname, exist_ok=True)
    
    # Initialize data buckets for each target file (1 to n_files)
    buckets = {i: [] for i in range(1, n_files + 1)}
    
    for dataset_name in dataset_names_list:
        # Stream series from the dataset generator
        for row in stream_dataset_series(dataset_name):
            # Assign the series to a random file bucket
            file_idx = random.randint(1, n_files)
            
            # Append updated attributes (frequency, prediction_length, values) to the bucket
            buckets[file_idx].append({
                'frequency': row.frequency,
                'prediction_length': row.prediction_length,
                'values': row.values
            })
            
    # Write each bucket's contents to its respective parquet file after shuffling
    for file_idx, data_list in buckets.items():
        if data_list:
            # Shuffle the order of rows inside this file bucket
            random.shuffle(data_list)
            
            df = pd.DataFrame(data_list)
            output_file_path = os.path.join(output_dirname, f"{file_idx}.parquet")
            df.to_parquet(output_file_path, index=False)
            cp("cyan", f"Saved {len(data_list):,} shuffled series to {output_file_path}")


if __name__ == "__main__":
    shuffle_datasets(
        dataset_names_list=["M4 Yearly", "M4 Quarterly", "M4 Monthly"],
        output_dirname=get_abs_path("../parquet/Shuffled_M4yqm"),
        n_files=1
    )