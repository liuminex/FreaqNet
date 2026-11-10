import os
import pandas as pd

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, get_abs_path
from data.src.config import get_datasets_metadata

DATASETS_METADATA = get_datasets_metadata()

def stream_dataset_series(dataset_name):
    """
    A RAM-efficient generator that yields time series rows one by one 
    from the sequential parquet files of a given dataset.
    
    Each yielded row contains:
    - frequency
    - prediction_length (prediction horizon)
    - values
    
    Args:
        dataset_name (str): Name of the dataset from DATASETS_METADATA (e.g., "M4 Yearly").
    """
    base_dir = get_abs_path("../parquet")
    if dataset_name not in DATASETS_METADATA:
        raise ValueError(f"Dataset '{dataset_name}' not found in metadata.")
        
    meta = DATASETS_METADATA[dataset_name]
    parquet_dir = os.path.join(base_dir, meta["parquet_dirpath"])
    
    if not os.path.exists(parquet_dir):
        cp("red", f"Error: Parquet directory '{parquet_dir}' does not exist for dataset '{dataset_name}'.")
        return
        
    # Find all parquet files and sort them numerically (1.parquet, 2.parquet, ...)
    files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    
    for file_name in files:
        file_path = os.path.join(parquet_dir, file_name)
        # Load one file at a time into memory to maintain high RAM efficiency
        df = pd.read_parquet(file_path)
        for row in df.itertuples(index=False):
            # Yields the row containing frequency, prediction_length, and values
            yield row


def print_dataset_statistics(dataset_name):
    """
    Uses the generator to extract frequency, prediction horizon, and values 
    for every yield, computing and printing aggregate dataset statistics.
    """
    total_series = 0
    min_len = float('inf')
    max_len = float('-inf')
    total_len = 0
    
    frequency = None
    prediction_length = None
    
    # Iterate through the generator efficiently without loading all files into RAM at once
    for row in stream_dataset_series(dataset_name):
        # Extract frequency, prediction horizon, and values for every yield
        if False: # disable for Shuffled datasets
            if frequency is not None and frequency != row.frequency:
                cp("yellow", f"Warning: Frequency mismatch detected in dataset '{dataset_name}'. Previous: {frequency}, Current: {row.frequency}")
            if prediction_length is not None and prediction_length != row.prediction_length:
                cp("yellow", f"Warning: Prediction length mismatch detected in dataset '{dataset_name}'. Previous: {prediction_length}, Current: {row.prediction_length}")
        frequency = row.frequency
        prediction_length = row.prediction_length
        values = row.values
        
        total_series += 1
        length = len(values)
        
        if length < min_len:
            min_len = length
        if length > max_len:
            max_len = length
        total_len += length
        
    if total_series == 0:
        cp("red", f"Dataset '{dataset_name}' has no data or files were not found.")
        return
        
    avg_len = total_len / total_series
    
    # Print results formatted clearly
    print(f"=== Statistics for: {dataset_name} ===")
    print(f"* Frequency         : {frequency}")
    print(f"* Prediction Length : {prediction_length}")
    print(f"* Total Num Series  : {total_series:,}")
    print(f"* Series Length Min : {min_len:,}")
    print(f"* Series Length Avg : {avg_len:,.2f}")
    print(f"* Series Length Max : {max_len:,}")
    print(f"* Total Data points : {total_len:,}")


if __name__ == "__main__":
    # Example usage: Print statistics for all datasets in DATASETS_METADATA
    for dataset_name in DATASETS_METADATA.keys():
        print_dataset_statistics(dataset_name)
        print()