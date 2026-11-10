import os
import pandas as pd

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, get_abs_path, get_tmp_path
from data.src.config import get_datasets_metadata


def convert_tsf_to_parquet(metadata_name, tsf_file_path, parquet_dirpath, max_data_points=1000000):
    """
    Convert a .tsf file to multiple .parquet files inside a directory.
    Ensures total data points per file do not exceed max_data_points, 
    without splitting individual series. Includes frequency and prediction 
    length as the first two columns.

    Args:
        metadata_name (str): Key for the dataset in DATASETS_METADATA.
        tsf_file_path (str): Path to the input .tsf file.
        parquet_dirpath (str): Directory path to save output .parquet files.
        max_data_points (int): Maximum total data points allowed per file.
    """
    # Fetch frequency and prediction length from dataset metadata
    datasets_meta = get_datasets_metadata()
    if metadata_name not in datasets_meta:
        raise ValueError(f"Dataset metadata for '{metadata_name}' not found.")
    
    frequency = datasets_meta[metadata_name]["frequency"]
    prediction_length = datasets_meta[metadata_name]["prediction_length"]

    # Create the output directory if it doesn't exist
    os.makedirs(parquet_dirpath, exist_ok=True)

    cp("cyan", f"Converting {tsf_file_path} to Parquet files in {parquet_dirpath} with max {max_data_points} data points per file.")
    
    current_batch = []
    current_file_data_points = 0
    file_index = 1
    
    with open(tsf_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Ignore empty lines and lines starting with # or @
            if not line or line.startswith('#') or line.startswith('@'):
                continue
            
            # Split into 3 parts: increment number, ID, and the comma-separated values string
            parts = line.split(':', 2)
            if len(parts) == 3:
                ts_increment = parts[0]
                ts_id = parts[1]
                
                # Parse values, handling potential missing values indicated by '?'
                values = [
                    float(val) if val != '?' else None 
                    for val in parts[2].split(',')
                ]
                
                num_points = len(values)
                
                # If adding this series exceeds the limit and the batch isn't empty, 
                # save the current batch before starting a new file
                if current_file_data_points + num_points > max_data_points and current_batch:
                    df = pd.DataFrame(current_batch)
                    file_path = os.path.join(parquet_dirpath, f"{file_index}.parquet")
                    df.to_parquet(file_path, index=False)
                    
                    # Reset for the next file
                    file_index += 1
                    current_batch = []
                    current_file_data_points = 0
                
                # Add series to the current batch with frequency and prediction_length as first columns
                current_batch.append({
                    'frequency': frequency,
                    'prediction_length': prediction_length,
                    'values': values
                })
                current_file_data_points += num_points
                
    # Save any remaining series in the final batch
    if current_batch:
        df = pd.DataFrame(current_batch)
        file_path = os.path.join(parquet_dirpath, f"{file_index}.parquet")
        df.to_parquet(file_path, index=False)


if __name__ == "__main__":
    convert_tsf_to_parquet(
        metadata_name="Australian",
        tsf_file_path=get_abs_path("../original/australian_electricity_demand_dataset.tsf"),
        parquet_dirpath=get_abs_path("../parquet/Australian"),
        max_data_points=50000000
    )