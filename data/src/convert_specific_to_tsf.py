
"""
This file contains functions to convert different formats of original datasets to a standard format.
The standard format is a .tsf file like this:

{ts increment number}:{ts ID}:{list of vaslues separated by commas}
T1:1979-01-01 12-00-00:5172.1,5133.5,5186.9,5084.6,5182,5414.3,5576.2,5752.9,5955.2,6087.8,6238.9,6317.2,6262.7,6361,6427.4,6654.9,6835.4,6925.5,7073.5,7144,7230.6,7349.6,7339.2,7250.8,7294.6,7393.9,7560.9,7651.4,7587.3,7530.5,7261.1,7290.2,7392.6,7527.6,7594.8,7720.7,7823.2
T2:1979-01-01 12-00-00:2070,2104,2394,1651,1492,1348,1198,1192,1105,1008,994,1420,1582,1286,1133,1322,1157,1244,1843,1794,1426,1170,1021,997,1062,1013,944,861,1002,1302,1603,1598,1366,1288,1148,1139,1062
"""

import os
import pandas as pd

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import cp, get_abs_path, get_tmp_path

def convert_m5_to_tsf():
    """
    Converts the M5 dataset from its original CSV format to the standard .tsf format,
    merging sales_train_evaluation.csv and sales_test_evaluation.csv into a single continuous time series.
    """
    train_file = get_abs_path("../original/sales_train_evaluation.csv")
    test_file = get_abs_path("../original/sales_test_evaluation.csv")
    output_file = get_tmp_path("m5_dataset.tsf")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the train dataset
    df_train = pd.read_csv(train_file)
    
    # Read and merge the test dataset (evaluation period) if available
    if os.path.exists(test_file):
        df_test = pd.read_csv(test_file)
        
        # If test file uses F1-F28 columns, rename them to chronological d_ columns (d_1942 to d_1969)
        f_columns = [col for col in df_test.columns if col.startswith('F')]
        if f_columns:
            rename_mapping = {f"F{i}": f"d_{1941 + i}" for i in range(1, len(f_columns) + 1)}
            df_test = df_test.rename(columns=rename_mapping)
            
        # Merge on shared metadata columns
        merge_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        # Drop 'id' from df_test if it exists to avoid collision duplicates
        if 'id' in df_test.columns:
            df_test = df_test.drop(columns=['id'])
            
        df = pd.merge(df_train, df_test, on=merge_cols, how='inner')
    else:
        df = df_train

    # Identify all daily columns starting with 'd_' and sort them chronologically
    day_columns = [col for col in df.columns if col.startswith('d_')]
    day_columns = sorted(day_columns, key=lambda x: int(x.split('_')[1]))

    missing_values = "no"
    frequency = "daily"
    univariate = "yes"

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write .tsf metadata headers
        f.write("@relation m5_dataset\n")
        f.write(f"@frequency {frequency}\n")
        f.write(f"@missing_values {missing_values}\n")
        f.write(f"@univariate {univariate}\n")
        f.write("@data\n")

        # Write each time series row
        for idx, row in df.iterrows():
            ts_increment = f"T{idx + 1}"
            ts_id = f"{row['item_id']}_{row['store_id']}_evaluation"
            
            # Extract all values as a list/numpy array
            vals = row[day_columns].values
            
            # Optional: Trim leading zeros/inactive days to match Monash archive variable lengths
            # (Find the index of the first non-zero value, or keep all if you prefer)
            non_zero_indices = (vals != 0).nonzero()[0]
            if len(non_zero_indices) > 0:
                start_idx = non_zero_indices[0]
                vals = vals[start_idx:]
            
            # Ensure minimum length condition if needed, or just join the sliced values
            values = ",".join(vals.astype(str))
            
            f.write(f"{ts_increment}:{ts_id}:{values}\n")
    


def convert_ETTh_to_tsf():
    """
    Converts the ETT dataset from its original CSV format to the standard .tsf format.
    """
    train1_file = get_abs_path("../original/ETTh1.csv")
    train2_file = get_abs_path("../original/ETTh2.csv")
    """
    Each file is like:

    date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
    2016-07-01 00:00:00,5.827000141143799,2.009000062942505,1.5989999771118164,0.4620000123977661,4.203000068664552,1.3400000333786009,30.5310001373291
    2016-07-01 01:00:00,5.692999839782715,2.075999975204468,1.4919999837875366,0.4259999990463257,4.142000198364259,1.371000051498413,27.78700065612793
    2016-07-01 02:00:00,5.1570000648498535,1.741000056266785,1.2790000438690186,0.35499998927116394,3.776999950408936,1.218000054359436,27.78700065612793

    so from every file i extract 7 (HUFL,HULL,MUFL,MULL,LUFL,LULL,OT) time series, so in total 14 time series.
    """
    output_file = get_tmp_path("ETTh_dataset.tsf")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = {
        "ETTh1": train1_file,
        "ETTh2": train2_file
    }
    
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    
    missing_values = "no"
    frequency = "hourly"
    univariate = "yes"

    series_list = []
    
    # Process each file and extract the 7 target columns as separate time series
    for file_prefix, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            for col in target_columns:
                if col in df.columns:
                    series_list.append({
                        "ts_id": f"{file_prefix}_{col}",
                        "values": df[col].astype(str).tolist()
                    })

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write .tsf metadata headers
        f.write("@relation ETTh_dataset\n")
        f.write(f"@frequency {frequency}\n")
        f.write(f"@missing_values {missing_values}\n")
        f.write(f"@univariate {univariate}\n")
        f.write("@data\n")
        
        # Write each extracted time series row
        for idx, series in enumerate(series_list):
            ts_increment = f"T{idx + 1}"
            ts_id = series["ts_id"]
            values = ",".join(series["values"])
            
            # Write in format: {ts increment number}:{ts ID}:{list of values}
            f.write(f"{ts_increment}:{ts_id}:{values}\n")





def convert_ETT15min_to_tsf():
    """
    Converts the ETT dataset from its original CSV format to the standard .tsf format.
    """
    train1_file = get_abs_path("../original/ETTm1.csv")
    train2_file = get_abs_path("../original/ETTm2.csv")
    """
    Each file is like:

    date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
    2016-07-01 00:00:00,5.827000141143799,2.009000062942505,1.5989999771118164,0.4620000123977661,4.203000068664552,1.3400000333786009,30.5310001373291
    2016-07-01 01:00:00,5.692999839782715,2.075999975204468,1.4919999837875366,0.4259999990463257,4.142000198364259,1.371000051498413,27.78700065612793
    2016-07-01 02:00:00,5.1570000648498535,1.741000056266785,1.2790000438690186,0.35499998927116394,3.776999950408936,1.218000054359436,27.78700065612793

    so from every file i extract 7 (HUFL,HULL,MUFL,MULL,LUFL,LULL,OT) time series, so in total 14 time series.
    """
    output_file = get_tmp_path("ETT15min_dataset.tsf")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = {
        "ETT15min1": train1_file,
        "ETT15min2": train2_file
    }
    
    target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    
    missing_values = "no"
    frequency = "15 min"
    univariate = "yes"

    series_list = []
    
    # Process each file and extract the 7 target columns as separate time series
    for file_prefix, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            for col in target_columns:
                if col in df.columns:
                    series_list.append({
                        "ts_id": f"{file_prefix}_{col}",
                        "values": df[col].astype(str).tolist()
                    })

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write .tsf metadata headers
        f.write("@relation ETT15min_dataset\n")
        f.write(f"@frequency {frequency}\n")
        f.write(f"@missing_values {missing_values}\n")
        f.write(f"@univariate {univariate}\n")
        f.write("@data\n")
        
        # Write each extracted time series row
        for idx, series in enumerate(series_list):
            ts_increment = f"T{idx + 1}"
            ts_id = series["ts_id"]
            values = ",".join(series["values"])
            
            # Write in format: {ts increment number}:{ts ID}:{list of values}
            f.write(f"{ts_increment}:{ts_id}:{values}\n")

    

if __name__ == "__main__":
    #convert_m5_to_tsf()
    #convert_ETTh_to_tsf()
    convert_ETT15min_to_tsf()