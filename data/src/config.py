"""
original_filename field is only used for data tracking purpose, it is not used on the code
"""

def get_datasets_metadata():
    return {

        "M4 Yearly": {
            "frequency": "year",
            "prediction_length": 6,
            "original_filename": "m4_yearly_dataset.tsf",
            "parquet_dirpath": "M4y",
            "source": "https://forecastingdata.org/",
        },
        "M4 Quarterly": {
            "frequency": "quarter",
            "prediction_length": 8,
            "original_filename": "m4_quarterly_dataset.tsf",
            "parquet_dirpath": "M4q",
            "source": "https://forecastingdata.org/",
        },
        "M4 Monthly": {
            "frequency": "month",
            "prediction_length": 18,
            "original_filename": "m4_monthly_dataset.tsf",
            "parquet_dirpath": "M4m",
            "source": "https://forecastingdata.org/",
        },
        "M3 Yearly": {
            "frequency": "year",
            "prediction_length": 6,
            "original_filename": "m3_yearly_dataset.tsf",
            "parquet_dirpath": "M3y",
            "source": "https://forecastingdata.org/",
        },
        "M3 Quarterly": {
            "frequency": "quarter",
            "prediction_length": 8,
            "original_filename": "m3_quarterly_dataset.tsf",
            "parquet_dirpath": "M3q",
            "source": "https://forecastingdata.org/",
        },
        "M3 Monthly": {
            "frequency": "month",
            "prediction_length": 18,
            "original_filename": "m3_monthly_dataset.tsf",
            "parquet_dirpath": "M3m",
            "source": "https://forecastingdata.org/",
        },
        "Tourism Yearly": {
            "frequency": "year",
            "prediction_length": 6,
            "original_filename": "tourism_yearly_dataset.tsf",
            "parquet_dirpath": "Tourismy",
            "source": "https://forecastingdata.org/",
        },
        "Tourism Quarterly": {
            "frequency": "quarter",
            "prediction_length": 8,
            "original_filename": "tourism_quarterly_dataset.tsf",
            "parquet_dirpath": "Tourismq",
            "source": "https://forecastingdata.org/",
        },
        "Tourism Monthly": {
            "frequency": "month",
            "prediction_length": 18,
            "original_filename": "tourism_monthly_dataset.tsf",
            "parquet_dirpath": "Tourismm",
            "source": "https://forecastingdata.org/",
        },
        "Weather": {
            "frequency": "day",
            "prediction_length": 30,
            "original_filename": "weather_dataset.tsf",
            "parquet_dirpath": "Weather",
            "source": "https://forecastingdata.org/",
        },
        "M5": {
            "frequency": "day",
            "prediction_length": 28,
            "original_filename": "sales_train_evaluation.csv, sales_test_evaluation.csv",
            "parquet_dirpath": "M5",
            "source": "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip",
        },
        "ETT Hourly": {
            "frequency": "hour",
            "prediction_length": 24,
            "original_filename": "ETTh1.csv, ETTh2.csv",
            "parquet_dirpath": "ETTh",
            "source": "https://github.com/zhouhaoyi/ETDataset",
        },
        "ETT 15Mins": {
            "frequency": "15 min",
            "prediction_length": 24,
            "original_filename": "ETTm1.csv, ETTm2.csv",
            "parquet_dirpath": "ETT15min",
            "source": "https://github.com/zhouhaoyi/ETDataset",
        },
        "Australian": {
            "frequency": "30 min",
            "prediction_length": 48,
            "original_filename": "australian_electricity_demand_dataset.tsf",
            "parquet_dirpath": "Australian",
            "source": "https://forecastingdata.org/",
        },


        # GENERATED
        "M4yqm Shuffled": {
            "frequency": "mixed",
            "prediction_length": None, # mixed
            "original_filename": "M4y/1.parquet, M4q/1.parquet, M4m/1.parquet",
            "parquet_dirpath": "Shuffled_M4yqm",
            "source": "shuffle.py",
        },



    }