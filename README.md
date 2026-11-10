# FreaqNet

> Freaq = because it handles multiple fre(a)quencies, Net = because it is a neural network !

- `data/original`: local folder containing datasets in their original format before conversion to parquet (ignored in git)
- `data/parquet`: datasets converted to .parquet for disk and cpu optimization
- `data/src`: source code to convert the datasets to .parquet
- `src`: main source code for model training and evaluation

Update config in run.py and run full or quick/testing version:
```bash
python run.py
python run.py -q
```

