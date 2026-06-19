import pandas as pd
import os

filepath = os.path.join("data", "cleaned", "dataset_final_modelisation.csv")
if os.path.exists(filepath):
    df = pd.read_csv(filepath, nrows=5)
    print("Columns in dataset:")
    print(list(df.columns))
else:
    print("Dataset not found!")
