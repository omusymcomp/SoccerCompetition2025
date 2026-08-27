import os
import pandas as pd
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

path = "/mnt/data1/yano/work/SoccerCompetition2025/robocup2d_data_processed2"
data_list = []
for root, dirs, files in tqdm(
    os.walk(path), total=len([None for _, _, _ in os.walk(path)])
):
    for file in tqdm(files, leave=False):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(root, file))
            df = df.where(pd.notnull(df), None)
            int_cols = df.select_dtypes(include=["int64"]).columns
            for col in int_cols:
                df[col] = df[col].astype("float64")
            data_list.append(df)
            
def generator(data):
    def _generator():
        for d in tqdm(data):
            yield d

    return _generator

print("=== Start Dataset.from_generator ===", flush=True)

dataset = Dataset.from_generator(generator(data_list))

print("=== Dataset creation completed ===", flush=True)

print("=== Start push_to_hub ===", flush=True)

dataset.push_to_hub("Shota3232/All_Datasets_SoccerCompetition2025", private=True, token="xxx")

print("=== Upload completed ===", flush=True)