import os
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from datetime import datetime

# ============================================================
# 設定
# ============================================================

input_path = (
    "/mnt/data1/yano/work/SoccerCompetition2025/"
    "robocup2d_data_processed2"
)

base_output_path = (
    "/mnt/data1/yano/work/SoccerCompetition2025"
)

# 実行開始時刻
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 実行ごとに異なる保存先を作る
output_path = (
    f"{base_output_path}/"
    f"All_Datasets_SoccerCompetition2025_{timestamp}"
)

cache_path = (
    f"{base_output_path}/"
    f"dataset_generator_cache_{timestamp}"
)

print(f"Input : {input_path}", flush=True)
print(f"Output: {output_path}", flush=True)
print(f"Cache : {cache_path}", flush=True)


# ============================================================
# CSVファイル一覧を取得
# ============================================================

print("=== Searching CSV files ===", flush=True)

csv_files = []

for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(
                os.path.join(root, file)
            )

csv_files.sort()

print(
    f"CSV files: {len(csv_files):,}",
    flush=True
)


# ============================================================
# Generator
# ============================================================

def generate_examples(files):
    """
    CSVを1つずつ読み込み、
    CSVの各行をdictとしてyieldする。
    """

    for file_index, filepath in enumerate(files, 1):

        try:
            df = pd.read_csv(filepath)

            # NaN → None
            df = df.where(pd.notnull(df), None)

            # 元コードと同じく int64 → float64
            int_cols = df.select_dtypes(
                include=["int64"]
            ).columns

            for col in int_cols:
                df[col] = df[col].astype("float64")

            # DataFrame全体ではなく「1行ずつ」渡す
            for row in df.to_dict(orient="records"):
                yield row

            # 進捗表示
            if file_index % 1000 == 0:
                print(
                    f"Processed CSV: "
                    f"{file_index:,}/{len(files):,}",
                    flush=True
                )

        except Exception as e:

            print(
                "\n========================================",
                flush=True
            )
            print(
                f"ERROR: {filepath}",
                flush=True
            )
            print(
                f"{type(e).__name__}: {e}",
                flush=True
            )
            print(
                "========================================\n",
                flush=True
            )

            # エラーを隠さず停止
            raise


# ============================================================
# Datasetを作成
# ============================================================

print(
    "=== Start Dataset.from_generator ===",
    flush=True
)

dataset = Dataset.from_generator(
    generate_examples,
    gen_kwargs={
        "files": csv_files
    },
    cache_dir=cache_path,
    keep_in_memory=False,
    num_proc=1,
)

print(
    "=== Dataset.from_generator completed ===",
    flush=True
)

print(
    f"Number of rows: {len(dataset):,}",
    flush=True
)

print(
    f"Number of columns: {len(dataset.column_names)}",
    flush=True
)

print(
    f"Columns: {dataset.column_names}",
    flush=True
)


# ============================================================
# DatasetDictにする
# ============================================================

dataset_dict = DatasetDict({
    "train": dataset
})


# ============================================================
# Localに保存
# ============================================================

print(
    "=== Start save_to_disk ===",
    flush=True
)

dataset_dict.save_to_disk(
    output_path,
    max_shard_size="1GB"
)

print(
    "=== Local dataset saved successfully ===",
    flush=True
)

print(
    f"Saved to: {output_path}",
    flush=True
)