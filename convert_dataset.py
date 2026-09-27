import os
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm


# ============================================================
# 設定
# ============================================================

# CSV データが保存されているディレクトリ
INPUT_PATH = "/mnt/data1/yano/work/SoccerCompetition2025/robocup2d_data_processed2"

# Hugging Face Dataset の保存先
OUTPUT_PATH = "/mnt/data1/yano/work/SoccerCompetition2025/0922_2155_converted_dataset"


# ============================================================
# CSV を読み込む
# ============================================================

def load_csv_files(path):
    data_list = []

    # 対象となるディレクトリ数を取得
    total_dirs = sum(1 for _ in os.walk(path))

    for root, dirs, files in tqdm(
        os.walk(path),
        total=total_dirs,
        desc="Scanning directories",
    ):
        for file in tqdm(files, leave=False, desc=f"Reading {root}"):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)

            df = pd.read_csv(file_path)

            # NaN -> None
            df = df.where(pd.notnull(df), None)

            # int64 -> float64
            # Dataset.from_generator() で複数 CSV の型を
            # 揃えるため
            int_cols = df.select_dtypes(include=["int64"]).columns

            for col in int_cols:
                df[col] = df[col].astype("float64")

            data_list.append(df)

    return data_list


# ============================================================
# Dataset を作成
# ============================================================

def create_dataset(data_list):
    def generator():
        for df in tqdm(data_list, desc="Creating dataset"):
            # DataFrame の各行を辞書として返す
            for row in df.to_dict(orient="records"):
                yield row

    dataset = Dataset.from_generator(generator)

    return dataset


# ============================================================
# メイン処理
# ============================================================

def main():
    print("Loading CSV files...")
    data_list = load_csv_files(INPUT_PATH)

    print(f"Number of CSV files: {len(data_list)}")

    print("Creating Hugging Face Dataset...")
    dataset = create_dataset(data_list)

    print(dataset)

    # 保存先の親ディレクトリが存在しない場合に作成
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Saving dataset to: {OUTPUT_PATH}")
    dataset.save_to_disk(OUTPUT_PATH)

    print("Dataset saved successfully.")

    # 保存した Dataset を確認
    print("Loading saved dataset...")
    loaded_dataset = load_from_disk(OUTPUT_PATH)

    print(loaded_dataset)
    print(f"Number of rows: {len(loaded_dataset)}")


if __name__ == "__main__":
    main()