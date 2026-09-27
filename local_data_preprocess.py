"""
SoccerCompetition2025 Dataset
前処理してNumPy形式で保存する。

メモリ節約のため、Dataset全体をPandasに変換せず、
チャンク単位で処理する。

例:
    python data_preprocess.py

    python data_preprocess.py \
        --dataset-path /path/to/dataset \
        --output-base-dir /path/to/output
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


# ============================================================
# 設定
# ============================================================

DEFAULT_DATASET_PATH = Path(
    "/mnt/data1/yano/work/SoccerCompetition2025/"
    "All_Datasets_SoccerCompetition2025_20260827_024418"
)

DEFAULT_OUTPUT_BASE_DIR = Path(
    "/mnt/data1/yano/work/SoccerCompetition2025"
)

CHUNK_SIZE = 50


# ============================================================
# Min-Max
# ============================================================

class MinMax:
    """列ごとのMin-Max正規化と逆変換を扱う。"""

    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

        if self.min >= self.max:
            raise ValueError("min must be less than max")

    def __call__(self, value):
        return (value - self.min) / (self.max - self.min)

    def inverse(self, value):
        return value * (self.max - self.min) + self.min

    def __repr__(self):
        return f"MinMax({self.min}, {self.max})"


# ============================================================
# 左右反転
# ============================================================

def swap_rl(df: pd.DataFrame) -> pd.DataFrame:
    """
    右ゴール側のデータを
    左ゴール側と同じ向きにそろえる。
    """

    df = df.copy()

    # チーム名を交換
    df["l_name"], df["r_name"] = (
        df["r_name"],
        df["l_name"]
    )

    # ボールのx座標を反転
    df["b_x"] *= -1

    # 選手座標を左右入れ替え
    for index in range(1, 12):

        l_x = f"l{index}_x"
        r_x = f"r{index}_x"

        l_y = f"l{index}_y"
        r_y = f"r{index}_y"

        df[l_x], df[r_x] = (
            -df[r_x].values,
            -df[l_x].values
        )

        df[l_y], df[r_y] = (
            df[r_y].values,
            df[l_y].values
        )

    return df


# ============================================================
# DataFrameの基本前処理
# ============================================================

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    1チャンク分のDataFrameを前処理する。
    """

    if df.empty:
        return df

    # NaNを含む行を削除
    df = df.dropna()

    if df.empty:
        return df

    # 数値列にinfがあれば、その行を削除
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) > 0:

        finite_mask = np.isfinite(
            df[numeric_columns].to_numpy()
        ).all(axis=1)

        df = df.loc[finite_mask]

    if df.empty:
        return df

    # goal_typeがgoal_rなら左右反転
    if "goal_type" in df.columns:

        mask = df["goal_type"] == "goal_r"

        if mask.any():

            # goal_rの行だけ反転
            df_r = df.loc[mask].copy()
            df_r = swap_rl(df_r)

            # 元DataFrameへ戻す
            df.loc[mask] = df_r

    return df


# ============================================================
# One-hot
# ============================================================

def name_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    左右チーム名をOne-hot表現に変換する。
    """

    df = df.copy()

    for index in range(10):

        df[f"l_name_{index}"] = (
            df["l_name"] == index
        ).astype(np.int8)

        df[f"r_name_{index}"] = (
            df["r_name"] == index
        ).astype(np.int8)

    return df


# ============================================================
# 不要列削除
# ============================================================

def drop_unnecessary_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    columns_to_drop = [
        "#",
        "cycle",
        "stopped",
        "playmode",
        "l_name",
        "r_name",
        "goal_type",
        "l_score",
        "r_score",
        "l_pen_score",
        "r_pen_score",
    ]

    # 実際に存在する列だけ削除
    columns_to_drop = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    return df.drop(
        columns=columns_to_drop
    )


# ============================================================
# Dataset → DataFrame
# ============================================================

def dataset_to_dataframe(
    dataset,
    start: int,
    end: int,
) -> pd.DataFrame:

    # Datasetの必要部分だけ取得
    data = dataset[start:end]

    # Dataset → DataFrame
    df = pd.DataFrame(data)

    return df


# ============================================================
# Train全体のMin/Maxを計算
# ============================================================

def calculate_min_max(
    dataset,
    chunk_size: int = CHUNK_SIZE,
):
    """
    Dataset全体を一度にPandasへ変換せず、
    チャンク単位で各列のmin/maxを計算する。

    戻り値:
        min_max_d
    """

    print(
        "=== Start calculating Min/Max ===",
        flush=True
    )

    global_min = {}
    global_max = {}

    total_rows = len(dataset)

    for start in range(
        0,
        total_rows,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_rows
        )

        df = dataset_to_dataframe(
            dataset,
            start,
            end
        )

        df = basic_preprocess(df)

        if df.empty:
            continue

        df = name_onehot(df)

        df = drop_unnecessary_columns(df)

        # 数値列
        for column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if values.isna().any():
                continue

            current_min = values.min()
            current_max = values.max()

            if column not in global_min:

                global_min[column] = current_min
                global_max[column] = current_max

            else:

                if current_min < global_min[column]:
                    global_min[column] = current_min

                if current_max > global_max[column]:
                    global_max[column] = current_max

        if (
            start % (chunk_size * 1000)
            == 0
        ):

            print(
                f"Min/Max progress: "
                f"{end:,}/{total_rows:,}",
                flush=True
            )

    # MinMaxオブジェクトへ変換
    min_max_d = {}

    for column in global_min:

        min_value = global_min[column]
        max_value = global_max[column]

        if min_value == max_value:

            min_max_d[column] = MinMax(
                min_value,
                min_value + 1
            )

        else:

            min_max_d[column] = MinMax(
                min_value,
                max_value
            )

    print(
        "=== Min/Max calculation completed ===",
        flush=True
    )

    return min_max_d


# ============================================================
# チャンクをNumPyへ変換
# ============================================================

def preprocess_chunk(
    df: pd.DataFrame,
    min_max_d,
) -> tuple[np.ndarray | None, np.ndarray | None]:

    df = basic_preprocess(df)

    if df.empty:
        return None, None

    # One-hot
    df = name_onehot(df)

    # 不要列削除
    df = drop_unnecessary_columns(df)

    # Min-Max
    for column in df.columns:

        scaler = min_max_d[column]

        df[column] = (
            scaler(df[column])
        )

    # float32へ変換
    array = df.to_numpy(
        dtype=np.float32
    )

    columns = df.columns.to_numpy()

    return array, columns


# ============================================================
# Dataset → NumPy shards
# ============================================================

def preprocess_dataset(
    dataset,
    output_dir: Path,
    split_name: str,
    min_max_d,
    chunk_size: int = CHUNK_SIZE,
):
    """
    Datasetをチャンク単位で処理して
    NumPy shardとして保存する。

    50行単位のデータをそのまま保存する。
    """

    split_dir = output_dir / split_name

    split_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    total_rows = len(dataset)

    shard_index = 0
    total_valid_rows = 0

    columns = None

    print(
        f"=== Start preprocessing {split_name} ===",
        flush=True
    )

    for start in range(
        0,
        total_rows,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            total_rows
        )

        df = dataset_to_dataframe(
            dataset,
            start,
            end
        )

        array, chunk_columns = preprocess_chunk(
            df,
            min_max_d
        )

        if array is None:
            continue

        # 列名を保存
        if columns is None:
            columns = chunk_columns

        # 50行未満のチャンクは除外
        if len(array) != chunk_size:
            print(
                f"Skipping final incomplete chunk: "
                f"{len(array)} rows",
                flush=True
            )
            continue

        shard_path = (
            split_dir
            / f"{split_name}_{shard_index:06d}.npy"
        )

        np.save(
            shard_path,
            array
        )

        shard_index += 1
        total_valid_rows += len(array)

        if (
            shard_index % 1000
            == 0
        ):

            print(
                f"{split_name}: "
                f"{end:,}/{total_rows:,} rows "
                f"-> {shard_index:,} shards",
                flush=True
            )

        # チャンクのDataFrameを削除
        del df
        del array

    if columns is None:
        raise ValueError(
            f"No valid data found in {split_name}."
        )

    # 列名
    np.save(
        output_dir
        / f"{split_name}_columns.npy",
        columns
    )

    print(
        f"=== {split_name} completed ===",
        flush=True
    )

    print(
        f"Valid rows: {total_valid_rows:,}",
        flush=True
    )

    print(
        f"Shards: {shard_index:,}",
        flush=True
    )

    return total_valid_rows, shard_index


# ============================================================
# 引数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH
    )

    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE_DIR
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main():

    args = parse_args()

    if not args.dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{args.dataset_path}"
        )

    # --------------------------------------------------------
    # 出力先
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        args.output_base_dir
        / f"datas_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print(
        f"Output directory: {output_dir}",
        flush=True
    )

    # --------------------------------------------------------
    # Dataset読み込み
    # --------------------------------------------------------

    print(
        "=== Loading Dataset ===",
        flush=True
    )

    dataset_dict = load_from_disk(
        str(args.dataset_path)
    )

    dataset = dataset_dict["train"]

    print(
        f"Dataset rows: {len(dataset):,}",
        flush=True
    )

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    print(
        "=== Train/Test split ===",
        flush=True
    )

    split_dataset = dataset.train_test_split(
        test_size=0.2,
        seed=42
    )

    train_raw = split_dataset["train"]
    test_raw = split_dataset["test"]

    print(
        f"Train: {len(train_raw):,}",
        flush=True
    )

    print(
        f"Test : {len(test_raw):,}",
        flush=True
    )

    # --------------------------------------------------------
    # TrainからMin/Maxを計算
    # --------------------------------------------------------

    min_max_d = calculate_min_max(
        train_raw,
        chunk_size=CHUNK_SIZE
    )

    # --------------------------------------------------------
    # Min/Max保存
    # --------------------------------------------------------

    np.save(
        output_dir / "min_max_d.npy",
        min_max_d,
        allow_pickle=True
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_rows, train_shards = preprocess_dataset(
        train_raw,
        output_dir,
        "train",
        min_max_d,
        chunk_size=CHUNK_SIZE
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_rows, test_shards = preprocess_dataset(
        test_raw,
        output_dir,
        "test",
        min_max_d,
        chunk_size=CHUNK_SIZE
    )

    # --------------------------------------------------------
    # 完了
    # --------------------------------------------------------

    print(
        "\n========================================",
        flush=True
    )

    print(
        "Preprocessing completed successfully.",
        flush=True
    )

    print(
        f"Output: {output_dir}",
        flush=True
    )

    print(
        f"Train rows : {train_rows:,}",
        flush=True
    )

    print(
        f"Train shards: {train_shards:,}",
        flush=True
    )

    print(
        f"Test rows  : {test_rows:,}",
        flush=True
    )

    print(
        f"Test shards : {test_shards:,}",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()