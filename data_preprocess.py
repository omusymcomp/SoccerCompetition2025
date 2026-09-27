"""データセット前処理ユーティリティ。

主にクラブ独自のデータフレームを正規化し、numpy 配列として保存します。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


DATA_ROOT = Path("/mnt/data1/yano/work/SoccerCompetition2025")
DATASET_PATH = DATA_ROOT / "dataset"
OUTPUT_DIR = Path("/mnt/data1/yano/work/SoccerCompetition2025/datas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MinMax:
    """Min-Max 正規化を扱うユーティリティクラス。"""
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value
        if self.min >= self.max:
            raise ValueError("min must be less than max")

    def __call__(self, x):
        """正規化を適用して 0-1 の範囲にスケーリングする。"""
        return (x - self.min) / (self.max - self.min)

    def inverse(self, x):
        """正規化を逆変換して元のスケールに戻す。"""
        return x * (self.max - self.min) + self.min

    def __repr__(self):
        return f"MinMax({self.min}, {self.max})"


def swap_rl(df):
    """左右（left/right）を入れ替えて座標符号を調整する。"""
    df["l_name"], df["r_name"] = df["r_name"], df["l_name"]
    df["b_x"] *= -1

    for i in range(1, 12):
        l_x, r_x = f"l{i}_x", f"r{i}_x"
        l_y, r_y = f"l{i}_y", f"r{i}_y"

        df[l_x], df[r_x] = -df[r_x].values, -df[l_x].values
        df[l_y], df[r_y] = df[r_y].values, df[l_y].values

    return df


def clean_and_merge_datasets(datas):
    """入力を受け取り、DataFrame リストに正規化して結合可能な形にする。"""
    if isinstance(datas, pd.DataFrame):
        frames = [datas]
    else:
        frames = []
        for data in datas:
            if not isinstance(data, pd.DataFrame):
                frames.append(pd.DataFrame(data))
            else:
                frames.append(data)

    data_list = []
    for df in frames:
        if df.empty:
            continue
        if df.isnull().values.any():
            continue

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            continue
        if not np.isfinite(numeric_df.values).all():
            continue

        if df["goal_type"].iloc[0] == "goal_r":
            df = swap_rl(df)
        data_list.append(df)

    if not data_list:
        return pd.DataFrame()

    return pd.concat(data_list, ignore_index=True)


def name_onehot(dfs):
    """選手名（カテゴリ）をワンホットエンコーディングする。"""
    for i in range(10):
        dfs[f"l_name_{i}"] = (dfs["l_name"] == i).astype(int)
        dfs[f"r_name_{i}"] = (dfs["r_name"] == i).astype(int)
    return dfs


def drop_unnecessary_columns(dfs):
    """不要列を削除する（存在しない列は無視）。"""
    return dfs.drop(
        columns=[
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
        ],
        errors="ignore",
    )


def min_max_normalize(dfs):
    """各列ごとに MinMax 正規化を行い、スケーラを返す。"""
    min_max_d = {col: MinMax(min(dfs[col]), max(dfs[col])) for col in dfs.columns}
    for col in dfs.columns:
        dfs[col] = min_max_d[col](dfs[col])
    return dfs, min_max_d


def divide_dataframe(dfs, df_size=50):
    """DataFrame を指定長のチャンクに分割して返す。"""
    df_list = []
    for i in range(0, len(dfs), df_size):
        df = dfs.iloc[i : i + df_size]
        df_list.append(df)
    return df_list


def list_to_numpy(dfs: list) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame のリストを numpy 配列と列名に変換する。"""
    cols = dfs[0].columns
    return np.array([df.values for df in dfs]).astype(np.float32), cols


def main() -> None:
    """前処理を実行して numpy ファイルとして出力するメイン関数。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_PATH}"
        )

    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Loaded dataset from: {DATASET_PATH}")

    train_raw, test_raw = dataset.train_test_split(test_size=0.2, seed=42).values()

    def preprocess_split(split_dataset):
        df = split_dataset.to_pandas()
        df = clean_and_merge_datasets(df)
        df = name_onehot(df)
        df = drop_unnecessary_columns(df)
        df, min_max_d = min_max_normalize(df)
        frames = divide_dataframe(df)
        arrays, cols = list_to_numpy(frames)
        return arrays, cols, min_max_d

    train, cols, min_max_d = preprocess_split(train_raw)
    test, _, _ = preprocess_split(test_raw)

    np.save(OUTPUT_DIR / "train.npy", train)
    np.save(OUTPUT_DIR / "cols.npy", cols)
    np.save(OUTPUT_DIR / "min_max_d.npy", min_max_d, allow_pickle=True)
    np.save(OUTPUT_DIR / "test.npy", test)

    print(f"Saved preprocessed data to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
