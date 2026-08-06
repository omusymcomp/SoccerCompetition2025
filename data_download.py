import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
from itertools import product
import tempfile
import zipfile

from process_data import process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download_process.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def process_file(file_path: str) -> str:
    """
    ダウンロードしたファイルを加工して必要なデータだけを保存する処理を実装する。
    """
    df = pd.read_csv(file_path)
    dfs = process_data(df)
    processed_dir = file_path.replace("robocup2d_data", "robocup2d_data_processed2")
    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(processed_dir.replace(".csv", f"_{i}.csv"), index=False)
    return processed_dir


def download_and_process_file(team, file_name, base_url, save_dir):
    """
    指定したURLからファイルをダウンロードし、
    加工処理を実行後、元ファイルを削除する一連の処理
    """
    file_url = f"{base_url}/{team}/{file_name}"
    team_dir = os.path.join(save_dir, team)
    os.makedirs(team_dir, exist_ok=True)
    file_path = os.path.join(team_dir, file_name)

    try:
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        logging.error(f"Error downloading {file_url}: {e}", exc_info=True)
        return None

    try:
        processed_file = process_file(file_path)
        logging.info(f"Processed {file_path} -> {processed_file}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}", exc_info=True)
        processed_file = None

    try:
        os.remove(file_path)
    except Exception as e:
        logging.error(f"Error deleting {file_path}: {e}", exc_info=True)

    return processed_file


def download_and_process_zip(team, base_url, save_dir):
    """
    チームごとの zip ファイルをダウンロードして展開し、
    含まれる tracking.csv を順に加工する。
    """
    zip_url = f"{base_url}/{team}_csv.zip"
    team_dir = os.path.join(save_dir, team)
    os.makedirs(team_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, f"{team}.zip")

        try:
            with requests.get(zip_url, stream=True) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            logging.error(f"Error downloading {zip_url}: {e}", exc_info=True)
            return []

        processed_files = []
        try:
            with zipfile.ZipFile(zip_path) as zip_file:
                for member in zip_file.namelist():
                    if not member.endswith("tracking.csv"):
                        continue

                    extracted_path = zip_file.extract(member, path=team_dir)
                    try:
                        processed_file = process_file(extracted_path)
                        processed_files.append(processed_file)
                        logging.info(f"Processed {extracted_path} -> {processed_file}")
                    except Exception as e:
                        logging.error(
                            f"Error processing {extracted_path}: {e}", exc_info=True
                        )
                    finally:
                        try:
                            os.remove(extracted_path)
                        except Exception as e:
                            logging.error(
                                f"Error deleting {extracted_path}: {e}", exc_info=True
                            )
        except Exception as e:
            logging.error(f"Error extracting {zip_path}: {e}", exc_info=True)
            return []

    return processed_files


def process_subpath(team, base_url, save_dir, download_num=-1, max_workers=5):
    """
    1つのサブパス（チーム）内で、対象ファイルのダウンロード・加工・削除を並列に処理する。
    ダウンロードするファイル数は download_num で制限可能（-1 の場合は全件）。
    """
    results = download_and_process_zip(team, base_url, save_dir)
    if download_num != -1:
        results = results[:download_num]
    logging.info(f"Team {team}: {len(results)} files processed.")
    return results


def process_all_subpaths(base_url, subpaths, save_dir, download_num=-1, max_workers=5):
    """
    サブパスごとに順次、ファイルのダウンロード・加工・削除を実施する。
    HDDの容量制限を考慮し、1サブパス単位で処理を完結させる。
    """
    for team in subpaths:
        logging.info(f"Starting processing team: {team}")
        process_subpath(team, base_url, save_dir, download_num, max_workers)
        logging.info(f"Finished processing team: {team}")

# ここから実行される
teams = [
    "aeteam2024",
    "cyrus2024",
    "fra2024",
    "helios2024",
    "itandroids2024",
    "mars2024",
    "oxsy2024",
    "r2d2",
    "robocin2024",
    "yushan2024",
]

subpaths = {
    f"{team1}-{team2}" for team1, team2 in product(teams, teams) if team1 != team2
}

process_all_subpaths(
    base_url="https://alab.idsci.nagasaki-u.ac.jp/robocupdata/rc2024-roundrobin",
    subpaths=subpaths,
    save_dir="/mnt/data1/yano/work/SoccerCompetition2025/robocup2d_data",
    download_num=-1,
    max_workers=20,
)
