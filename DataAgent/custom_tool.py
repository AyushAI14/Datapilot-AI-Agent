import requests
# from pathlib import Path
import pandas as pd
import os


RAW_DIR = "/home/ayush/Documents/AI/Projects/GENAI/Datapilot-AI-Agent/data/raw"


def save_to_raw(download_url: str, original_filename: str) -> dict:
    """
    Download dataset file safely using download_url and save to path.
    """

    try:
        os.makedirs(RAW_DIR, exist_ok=True)

        save_path = os.path.join(RAW_DIR, original_filename)
        print(f"File created at {save_path}")
        with requests.get(download_url, stream=True, timeout=120) as response:
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return {
            "status": "success",
            "saved_path": save_path,
            "filename": original_filename,
        }
    except requests.exceptions.HTTPError as e:
        return {
            "status": "error",
            "message": f"HTTP Error: {str(e)}",
            "url": download_url,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


RAW_DIR = "/home/ayush/Documents/AI/Projects/GENAI/Datapilot-AI-Agent/data/raw"
PROCESSED_DIR = "/home/ayush/Documents/AI/Projects/GENAI/Datapilot-AI-Agent/data/processed"

def load_local_data() -> dict:
    """
    Look into RAW_DIR, find the most recent CSV/Parquet file,
    load it, and return preview + metadata.
    """
    if not os.path.exists(RAW_DIR):
        return {"status": "error", "message": f"RAW_DIR does not exist: {RAW_DIR}"}

    candidates = [
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".csv") or f.endswith(".parquet")
    ]
    if not candidates:
        return {"status": "error", "message": f"No CSV/Parquet files found in {RAW_DIR}"}

    def full(p): return os.path.join(RAW_DIR, p)
    latest_file = max(candidates, key=lambda f: os.path.getmtime(full(f)))
    file_path = full(latest_file)

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        return {"status": "error", "message": "Unsupported format"}

    return {
        "status": "success",
        "file_path": file_path,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "preview": df.head(50).to_dict(),
    }


def run_cleaning_code(file_path: str, cleaning_code: str) -> dict:
    """
    Load df from file_path, run user-provided cleaning_code that modifies `df`,
    then save cleaned df to PROCESSED_DIR and return path.
    """
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        ext = ".csv"
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
        ext = ".parquet"
    else:
        return {"status": "error", "message": "Unsupported format"}

    local_env = {"pd": pd, "df": df}
    exec(cleaning_code, {}, local_env)  # cleaning_code must update `df`
    df_clean = local_env["df"]
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "clean_data" + ext)  # 👈 fixed name
    out_path = os.path.abspath(out_path)

    if ext == ".csv":
        df_clean.to_csv(out_path, index=False)
    else:
        df_clean.to_parquet(out_path, index=False)

    return {
        "status": "success",
        "cleaned_file": out_path,
        "shape": df_clean.shape,
        "columns": df_clean.columns.tolist(),
    }