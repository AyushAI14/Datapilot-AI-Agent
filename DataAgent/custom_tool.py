import os
import requests

def save_to_raw(download_url: str, original_filename: str) -> str:
    """
    Just download the file and save it to data/raw without extracting.
    """

    raw_dir = "../data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Download the file bytes
    resp = requests.get(download_url)
    resp.raise_for_status()

    # Save in raw folder
    out_path = os.path.join(raw_dir, original_filename)
    with open(out_path, "wb") as f:
        f.write(resp.content)

    return f"File saved to {out_path}"


import os
import pandas as pd

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
