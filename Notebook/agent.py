import os
import nest_asyncio
from dotenv import load_dotenv
import asyncio
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.code_executors import BuiltInCodeExecutor


from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("✅ ADK components imported successfully.")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
print("✅ Gemini API key setup complete.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)
print("✅ HttpRetryOptions Intialized successfully.")


Planner_instruction = """
You are the PlannerAgent in a SequentialAgent pipeline.

YOUR ROLE IS VERY LIMITED:
- You DO NOT search the web.
- You DO NOT scrape pages.
- You DO NOT download or inspect data.
- You ONLY read the 'page_content' provided by the previous agent (the WebScraperAgent)
  and create a machine learning project plan based on it.

------------------- INPUT YOU RECEIVE -------------------
- You will receive a detailed dataset summary (markdown) with:
  • Dataset title/name
  • Description
  • File details (names, extensions, sizes, descriptions if visible)
  • Metadata (license, creator, tags, etc.)

- You MUST use ONLY this information.
- You MUST NOT invent dataset columns or details that are not mentioned.

------------------- WHAT YOU MUST PRODUCE -------------------
Create a clear, practical, step-by-step ML plan for building a classification model.
Your output MUST include these sections:

## 🎯 Objective
- Based on the dataset description ONLY, define the classification goal.

## 📥 Data Understanding & Access
- Specify which file(s) will be used based on the scraper summary.
- Do NOT assume columns unless listed; if missing, say they must be inspected after loading.

## 🧹 Data Preparation Plan
- Outline data cleaning steps (NULL checks, formatting, encoding, splitting).
- Do NOT fabricate columns or statistical assumptions.

## 📊 Exploratory Analysis (EDA)
- List EDA tasks (visualizations, summary stats, distributions).
- DO NOT mention specific plots for non-existent fields.

## 🤖 Model Training Strategy
- Suggest suitable ML models for classification (e.g., Logistic Regression, Random Forest, XGBoost).
- Specify validation strategy (e.g., train/test split, cross-validation).
- No training code. Just plan.

## 🧪 Evaluation Metrics
- List classification metrics (Accuracy, F1, Precision, Recall, ROC-AUC).

## 🚀 Final Deliverables
- What outputs the pipeline should produce (saved model, evaluation report, feature pipeline, etc.)
- Do NOT implement them, only describe.

------------------- STRICT RULES -------------------
- DO NOT scrape or download anything.
- DO NOT talk about how to write code or give code examples.
- DO NOT fabricate dataset details not visible in the scraped summary.
- DO NOT make assumptions about column names or target labels unless explicitly written.
- DO NOT perform calculations or analysis. Only plan.

You are ONLY responsible for transforming the summary into a clean, realistic ML project plan.
"""

Ingestion_instruction_with_save = """
You are DataIngestionAgent. Your job is to return a Kaggle dataset download URL and Kaggle dataset page url, and only save the file if the user clearly asks to download it.

--- BEHAVIOR RULES ---

1) If the user's message contains the word "download":
   - Use `search_datasets` to find the dataset.
   - Use `list_dataset_files` to pick the most relevant file (prefer CSV).
   - Use `download_dataset` to get the file's download URL.
   - print the download URL 
   - Then call the local tool `save_to_raw(download_url)` to store the data in data/raw.
   - Return ONLY this JSON:
     {
       "status": "saved",
     }

2) If the user does NOT ask to download:
   - Only provide dataset metadata.
   - Use:
       search_datasets
       get_dataset_info
       list_dataset_files

    -  **SEARCH & CHAINING:** Call **'search_datasets'**, then **'get_dataset_info'**, and then **'list_dataset_files'** for the single most relevant dataset.
    -  **OUTPUT TRANSFORMATION:** Transform the raw data into the JSON OUTPUT SCHEMA (original schema with 'datasets' array).
   - Return ONLY JSON:
     {
       "datasets": [...],
       "errors": []
     }

--- RULES ---
- Output JSON ONLY.
- No explanations, no extra text.
- Never save files unless the user explicitly asks to download.
- When saving, always call `save_to_raw(download_url)` (no other tool).

--- ALLOWED TOOLS ---
- search_datasets
- get_dataset_info
- list_dataset_files
- download_dataset
- save_to_raw
"""


Scraper_instruction = """
You are a WebScraperAgent in a SequentialAgent pipeline.

YOUR ROLE IS EXTREMELY LIMITED:

- You DO NOT answer the user directly.
- You DO NOT make plans, build models, or comment on ML tasks.
- You ONLY scrape a dataset page URL provided by the previous agent
  and produce a DETAILED SUMMARY based strictly on the scraped markdown.

------------------ INPUT SOURCE ------------------
- Extract the dataset page URL ONLY from the previous agent output under the key `Dataset_files`.
- Do NOT search, guess, or change the URL.
- If the URL is missing or invalid, return an error and STOP.

------------------ TOOLS ------------------
- You may ONLY call: `firecrawl_scrape`
- You MUST call `firecrawl_scrape` EXACTLY ONCE using the extracted URL.

------------------ WHAT YOU MUST DO ------------------
1. Extract the dataset page URL from `Dataset_files`.
2. Call `firecrawl_scrape` with that URL.
3. Read the returned `markdown` and produce a detailed summary that includes:
   - Dataset title/name (if visible).
   - Dataset description.
   - Purpose or use case (ONLY if explicitly written on the page).
   - A breakdown of dataset files (names, extensions, sizes/descriptions if visible).
   - Any additional metadata clearly shown (e.g., creator, license, versions, tags).

------------------ OUTPUT FORMAT (MANDATORY) ------------------
Your output MUST be a single structured markdown summary:

# <Dataset Title (if visible)>
**URL:** <dataset page URL>

## 📌 Dataset Description
- Summarize what the dataset page says. Only use information visible in the scraped markdown.

## 📂 Files Overview
- List each file found on the page with name, type, and any visible details (size, description, etc.).
- Do NOT invent missing information.

## 🏷 Metadata (If Present on Page)
- Creator / author
- Update dates
- License
- Tags / categories
- Any other page metadata shown

------------------ STRICT RULES ------------------
- DO NOT paste the full markdown.
- DO NOT summarize anything that is not in the scraped markdown.
- DO NOT create plans, pipelines, or next steps.
- DO NOT call tools other than `firecrawl_scrape`.
- DO NOT download, save, or modify any dataset.
- DO NOT hallucinate or fabricate missing information.

Your ONLY job is to scrape and summarize the dataset page for the next agent.
"""


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


os.environ['KAGGLE_USERNAME'] = "ayushvishwakarma14"
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')


# Data Ingestion Mcp Agent
mcp_kaggle_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=[
                '-y',
                'mcp-remote',
                'https://www.kaggle.com/mcp'
            ]
        ),
        timeout=60,
    ),
    tool_filter=[
        "search_datasets", 
        "get_dataset_info", 
        "list_dataset_files",
        "download_dataset"
    ]
)

# Data Ingestion Agent: Its job is to find the dataset and its info from kaggle mcp
ingest_agent = Agent(
    name="DataIngestion_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction=Ingestion_instruction_with_save,
    tools=[mcp_kaggle_server,save_to_raw],
    output_key="Dataset_files", 
)

# async def run_ingestion():
#     """Defines the async context for running the agent."""
#     runner = InMemoryRunner(agent = ingest_agent)
#     response = await runner.run_debug("Find a small Kaggle dataset about Netflix movie ratings and download it for the pipeline.")
#     print(response)

# await run_ingestion()

# print("✅ Ingest_agent created.")

# ---------------- FIRECRAWL MCP SERVER ----------------
mcp_firecrawl_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env={
                # Put your API key here OR set as system env
                "FIRECRAWL_API_KEY": os.environ.get("FIRECRAWL_API_KEY", os.getenv('FIRECRAWL_API_KEY'))
            },
        ),
        timeout=60,
    ),
    tool_filter=["firecrawl_scrape"]  # Only expose the scrape tool
)

# ---------------- FIRECRAWL SCRAPER AGENT ----------------
scraper_agent = Agent(
    name="FirecrawlWebScraper",
    model=Gemini(
        model="gemini-2.5-flash"
    ),
    instruction=Scraper_instruction,
    tools=[mcp_firecrawl_server],
    output_key="page_content"
)


# async def run_ingestion():
#     """Defines the async context for running the agent."""
#     runner = InMemoryRunner(agent = scraper_agent)
#     response = await runner.run_debug("Scrape this link : https://www.kaggle.com/learn-guide/5-day-genai and return a markdown of it")
#     # print(response)

# await run_ingestion()

print("✅ scraper_agent created.")

# Planner Agent: Its job is to use the google_search tool and present findings.
Planner_agent = Agent(
    name="Planner_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction=Planner_instruction,
    tools=[google_search],
    output_key="Planner_findings",  # The result of this agent will be stored in the session state with this key.
)

print("✅ Planner_agent created.")

# runner = InMemoryRunner(agent = Planner_agent)
# response = await runner.run_debug(Query)

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


Cleaning_instruction = """
You are a DataCleaningAgent in a local pipeline.

GOAL:
- Generate SAFE Python/Pandas cleaning code that takes a DataFrame `df`,
  cleans it, and produces an improved DataFrame.
- Then execute that code using the tool `run_cleaning_code` to save the cleaned dataset.
- You MUST show the exact code used.

ALLOWED TOOLS:
- load_local_data
- run_cleaning_code

WORKFLOW (STRICT):
1. First, call `load_local_data` (NO arguments).
   - It will automatically select the most recent CSV/Parquet file from:
     /home/ayush/Documents/AI/Projects/GENAI/Datapilot-AI-Agent/data/raw/
   - It returns:
     - file_path
     - columns
     - shape
     - preview (first 50 rows)

2. Inspect:
   - columns
   - shape
   - preview

3. Generate a SAFE cleaning code string that:
   - assumes `df` is already loaded
   - NEVER removes ALL rows or ALL columns
   - NEVER removes the target/label column if one is present
   - uses only SAFE transformations, such as:
       • removing duplicates
       • dropping KNOWN useless ID columns ONLY if they clearly exist
       • handling missing values (ffill/bfill/mean/median ONLY if numeric)
       • fixing dtypes (e.g., categorical conversion)
   - MUST reassign the DataFrame back to `df` or modify `df` inplace
   - MUST end with a valid DataFrame in variable name `df`

4. Call `run_cleaning_code(file_path, cleaning_code)` passing:
   - the SAME `file_path` returned by load_local_data
   - the cleaning code string

MANDATORY OUTPUT FORMAT:
Return a JSON object exactly like this:
{
  "cleaning_code_used": "<the exact python code as a string>",
  "cleaning_result": <the JSON returned by run_cleaning_code>
}

STRICT RULES:
- DO NOT guess column names that do not appear in the preview.
- DO NOT remove the label/target column if it exists (e.g., diagnosis, target, survived, etc.).
- DO NOT drop more than 1 identifier-like column (e.g., id, patient_id) without proof.
- DO NOT convert non-numeric columns to numeric unless they clearly contain numeric values.
- DO NOT output explanations outside the required JSON format.
- DO NOT change the output structure.

Your code MUST ALWAYS produce a valid DataFrame named `df`. If unsure, choose the safest option.
"""


code_agent = Agent(
    name="code_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # code_executor=BuiltInCodeExecutor(),
    instruction=Cleaning_instruction,
    tools=[load_local_data,run_cleaning_code],
    output_key="code"
)

print("✅ code_agent created.")

# runner = InMemoryRunner(agent = code_agent)
# response = await runner.run_debug("Write a code to clean a Breast Cancer Prediction dataset")


# Sequential Orchestrator
root_agent = SequentialAgent(
    name="DataSciencePipeline",
    sub_agents=[ingest_agent,scraper_agent, Planner_agent,code_agent],
)

print("✅ Sequential Agent created.")

