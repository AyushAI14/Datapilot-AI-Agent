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
   - Return ONLY Markdown format:
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
import requests
import os
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from DataAgent.custom_tool import save_to_raw
from DataAgent.agent_config import retry_config
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
os.environ['KAGGLE_USERNAME'] = "ayushvishwakarma14"


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

# async def main():
#     runner = InMemoryRunner(agent=ingest_agent)
    
#     async with runner:
#         response = await runner.run_debug("Find a  Kaggle dataset about churn prediction and download it for the pipeline.")
#         print(response)

# if __name__ == "__main__":
#     asyncio.run(main())