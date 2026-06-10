from Utils.Prompt import Ingestion_instruction_with_save
from DataAgent.custom_tool import save_to_raw
from DataAgent.agent_config import retry_config

from google.adk.agents import Agent
from google.adk.apps import App

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner,InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm

import warnings
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()


warnings.filterwarnings("ignore")

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# apis
os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY_PROMPT")
os.environ["OPENAI_API_KEY"] = os.getenv("OLLAMA_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OLLAMA_API_BASE")

groq = LiteLlm("groq/meta-llama/llama-4-scout-17b-16e-instruct")
ollama = LiteLlm("openai/gpt-oss:120b")

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
    name="dataingestion_agent",
    # model=groq,
    model=Gemini(
        model="gemini-3.1-flash-lite",
        # model = groq,
        retry_options=retry_config
    ),
    instruction=Ingestion_instruction_with_save,
    tools=[mcp_kaggle_server,save_to_raw],
    output_key="Dataset_files",
)

# session_service = InMemorySessionService()
app = App(name="dataset_app", root_agent=ingest_agent)

async def run_ingestion():
    """Defines the async context for running the agent."""
    async with InMemoryRunner(app=app) as runner:
        response = await runner.run_debug("Find a Kaggle dataset about Netflix movie ratings and download it using save_to_raw tool")
        print(response)

if __name__ == "__main__":
    asyncio.run(run_ingestion())
    print("Ingest_agent created.")
