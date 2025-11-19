import os
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams, StdioServerParameters
# Import the necessary ADK components for Agent, LLM, and Runner
from google.adk.agents import Agent
from google.adk.llms import Gemini
from google.adk.runners import InMemoryRunner
# Assuming 'retry_config' and 'Ingestion_instruction' are defined elsewhere

# --- 1. SET KAGGLE AUTH (Use actual values or secure loading) ---
# NOTE: Replace with your actual credentials or load them securely.
os.environ['KAGGLE_USERNAME'] = "ayushvishwakarma14"
os.environ['KAGGLE_KEY'] = "c80d1af1f212096bb46264a13951" 

# --- 2. Corrected McpToolset Configuration ---
# Data Ingestion Mcp Agent Toolset Definition
mcp_kaggle_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            # Command to run the MCP server: `npx -y mcp-remote <server-url>`
            command='npx',
            args=[
                '-y',
                'mcp-remote',
                'https://www.kaggle.com/mcp'
            ],
        ),
        timeout=60, # Increased timeout for potential long API calls/downloads
    ),
    # >> TOOL_FILTER moved here, outside of connection_params <<
    # These are the tools the LLM will see and be able to call.
    tool_filter=[
        "search_kaggle_datasets",
        "get_dataset_metadata",
        "download_kaggle_dataset"
    ]
)

# --- 3. Agent Definition (Looks correct) ---
# Data Ingestion Agent
ingest_agent = Agent(
    name="DataIngestion_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        # retry_options=retry_config # Assuming retry_config is defined
    ),
    # instruction=Ingestion_instruction, # Assuming Ingestion_instruction is defined
    tools=[mcp_kaggle_server],
    output_key="Dataset_files",
)

print("✅ Ingest_agent created.")

# --- 4. Agent Execution (Awaiting function context for 'await') ---
async def run_ingestion():
    runner = InMemoryRunner(agent = ingest_agent)
    response = await runner.run_debug("""Find a Kaggle dataset suitable for customer churn prediction and return its metadata and file list using the schema.""")
    print(response.text)

# You would call: asyncio.run(run_ingestion())