from Utils.Prompt import Scraper_instruction
import os
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.models.google_llm import Gemini
import os
from google.adk.runners import InMemoryRunner
from google.adk.models.lite_llm import LiteLlm
import asyncio
from google.adk.apps import App


os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY_PROMPT")
groq = LiteLlm("groq/openai/gpt-oss-120b")
ollama = LiteLlm("openai/gpt-oss:120b")

os.environ["OPENAI_API_KEY"] = os.getenv("OLLAMA_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OLLAMA_API_BASE")

# from DataAgent.custom_tool import save_to_raw
# from DataAgent.agent_config import retry_config


# ---------------- FIRECRAWL MCP SERVER ----------------
mcp_firecrawl_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env={
                "FIRECRAWL_API_KEY": os.environ.get("FIRECRAWL_API_KEY", os.getenv('FIRECRAWL_API_KEY'))
            },
        ),
        timeout=60,
    ),
    tool_filter=["firecrawl_scrape"] 
)

# ---------------- FIRECRAWL SCRAPER AGENT ----------------
scraper_agent = Agent(
    name="FirecrawlWebScraper",
    model = ollama,
    # model=Gemini(
    #     model="gemini-2.5-flash-lite"
    # ),
    instruction=Scraper_instruction,
    tools=[mcp_firecrawl_server],
    output_key="page_content"
)



app = App(name="scraper_app", root_agent=scraper_agent)

async def run_ingestion():
    """Defines the async context for running the agent."""
    async with InMemoryRunner(app=app) as runner:
        response = await runner.run_debug("")
        print(response)

if __name__ == "__main__":
    asyncio.run(run_ingestion())
    print("scraper_agent created.")
