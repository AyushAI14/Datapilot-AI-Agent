from Utils.Prompt import monitor_instruction
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from DataAgent.agent_config import retry_config

from google.adk.runners import InMemoryRunner
from google.adk.apps import App
import os
import asyncio
from google.adk.models.lite_llm import LiteLlm

os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY_PROMPT")
# groq = LiteLlm("meta/llama-3.1-405b-instruct")
ollama = LiteLlm("openai/gpt-oss:120b")
openrouter = LiteLlm("openrouter/meta-llama/llama-3.3-70b-instruct")

os.environ["OPENAI_API_KEY"] = os.getenv("OLLAMA_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OLLAMA_API_BASE")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENROUTER_API_BASE"] = os.getenv("OPENROUTER_API_BASE")

monitoragent = Agent(
    name="monitoringagent",
    model = openrouter,
    instruction=monitor_instruction
)

app = App(name="planner_app", root_agent=monitoragent)

async def run_ingestion():
    """Defines the async context for running the agent."""
    async with InMemoryRunner(app=app) as runner:
        response = await runner.run_debug("")
        # print(response)

if __name__ == "__main__":
    asyncio.run(run_ingestion())
    print("monitoragent created.")