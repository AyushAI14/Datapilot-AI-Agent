from Utils.Prompt import Cleaning_instruction
from DataAgent.custom_tool import load_local_data, run_cleaning_code
from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.google_llm import Gemini
from DataAgent.agent_config import retry_config
import os
import asyncio
from google.adk.runners import InMemoryRunner
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm

os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY_PROMPT")
groq = LiteLlm("groq/qwen/qwen3-32b")
ollama = LiteLlm(
    model="openai/glm-4.7",
    api_base="https://ollama.com/v1",
    api_key=os.getenv("OLLAMA_API_KEY")
)

os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API_KEY")
os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE")

DataCleaning_agent = Agent(
    name="DataCleaning_agent_agent",
    model=ollama,
    #model=Gemini(
    #    model="gemini-3.1-flash-lite",
    #    retry_options=retry_config
    #),
    # code_executor=BuiltInCodeExecutor(),
    instruction=Cleaning_instruction,
    tools=[load_local_data, run_cleaning_code],
    output_key="code"
)


app = App(name="cleaning_app", root_agent=DataCleaning_agent)

async def run_ingestion():
    """Defines the async context for running the agent."""
    async with InMemoryRunner(app=app) as runner:
        response = await runner.run_debug("")
        # print(response)

if __name__ == "__main__":
    asyncio.run(run_ingestion())
    print("DataCleaning_agent created.")
