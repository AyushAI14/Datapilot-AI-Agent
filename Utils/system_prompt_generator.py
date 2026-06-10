from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

import os
import asyncio
import warnings
from Utils.Prompt import prompt_constructor
from google.adk.models.lite_llm import LiteLlm
os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY_PROMPT")
groq = LiteLlm("groq/openai/gpt-oss-120b")
warnings.filterwarnings("ignore")

load_dotenv()



console = Console()



inputtext = input("\nEnter Your Agent Subject : ")


PromptAgent = Agent(
    model=groq,
    name="PromptAgent",
    instruction=prompt_constructor,
)


async def run_ingestion():
    runner = InMemoryRunner(agent=PromptAgent)
    response = await runner.run_debug(inputtext)
    collected_text = []
    for event in response:
        # ADK event parsing
        if hasattr(event, "content") and event.content:
            parts = getattr(event.content, "parts", [])
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    collected_text.append(text)

    latest_output = "\n".join(collected_text)

    console.print(
        Panel(
            Markdown(latest_output),
            title="[bold cyan]Generated System Prompt[/bold cyan]",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    asyncio.run(run_ingestion())