from .Subagents.Ingest_agent import ingest_agent
from .Subagents.Planner_agent import Planner_agent
from .Subagents.WebScrapper import scraper_agent
from .Subagents.DataCleaning_agent import DataCleaning_agent
from google.adk.agents import SequentialAgent,LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import InMemoryRunner
import asyncio


root_agent = SequentialAgent(
    name="DataSciencePipeline",
    sub_agents=[ingest_agent,scraper_agent, Planner_agent,DataCleaning_agent]
)


print("Sequential Agent created.")
# runner = InMemoryRunner(agent=root_agent)


# async def main():
#     response = await runner.run_debug("""
# Find the small Breast Cancer Prediction dataset , download it and make a markdown of the dataset page url and make a plan to create a classification model and Write a code to clean and save the dataset .""")
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())

# root_agent = LlmAgent(
#     name="DataPilot", 
#     instruction="""You are **DataPilot**, the **Master Data Science Workflow Agent and Pipeline Orchestrator**. 
    
#     **Identity:** If the user asks for your name, state clearly: "My name is DataPilot, and I am the Master Data Science Workflow Agent."
    
#     **Primary Role & Execution Order:** Your sole purpose is to receive the user's request and manage the entire data science workflow by calling your sub-agents as tools in the following strict, sequential order:
    
#     1.  **ingest_agent**
#     2.  **scraper_agent**
#     3.  **Planner_agent**
#     4.  **DataCleaning_agent**
    
#     You must ensure the output and context from one tool call is correctly and seamlessly passed as the input to the next relevant tool call to drive the task toward completion. **Do not deviate from this prescribed sequence.**
#     """, 
#     model="gemini-2.5-pro", 
#     tools=[
#         AgentTool(ingest_agent),
#         AgentTool(scraper_agent),
#         AgentTool(Planner_agent),
#         AgentTool(DataCleaning_agent),
#     ],
#     )