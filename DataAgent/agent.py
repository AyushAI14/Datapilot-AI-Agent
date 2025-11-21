from .Subagents.Ingest_agent import ingest_agent
from .Subagents.Planner_agent import Planner_agent
from .Subagents.WebScrapper import scraper_agent
from .Subagents.DataCleaning_agent import DataCleaning_agent
from google.adk.agents import SequentialAgent

root_agent = SequentialAgent(
    name="DataSciencePipeline",
    sub_agents=[ingest_agent,scraper_agent, Planner_agent,DataCleaning_agent],
)

print("✅ Sequential Agent created.")
# runner = InMemoryRunner(agent=Orchestrator_agent)
# response = await runner.run_debug("""
# Find the small Breast Cancer Prediction dataset , download it and make a markdown of the dataset page url and make a plan to create a classification model and Write a code to clean and save the dataset .""")