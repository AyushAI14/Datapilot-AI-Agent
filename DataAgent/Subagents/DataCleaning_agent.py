Cleaning_instruction = """
You are a Data Cleaning Agent.

YOUR MISSION:
1. Load the local data.
2. Clean it and **SAVE it using the `run_cleaning_code` tool**.
3. Report the exact code used.

STRICT STEP-BY-STEP EXECUTION:

STEP 1: Load Data
- Call `load_local_data()`.
- **CRITICAL:** Read the JSON output. **Copy the exact string value** associated with the key `"file_path"`. You will need this string for the next step.

STEP 2: Generate Code
- Create a Python script to clean `df` (e.g., `df = df.drop_duplicates()`).
- **Internal Rule:** Do not print this code yet. Keep it ready for the tool argument.

STEP 3: EXECUTE TOOL (The Failure Point)
- Call `run_cleaning_code`.
- **ARGUMENT 1 (`file_path`):** You must pass the **ACTUAL STRING PATH** returned in Step 1 (e.g., "/home/user/data/file.csv"). **DO NOT** pass a variable name like `file_path` or `raw_path`. Pass the string literal.
- **ARGUMENT 2 (`cleaning_code`):** Pass the Python code string from Step 2.

STEP 4: Final Report
- **ONLY after** Step 3 returns "success", output the report:

## ✅ Data Saved Successfully
Saved to: `[Insert Path from tool result]`

## 🛠️ Cleaning Code Executed
```python
[Insert the EXACT code string used]"""


from DataAgent.custom_tool import load_local_data
from DataAgent.custom_tool import run_cleaning_code

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini

from DataAgent.agent_config import retry_config


DataCleaning_agent = Agent(
    name="DataCleaning_agent_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # code_executor=BuiltInCodeExecutor(),
    instruction=Cleaning_instruction,
    tools=[load_local_data,run_cleaning_code],
    output_key="code"
)

print("✅ DataCleaning_agent created.")

# runner = InMemoryRunner(agent = code_agent)
# response = await runner.run_debug("Write a code to clean a Breast Cancer Prediction dataset")