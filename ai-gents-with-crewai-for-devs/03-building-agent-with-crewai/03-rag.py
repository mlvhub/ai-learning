# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import os

# %%
# %load_ext dotenv
# %dotenv

# %%
# %reload_ext dotenv

openai_api_key = os.getenv("OPENAI_API_KEY")

# %%
from crewai import Agent, Crew, Task, Process
from crewai_tools import WebsiteSearchTool

# %%
search_tool = WebsiteSearchTool(website='https://en.wikipedia.org/wiki/Alan_Turing')

# %%
search_agent = Agent(
    role='Website Researcher',
    goal='Search and extract relevant information from a specific website.',
    verbose=True,
    memory=True,
    backstory='You are an expert in searching websites for the most relevant and up-to-date information.',
    tools=[search_tool]
)

# %%
search_task = Task(
    description=(
        "Use the provided website to find information on the topic '{topic}'. "
        "Make sure to gather all the relevant data available on this site."
    ),
    expected_output='A detailed summary of the information found on the website.',
    tools=[search_tool],
    agent=search_agent,
)

# %%
research_crew = Crew(
    agents=[search_agent],
    tasks=[search_task],
    process=Process.sequential  # Executes tasks one after the other
)

# %%
result = research_crew.kickoff(inputs={'topic': 'Artificial intelligence'})
print(result)
