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
# %load_ext dotenv
# %dotenv

# %%
# %reload_ext dotenv

# %% [markdown]
# ## Our First Agent in CrewAI

# %%
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, FileWriterTool, TXTSearchTool
import requests
import os

# %%
# Initialize the tool, potentially passing the session
scrape_tool = ScrapeWebsiteTool(website_url='https://en.wikipedia.org/wiki/Artificial_intelligence')

# Extract the text
text = scrape_tool.run()
print(text[:100])

# %%
# Initialize the tool
file_writer_tool = FileWriterTool()

# Write content to a file in a specified directory
result = file_writer_tool.run(filename='ai.txt', content = text, directory = '', overwrite='true')
print(result)


# %%
# Initialize the tool with a specific text file, so the agent can search within the given text file's content
# uses chromadb to chunk and vectorize data
search_tool = TXTSearchTool(txt='ai.txt')

# %%
data_analyst = Agent(
    role='Educator',
    goal=f'Based on the context provided, answer the question.',
    backstory='You are a data expert',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

# %%
test_task = Task(
    description="Understand the topic of Natural Language Processing and summarize it for me",
    agent=data_analyst,
    expected_output='I want the response to be as short as possible.'
)

# %%
crew = Crew(
    tasks=[test_task],
    process=Process.sequential,
)

# %%
output = crew.kickoff()

# %%
print(output.raw)

# %% [markdown]
# ## Our Second Agent in CrewAI
#

# %%
from langchain_openai import ChatOpenAI

# %%
nlp_task = Task(
    description="Understand the topic of Natural Language Processing and sumamrize it for me",
    # agent=data_analyst,  # commenting out the agent now, letting the manager decide
    expected_output='Give a correct response'
)

# %%
calculator_agent = Agent(
    role='Calculator',
    goal=f'You calculate things',
    backstory='You love math',
    verbose=True,
    allow_delegation=False,
    tools=[]
)


# %%
math_task = Task(
    description="Tell me what 123 * 34 is",
    expected_output='Calculate this'
)

# %%
crew = Crew(
    tasks=[math_task, nlp_task],
    agents=[calculator_agent, data_analyst],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7),  # required if you are using hierarhical
    verbose=True
)


# %%
output = crew.kickoff()
