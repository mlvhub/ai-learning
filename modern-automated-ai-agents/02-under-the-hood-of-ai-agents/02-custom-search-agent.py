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

# %%
import json
import os
import requests

from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from serpapi import GoogleSearch
from crewai.tools import BaseTool


# %%
class SearchTool(BaseTool):
    name: str = "Google Search"
    description: str = """Useful to search the internet about a given topic and return relevant results."""

    def _run(self, query: str, top_k: int = 3) -> str:
        params: dict = {
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "q": query,
            "api_key": os.environ["SERP_API_KEY"],
        }

        search = GoogleSearch(params)
        response = search.get_dict()
        # check if there is an organic results key, don't include sponsor results
        if 'organic_results' not in response:
            return "Sorry, I couldn't find anything about that, there could be an error with you serper api key."
        else:
            results = response['organic_results']
            string = []
            for result in results[:top_k]:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}", f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}", "\n-----------------"
                    ]))
                except KeyError:
                    next

            return '\n'.join(string)


# %%
SearchTool().description # where I specifically describe how to use the tool

# %%
print(SearchTool().run(query="Who is Sinan Ozdemir?"))

# %%
researcher_agent = Agent(
    role='Researcher',
    goal='Look up information on the internet and scrape the websites for relevant information',
    backstory="""You have a knack for dissecting complex data and presenting actionable insights.""",
    tools=[
        SearchTool()
    ],
    verbose=True
)

# %%
email_agent = Agent(
    role='Emailer',
    goal='Write short, warm emails to people',
    backstory="""You are an assistant.""",
    tools=[],
    verbose=True
)

# %%
sample_lookup_task = Task(
  description="""Lookup Sinan Ozdemir""",
  expected_output="A lookup report of a query",
  agent=researcher_agent
)

# %%
email_task = Task(
  description="""Write an email to Sinan Ozdemir asking him to get coffee. Personalize it""",
  expected_output="A short warm email",
  agent=email_agent
)


# %%

crew = Crew(
    tasks=[sample_lookup_task, email_task],
    process=Process.sequential,
    verbose=True
)

# %%
result = crew.kickoff()
result

# %%
print(result)

# %%
