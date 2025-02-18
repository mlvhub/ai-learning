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

serper_api_key = os.getenv("SERPER_API_KEY")

# %%
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel
from crewai import Agent, Crew, Task

# %%
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# %%
market_researcher = Agent(
    role="Market Researcher",
    goal="Conduct thorough market research to identify target demographics and competitors.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Analytical and detail-oriented, you excel at gathering insights about the market, "
        "analyzing competitors, and identifying the best strategies to target the desired audience."
    )
)

# %%
content_creator = Agent(
    role='Content Creator',
    goal="Develop engaging content for the product launch, including blogs, social media posts, and videos.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and persuasive, you craft content that resonates with the audience, "
        "driving engagement and excitement for the product launch."
    )
)

# %%
pr_outreach_specialist = Agent(
    role="PR and Outreach Specialist",
    goal="Reach out to influencers, media outlets, and key opinion leaders to promote the product launch.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "With strong networking skills, you connect with influencers and media outlets to ensure "
        "the product launch gains maximum visibility and coverage."
    )
)


# %%
class MarketResearchData(BaseModel):
    target_demographics: str
    competitor_analysis: str
    key_findings: str


# %%
market_research_task = Task(
    description="Conduct market research for the {product_name} launch, focusing on target demographics and competitors.",
    expected_output="A detailed report on market research findings, including target demographics and competitor analysis.",
    human_input=True,
    output_json=MarketResearchData,
    output_file="market_research.json",
    agent=market_researcher
)

# %%
content_creation_task = Task(
    description="Create content for the {product_name} launch, including blog posts, social media updates, and promotional videos.",
    expected_output="A collection of content pieces ready for publication.",
    human_input=True,
    async_execution=False,  # Change to synchronous
    output_file="content_plan.txt",
    agent=content_creator
)

# %%
pr_outreach_task = Task(
    description="Contact influencers, media outlets, and key opinion leaders to promote the {product_name} launch.",
    expected_output="A report on outreach efforts, including responses from influencers and media coverage.",
    async_execution=False,  # Change to synchronous
    output_file="outreach_report.md",
    agent=pr_outreach_specialist
)

# %%
product_launch_crew = Crew(
    agents=[market_researcher, content_creator, pr_outreach_specialist],
    tasks=[market_research_task, content_creation_task, pr_outreach_task],  # Ensure only one async task is at the end
    verbose=True
)

# %%
launch_details = {
    'product_name': "SmartHome 360",
    'product_description': "A cutting-edge smart home system that integrates with all your devices.",
    'launch_date': "2024-10-01",
    'target_market': "Tech-savvy homeowners",
    'budget': 50000
}

result = product_launch_crew.kickoff(inputs=launch_details)


# %%
import json 
from pprint import pprint  
from IPython.display import Markdown  # Add this line to import the Markdown function

# Display the generated market_research.json file
with open('market_research.json') as f:
   data = json.load(f)
pprint(data)

# Display the generated content_plan.txt file
with open('content_plan.txt') as f:
    content = f.read()
print(content)

# Display the generated outreach_report.md file
Markdown("outreach_report.md")
