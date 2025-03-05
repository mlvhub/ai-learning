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
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool

# %%
scrape_tool = ScrapeWebsiteTool(website_url='https://en.wikipedia.org/wiki/Artificial_intelligence')


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
class TripAgents():

  def local_expert(self):
    return Agent(
        role='Local Expert at this city',
        goal='Provide 2-3 insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            SearchTool(),
            scrape_tool
        ],
        verbose=True)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with
        decades of experience""",
        tools=[],
        verbose=True)


# %%
from crewai import Task
from textwrap import dedent
from datetime import date


class TripTasks():

  def gather_task(self, agent, origin, city, interests, range):
    return Task(description= dedent(f"""
        I want you to create a comprehensive city guide.
        This guide should provide a thorough overview of what
        the city has to offer, including hidden gems, cultural
        hotspots, and must-visit landmarks.

        Trip Date: {range}
        Traveling from: {origin}
        Traveling to: {city}
        Traveler Interests: {interests}
        """),

            expected_output=dedent(f"""
        The final answer must be a comprehensive city guide,
        rich in cultural insights and practical tips,
        tailored to enhance the travel experience.

        """),
                agent=agent)

  def plan_task(self, agent, origin, city, interests, range):
    return Task(description=dedent(f"""
        Expand this guide into a a full 1-day travel
        itinerary with detailed per-day plans, including
        weather forecasts, places to eat, packing suggestions,
        and a budget breakdown.

        You MUST suggest actual places to visit, actual hotels
        to stay and actual restaurants to go to.

        Trip Date: {range}
        Traveling from: {origin}
        Traveling to: {city}
        Traveler Interests: {interests}
      """),
            expected_output=dedent(f"""
        Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule,
        anticipated weather conditions, recommended clothing and
        items to pack, and a detailed budget, ensuring THE BEST
        TRIP EVER, Be specific and give it a reason why you picked
        # up each place, what make them special!
        """),
                agent=agent)



# %%


class TripCrew:

  def __init__(self, origin, city, date_range, interests):
    self.city = city
    self.origin = origin
    self.interests = interests
    self.date_range = date_range

  def run(self):
    agents = TripAgents()
    tasks = TripTasks()

    gather_task = tasks.gather_task(
      agents.local_expert(),
      self.origin,
      self.city,
      self.interests,
      self.date_range
    )
    plan_task = tasks.plan_task(
      agents.travel_concierge(),
      self.origin,
      self.city,
      self.interests,
      self.date_range
    )

    # Initiate Crew
    crew = Crew(
      tasks=[gather_task, plan_task],
      process=Process.sequential,
      verbose=True
    )

    result = crew.kickoff()
    return result


# %%
trip_crew = TripCrew('SF', "Paris", 'June 2025', 'art')
result = trip_crew.run()

# %%
