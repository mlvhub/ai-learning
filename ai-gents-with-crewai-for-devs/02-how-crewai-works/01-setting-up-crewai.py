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
from crewai import Agent, Crew, Task

# %%
designer = Agent(
    role="Feature Designer",
    goal="Design a user-friendly and technically feasible feature on {feature_name}",
    backstory="You're responsible for designing a new feature called {feature_name}."
    "Your goal is to create a feature that addresses user needs "
    "while considering the technical constraints. "
    "Your work will serve as the blueprint for "
    "the Developer to implement this feature.",
    allow_delegation=False,
    verbose=True,
)

# %%
developer = Agent(
    role="Developer",
    goal="Implement the feature {feature_name} based on the design provided.",
    backstory="You're responsible for coding and implementing the new feature: {feature_name}. "
    "You base your work on the design and specifications provided by "
    "the Feature Designer. "
    "Your goal is to ensure the feature works as intended, "
    "is efficient, and follows best practices in coding.",
    allow_delegation=False,
    verbose=True,
)

# %%
tester = Agent(
    role="Quality Assurance Tester",
    goal="Test the feature {feature_name} to ensure it functions correctly and meets requirements.",
    backstory="You are a QA Tester who receives the implemented feature {feature_name} "
    "from the Developer. "
    "Your goal is to test the feature thoroughly, "
    "identify any bugs, and ensure it meets the design and functionality requirements.",
    allow_delegation=False,
    verbose=True,
)

# %%
design_task = Task(
    description=(
        "1. Analyze user needs and gather requirements for {feature_name}.\n"
        "2. Create wireframes and a detailed design document.\n"
        "3. Ensure the design is user-friendly and technically feasible.\n"
        "4. Include any necessary technical specifications and user flow diagrams."
    ),
    expected_output="A comprehensive design document with wireframes, "
    "user flow diagrams, and technical specifications.",
    agent=designer,
)

# %%
develop_task = Task(
    description=(
        "1. Use the design document to code and implement the feature {feature_name}.\n"
        "2. Ensure the code follows best practices and is efficient.\n"
        "3. Write unit tests and documentation for the feature.\n"
        "4. Conduct a basic test to ensure the feature works as expected."
    ),
    expected_output="A fully implemented feature {feature_name} "
    "with associated unit tests and documentation.",
    agent=developer,
)

# %%
test_task = Task(
    description=(
        "1. Test the feature {feature_name} thoroughly to ensure it meets the design requirements.\n"
        "2. Identify and document any bugs or issues.\n"
        "3. Verify the feature's performance under different scenarios.\n"
        "4. Provide feedback on any improvements or changes needed."
    ),
    expected_output="A detailed test report with identified issues, "
    "recommendations for improvement, and confirmation that the feature meets the requirements.",
    agent=tester,
)

# %%
crew = Crew(
    agents=[designer, developer, tester],
    tasks=[design_task, develop_task, test_task],
    verbose=True,
)

# %%
result = crew.kickoff(inputs={"feature_name": "Smart Notifications"})
