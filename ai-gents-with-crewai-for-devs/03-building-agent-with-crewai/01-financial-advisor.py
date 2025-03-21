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
budgeting_agent = Agent(
    role="Budgeting Advisor",
    goal="Create a monthly budget to help users manage their income and expenses effectively.",
    backstory=(
        "You are an experienced financial advisor specializing in personal finance. "
        "Your goal is to help users allocate their income efficiently, ensuring they cover "
        "all necessary expenses while also saving for the future."
    ),
    allow_delegation=False,
    verbose=True
)

# %%
investment_agent = Agent(
    role="Investment Advisor",
    goal="Recommend suitable investment options based on the user's financial goals and risk tolerance.",
    backstory=(
        "You are an investment expert with years of experience in the financial markets. "
        "Your goal is to help users grow their wealth by providing sound investment advice "
        "tailored to their risk tolerance and financial objectives."
    ),
    allow_delegation=False,
    verbose=True
)

# %%
debt_management_agent = Agent(
    role="Debt Management Specialist",
    goal="Help users manage and reduce their debt through effective strategies.",
    backstory=(
        "You specialize in helping individuals overcome debt by creating personalized repayment plans. "
        "Your focus is on reducing interest payments and improving the user's financial health."
    ),
    allow_delegation=False,
    verbose=True
)

# %%
budgeting_task = Task(
    description=(
        "1. Analyze the user's income and expenses.\n"
        "2. Create a detailed monthly budget that includes essential expenses, savings, and discretionary spending.\n"
        "3. Provide tips for optimizing spending and increasing savings."
    ),
    expected_output="A comprehensive monthly budget with recommendations for optimizing spending and savings.",
    agent=budgeting_agent
)

# %%
investment_task = Task(
    description=(
        "1. Assess the user's financial goals and risk tolerance.\n"
        "2. Recommend suitable investment options such as stocks, bonds, mutual funds, or ETFs.\n"
        "3. Provide a brief overview of each recommended investment's potential risks and returns."
    ),
    expected_output="A personalized investment plan with recommendations and risk assessments.",
    agent=investment_agent
)

# %%
debt_management_task = Task(
    description=(
        "1. Analyze the user's current debts, including interest rates and balances.\n"
        "2. Develop a repayment plan that prioritizes high-interest debt and suggests strategies for paying off balances faster.\n"
        "3. Provide advice on consolidating debt or negotiating lower interest rates."
    ),
    expected_output="A debt management plan with actionable steps to reduce and eliminate debt.",
    agent=debt_management_agent
)

# %%
crew = Crew(
    agents=[budgeting_agent, investment_agent, debt_management_agent],
    tasks=[budgeting_task, investment_task, debt_management_task],
    verbose=True  # Set to True for detailed logging or False to reduce output
)

# %%
user_financial_data = {
    "income": 5000,  # Monthly income in dollars
    "expenses": {
        "rent": 1500,
        "utilities": 300,
        "groceries": 400,
        "transportation": 200,
        "entertainment": 150,
        "other": 450
    },
    "debts": {
        "credit_card": {
            "balance": 2000,
            "interest_rate": 0.18  # 18% interest rate
        },
        "student_loan": {
            "balance": 15000,
            "interest_rate": 0.045  # 4.5% interest rate
        }
    },
    "savings_goal": 500  # Monthly savings goal in dollars
}

# Now run the crew kickoff with the defined data
result = crew.kickoff(inputs=user_financial_data)

# Extract the raw text from the result
raw_result = result.raw

# Display the result as markdown
from IPython.display import Markdown
Markdown(raw_result)

