import os
from io import StringIO
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from openai import OpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = "gpt-3.5-turbo"

model = ChatOpenAI(api_key=openai_api_key, model=llm, temperature=0)

tavily_client = TavilyClient(api_key=tavily_api_key)


class AgentState(TypedDict):
    task: str
    competitors: list[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: list[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: list[str] = Field(description="A list of queries to search the web for")


# Prompts for each agent/node. Improve as needed.
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


def gather_financials_agent(state: AgentState):
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    financial_data = df.to_string(index=False)
    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}


def analyze_data_agent(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]

    response = model.invoke(messages)
    return {"analysis": response.content}


def research_competitors_agent(state: AgentState):
    content = state.get("content", [])

    for competitor in state["competitors"]:
        messages = [
            SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
            HumanMessage(content=competitor),
        ]

        queries = model.with_structured_output(Queries).invoke(messages)
        for q in queries.queries:
            response = tavily_client.search(query=q, max_results=3)
            for r in response["results"]:
                content.append(r["content"])

    return {"content": content}


def compare_performance_agent(state: AgentState):
    content = "\n\n".join(state.get("content", []))
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        HumanMessage(
            content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
        ),
    ]

    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def research_critique_agent(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state["feedback"]),
    ]
    queries = model.with_structured_output(Queries).invoke(messages)

    content = state.get("content", [])
    for q in queries.queries:
        response = tavily_client.search(query=q, max_results=3)
        for r in response["results"]:
            content.append(r["content"])

    return {"content": content}


def feedback_agent(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]

    response = model.invoke(messages)
    return {"feedback": response.content}


def write_report_agent(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]

    response = model.invoke(messages)
    return {"report": response.content}


def should_continue(state: AgentState):
    if state["revision_number"] >= state["max_revisions"]:
        return END
    return "collect_feedback"


builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_agent)
builder.add_node("analyze_data", analyze_data_agent)
builder.add_node("research_competitors", research_competitors_agent)
builder.add_node("compare_performance", compare_performance_agent)
builder.add_node("research_critique", research_critique_agent)
builder.add_node("collect_feedback", feedback_agent)
builder.add_node("write_report", write_report_agent)

builder.set_entry_point("gather_financials")
builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {
        END: END,
        "collect_feedback": "collect_feedback",
    },
)

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

# Streamlit app

import streamlit as st


def main():
    checkpointer_context = SqliteSaver.from_conn_string(":memory:")
    with checkpointer_context as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        st.title("Financial Performance Reporting Agent")

        task = st.text_input(
            "Enter the task:",
            "Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
        )
        competitors = st.text_area("Enter competitor names (one per line):").split("\n")
        max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
        uploaded_file = st.file_uploader(
            "Upload a CSV file with the company's financial data", type=["csv"]
        )

        if st.button("Start Analysis") and uploaded_file is not None:
            # Read the uploaded CSV file
            csv_data = uploaded_file.getvalue().decode("utf-8")

            initial_state = {
                "task": task,
                "competitors": [comp.strip() for comp in competitors if comp.strip()],
                "csv_file": csv_data,
                "max_revisions": max_revisions,
                "revision_number": 1,
            }
            thread = {"configurable": {"thread_id": "1"}}

            final_state = None
            for s in graph.stream(initial_state, thread):
                st.write(s)
                final_state = s

            if final_state and "report" in final_state:
                st.subheader("Final Report")
                st.write(final_state["report"])


if __name__ == "__main__":
    main()

# CLI testing

# def read_csv(csv_file: str):
#     with open(csv_file, "r") as f:
#         return f.read()


# if __name__ == "__main__":
#     checkpointer_context = SqliteSaver.from_conn_string(":memory:")
#     with checkpointer_context as checkpointer:
#         graph = builder.compile(checkpointer=checkpointer)
#         task = "Analyze the financial data for Apple Inc."
#         competitors = ["Microsoft", "Google", "Amazon", "Nvidia"]
#         csv_file = "./05-financial-report-writer-agent/data/financial_data.csv"

#         if not os.path.exists(csv_file):
#             print(f"File {csv_file} not found")
#         else:
#             print("Starting the financial report writer agent...")
#             csv_data = read_csv(csv_file)

#             initial_state = {
#                 "task": task,
#                 "competitors": competitors,
#                 "csv_file": csv_data,
#                 "max_revisions": 3,
#                 "revision_number": 1,
#             }
#             thread = {"configurable": {"thread_id": 1}}

#             for s in graph.stream(initial_state, config=thread):
#                 print(s)
