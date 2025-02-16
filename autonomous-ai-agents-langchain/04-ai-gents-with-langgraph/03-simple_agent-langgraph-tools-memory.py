import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = "gpt-3.5-turbo"

client = OpenAI(api_key=openai_api_key)
model = ChatOpenAI(api_key=openai_api_key, model=llm, temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]


tool = TavilySearchResults(api_key=tavily_api_key, max_results=2)
tools = [tool]
# res = tool.invoke("What's the capital of France?")
# print(res)


model_with_tools = model.bind_tools(tools)
# res = model_with_tools.invoke("What's a node in LangGraph")
# print(res)


def bot(state: State):
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)

tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

graph_builder.add_node("bot", bot)

graph_builder.set_entry_point("bot")

# checkpointer to save the state of the graph adding "memory" to the graph
checkpointer_context = SqliteSaver.from_conn_string(":memory:")

with checkpointer_context as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)
    # Without the checkpointer, the graph will not know my amazing name
    # graph = graph_builder.compile()

    config = {
        "configurable": {"thread_id": 1},
    }
    user_input = "My name is Slim Shady"

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    )

    for event in events:
        event["messages"][-1].pretty_print()

    user_input = "What's my name?"

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    )

    for event in events:
        event["messages"][-1].pretty_print()

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["exit", "quit", "q"]:
#         print("Exiting...")
#         break
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
#         for value in event.values():
#             print("Assistant: ", value["messages"][-1].content)
