import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = "gpt-3.5-turbo"

client = OpenAI(api_key=openai_api_key)
model = ChatOpenAI(api_key=openai_api_key, model=llm, temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def bot(state: State):
    print(state["messages"])
    return {"messages": [model.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("bot", bot)

graph_builder.set_entry_point("bot")
graph_builder.set_finish_point("bot")

graph = graph_builder.compile()

# res = graph.invoke({"messages": [{"role": "user", "content": "Hello, how are you? "}]})
# print(res)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting...")
        break
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)
