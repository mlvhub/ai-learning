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

# %% [markdown]
# # Reflection Agents
#
# Inspiration: https://blog.langchain.dev/reflection-agents/
#
# Reflection Agents consists of two basic components:
# - A generator, which is really any agent able to accomplish a task, with or without tools.
# - A reflector, which offers criticism and feedback to the generator to have it re-try with advice.
#

# %%
# %load_ext dotenv
# %dotenv

# %%
# %reload_ext dotenv

# %%
import os
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

# %% [markdown]
# ## A simple Agent - Writing Linkedin Posts
#

# %%
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write engaging Linkedin Posts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

generate = prompt | llm

# %%
request = "Write one about me completing day one of filming my AI agent course"
first_draft_response = generate.invoke({"messages": [("user", request)]})
first_draft_response

# %% [markdown]
# ## Reflecting on our agent's work
#

# %%
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a marketing strategist looking over social media posts. Generate critique and recommendations for the user's post."
            " Provide detailed recommendations on things including length, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

# %%
reflection = reflect.invoke({"messages": [("user", request), ("user", first_draft_response.content)]})
reflection

# %%
print(reflection.content)

# %% [markdown]
# ## Turning this loop into a graph

# %%
from typing import Annotated, List, Sequence
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# %%
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def generation_node(state: State) -> State:
    return {"messages": [generate.invoke(state["messages"])]}


def reflection_node(state: State) -> State:
    # Other messages we need to adjust
    # First message is the original user request. We hold it the same for all nodes and use the most recent iteration
    translated = [state["messages"][0], HumanMessage(content=state["messages"][-1].content)]
    res = reflect.invoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")


# %%
def should_continue(state: State):
    if len(state["messages"]) >= 6:
        # End after 2 reflections
        return END
    return "reflect"


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()

# %%
# Compile the graph
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
drafts = []
for event in graph.stream({"messages": [("user", request)]}):
    for node_name, output in event.items():
        print(f'At Node: {node_name}. Output: {output["messages"][-1].content}')
        if node_name == 'generate':
            drafts.append(output["messages"][-1])
        print('----')

# %%
for i, draft in enumerate(drafts):
    print(f'Draft {i}')
    print('-----------------------------------------------------------')
    print(draft.content)
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')

# %%
# Listen for state updates
for event in graph.stream({"messages": [("user", request)]}, stream_mode=["messages", "values"]):
    if event[0] == 'values': 
        print(event[1].keys(), len(event[1]['messages']))
        print(event[1]['messages'][-1].content)

# %%
