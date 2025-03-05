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
from supabase import create_client, Client
from openai import OpenAI
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from serpapi import GoogleSearch
from pydantic import BaseModel, Field
import datetime
import re
import os
import sys
import torch

from copy import copy
from functools import lru_cache
from io import StringIO
from typing import Dict, Optional, Any, List, Tuple
import PIL
import matplotlib.pyplot as plt


# %% [markdown]
# ## First Tools

# %%
class ToolInterface(BaseModel):
    name: str
    description: str

    def use(self, input_text: str) -> str:
        raise NotImplementedError("use() method not implemented")  # Must implement in subclass



# %%
class PythonREPLTool(ToolInterface):
    """A tool for running python code in a REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    name: str = "Python REPL"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`. Include examples of using the code and print "
        "the output."
    )

    def _run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        return output.strip()

    def use(self, input_text: str) -> str:
        input_text = input_text.strip().replace("```python" , "")
        input_text = input_text.strip().strip("```")
        return self._run(input_text)


# %%
repl_tool = PythonREPLTool()
result = repl_tool.use('print(1+2)')
print(result)
assert result == "3"


# %%
@lru_cache(maxsize=None)
def search(query: str) -> str:
    params: dict = {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "q": query,
        "api_key": os.getenv("SERP_API_KEY"),
    }

    search = GoogleSearch(params)
    res = search.get_dict()

    return _process_response(res)


def _process_response(res: dict) -> str:
    """Process response from SerpAPI."""
    if "error" in res.keys():
        raise ValueError(f"Got error from SerpAPI: {res['error']}")
    if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
        toret = res["answer_box"]["answer"]
    elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
        toret = res["answer_box"]["snippet"]
    elif (
        "answer_box" in res.keys()
        and "snippet_highlighted_words" in res["answer_box"].keys()
    ):
        toret = res["answer_box"]["snippet_highlighted_words"][0]
    elif (
        "sports_results" in res.keys()
        and "game_spotlight" in res["sports_results"].keys()
    ):
        toret = res["sports_results"]["game_spotlight"]
    elif (
        "knowledge_graph" in res.keys()
        and "description" in res["knowledge_graph"].keys()
    ):
        toret = res["knowledge_graph"]["description"]
    elif "snippet" in res["organic_results"][0].keys():
        toret = res["organic_results"][0]["snippet"]

    else:
        toret = "No good search result found"
    return toret


class SerpAPITool(ToolInterface):
    """Tool for Google search results."""

    name: str = "Google Search"
    description: str = "Get specific information from a search query. Input should be a question like 'How to add number in Clojure?'. Result will be the answer to the question."

    def use(self, input_text: str) -> str:
        return search(input_text)


# %%
serp_tool = SerpAPITool()
serp_tool.use("Who is the current Ravens QB?")

# %% [markdown]
# ## OpenAI LLM

# %%
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_API_KEY')
supabase: Client = create_client(url, key)


# %%
class ChatLLM(BaseModel):
    model: str = 'gpt-4o'
    temperature: float = 0.0

    def generate(self, prompt: str, stop: List[str] = None):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )
        try:
            supabase.table('cost_projecting').insert({
                'prompt': prompt,
                'response': response.choices[0].message.content,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'model': self.model,
                'inference_params' : {
                    'temperature': self.temperature,
                    'stop': stop
                },
                'is_openai': True,
                'app': 'AGENT'
            }).execute()
        except Exception as e:
            print(f"Error inserting data into Supabase: {e}")
        return response.choices[0].message.content


# %%
llm = ChatLLM()
result = llm.generate(prompt='Who is the president of Turkey?')
print(result)

# %%
FINAL_ANSWER_TOKEN = "Assistant Response:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PROMPT_TEMPLATE = """Today is {today} and you can use tools to get new information. Respond to the user's input as best as you can using the following tools:

{tool_description}

You must follow the following format for every single turn of the conversation:

User Input: the input question you must answer
Thought: comment on what you want to do next.
Action: the action to take, exactly one element of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: Now comment on what you want to do next.
Action: the next action to take, exactly one element of [{tool_names}]
Action Input: the input to the next action
Observation: the result of the next action
Thought: Now comment on what you want to do next.
... (this Thought/Action/Action Input/Observation repeats until you are sure of the answer)
Assistant Thought: I have enough information to respond to the user's input.
Assistant Response: your final answer to the original input question
User Input: the input question you must answer
Thought: comment on what you want to do next.
Action: the next action to take, exactly one element of [{tool_names}]
Action Input: the input to the next action
Observation: the result of the next action
... (this Thought/Action/Action Input/Observation repeats until you are sure of the answer)
Assistant Thought: I have enough information to respond to the user's input.
Assistant Response: your final answer to the original input question

You must end every round with "Assistant Thought:" and "Assistant Response:"

Begin:

{previous_responses}
"""


# %%
class Agent(BaseModel):
    llm: ChatLLM
    tools: List
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 5
    # The stop pattern is used, so the LLM does not hallucinate until the end
    stop_pattern: List[str] = [f'\n{OBSERVATION_TOKEN}', f'\n\t{OBSERVATION_TOKEN}']
    human_responses: List[str] = []
    ai_responses: List[str] = []
    pretty_responses: List[str] = []
    verbose: bool = False

    @property
    def tool_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ", ".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, ToolInterface]:
        return {tool.name: tool for tool in self.tools}

    def run(self, question: str):
        self.ai_responses.append(f'User Input: {question}')
        self.human_responses.append(question)
        previous_responses = copy(self.ai_responses)
        num_loops = 0
        prompt = self.prompt_template.format(
                today = datetime.date.today(),
                tool_description=self.tool_description,
                tool_names=self.tool_names,
                question=question,
                previous_responses='{previous_responses}'
        )
        if self.verbose:
            print('------')
            print(prompt.format(previous_responses=''))
            print('------')
        while num_loops < self.max_loops:
            num_loops += 1
            curr_prompt = prompt.format(previous_responses='\n'.join(previous_responses))
            generated, tool, tool_input = self.decide_next_action(curr_prompt)
            if self.verbose:
                print('------')
                print('CURR PROMPT')
                print('------')
                print(curr_prompt)
                print('------')
                print('------')
                print('RAW GENERATED')
                print('------')
                print(generated)
                print('------')
            if tool == 'Assistant Response':
                if self.verbose:
                    print('------')
                    print('FINAL PROMPT')
                    print('------')
                    print(curr_prompt)
                    print('------')
                self.ai_responses.append(f'Assistant Response: {tool_input}')
                return tool_input
            if tool not in self.tool_by_names:
                raise ValueError(f"Unknown tool: {tool}")
            if self.verbose:
                print('tool_input', tool_input)
            tool_result = self.tool_by_names[tool].use(tool_input)
            if type(tool_result) == PIL.Image.Image:
                plt.imshow(tool_result)
                plt.show()
            generated += f"\n{OBSERVATION_TOKEN} {tool_result}\n"
            self.ai_responses.append(generated.strip())
            if self.verbose:
                print('------')
                print('PARSED GENERATED')
                print('------')
                print(generated)
                print('------')
            previous_responses.append(generated)

    def decide_next_action(self, prompt: str) -> str:
        generated = self.llm.generate(prompt, stop=self.stop_pattern)

        tool, tool_input = self._parse(generated)
        return generated, tool, tool_input

    def _parse(self, generated: str) -> Tuple[str, str]:
        if FINAL_ANSWER_TOKEN in generated:
            if self.verbose:
                print('------')
                print('FINAL ANSWER')
                print('------')
                print(generated)
                print('------')
            final_answer = generated.split(FINAL_ANSWER_TOKEN)[-1].strip()
            self.pretty_responses.append(final_answer)
            return "Assistant Response", final_answer
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        if not match:
            raise ValueError(f"Output of LLM is not parsable for next tool use: `{generated}`")
        tool = match.group(1).strip()
        tool_input = match.group(2)
        return tool, tool_input.strip(" ").strip('"')



# %%
agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), SerpAPITool()], verbose=True)
result = agent.run("please write me a function to take in a number and return 2 times it")

print(f"Final answer is {result}")

# %%
for a in agent.ai_responses:
    print(a)
    print('------')

# %%
agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), SerpAPITool()])
result = agent.run("How many people are allowed on the baseball field during play?")

print(f"Final answer is {result}")

for a in agent.ai_responses:
    print(a)
    print('---')

# %% [markdown]
# ## Conversational Tools
#
# Making our agent more efficient by letting it speak without the need of strict tools

# %%
agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), SerpAPITool()])

result = agent.run("What state is San Francisco in?") # probably didn't need to look that up..

for a in agent.ai_responses:
    print(a)
    print('---')


# %%
class SimplyRespond(ToolInterface):  # sometimes referred to as "directly answer"

    name: str = 'Simply Respond'
    description: str = 'Choose this option if the user is giving a pleasantry '
    'or if the answer is very simple knowledge. The action input is nothing.'

    def use(self, input_text: str) -> str:
        return input_text


# %%
class Inquire(ToolInterface):

    name: str = 'Inquire for more Information'
    description: str = 'Choose this option if further information is required '
    'to respond to the user. The action input is a question to ask the user'

    def use(self, input_text: str) -> str:
        return input_text


# %%
agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), SerpAPITool(), SimplyRespond(), Inquire()])

result = agent.run("What state is San Francisco in?") # probably Didn't look anything up

for a in agent.ai_responses:
    print(a)
    print('---')

# %%
agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), SerpAPITool(), SimplyRespond(), Inquire()])

result = agent.run("Hey") # didn't need to look that up..

for a in agent.ai_responses:
    print(a)
    print('---')

# %%
agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), SerpAPITool(), SimplyRespond(), Inquire()])

result = agent.run("What state is this city in?") # inquires for more information

for a in agent.ai_responses:
    print(a)
    print('---')

# %%
agent = Agent(llm=ChatLLM(), tools=[SimplyRespond(), PythonREPLTool(), SerpAPITool(),  Inquire()])
agent.run("What is 1064 + the current price of the cryptocurrency in USD?")  # inquiring

# %%
agent.run("Sorry, Etheruem")

# %%
agent.run("one more thing, what day of the week is tomorrow?")

# %%
agent.run("I forgot to ask, what is the reversed name of the current executive director of common crawl?")

# %%
for h, a in list(zip(agent.human_responses, agent.pretty_responses)):
    print(f"{h} -> \n\t{a}\n")

# %%
for a in agent.ai_responses:
    print(a)
    print('---')

# %% [markdown]
# ## Stock Price Tool

# %%
from typing import Optional, Dict
from pydantic import BaseModel
import alpaca_trade_api as tradeapi

class CheckStockBalance(ToolInterface):
    api_key: str
    api_secret: str
    base_url: str
    api_version: str = 'v2'
    """A tool for checking the current stock wallet balance using Alpaca API."""

    name: str = "Check Stock Balance"
    description: str = (
        "A tool that uses the Alpaca Trade API to retrieve the current wallet balance, "
        "allowing users to check their available cash and stock positions. "
        "The action input to this tool is exactly one of the following commands: "
        "[get_balance]"

    )

    def get_account_balance(self) -> Dict[str, float]:
        """Retrieves the current wallet balance including cash and account value."""
        alpaca_api = tradeapi.REST(
            self.api_key, self.api_secret, self.base_url,
            api_version=self.api_version)
        account = alpaca_api.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value)
        }

    def use(self, command: str) -> str:
        """Run a command to get the account balance."""
        if command == "get_balance":
            balance = self.get_account_balance()
            return f"Cash: ${balance['cash']}, Portfolio Value: ${balance['portfolio_value']}"
        else:
            return "Unknown command. Please use 'get_balance' to check your wallet balance."



# %%
check_stock_balance = CheckStockBalance(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_API_SECRET'),
    base_url='https://paper-api.alpaca.markets'
    )

print(os.getenv('ALPACA_API_KEY'))
print(os.getenv('ALPACA_API_SECRET'))
check_stock_balance.use('get_balance')

# %%
sawyer = Agent(llm=ChatLLM(), tools=[
    check_stock_balance,
    PythonREPLTool(),
    SerpAPITool(),
    SimplyRespond(),
    Inquire()
])


# %%
def chat_with(agent):
# Chat loop
    while True:
        user_input = input("You (exit to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print("Agent:", response)

    # After exiting the loop, print all AI responses
    print("\nAI Responses:")
    for response in agent.ai_responses:
        print(response)


# %%
chat_with(agent := sawyer)

# %% [markdown]
# ## Image Generation Tool
#

# %%
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# %%
# class StableDiffusionTool(ToolInterface):
#     """A tool for generating images"""
#     name: str = "Stable Diffusion"
#     description: str = (
#         "A tool for performing image generation using stable diffusion. "
#         "The action input to this tool is a prompt to create an image "
#         "like 'A photo of a cat on a white background' or 'A surrealist painting of a sunset'."
#     )

#     def use(self, prompt: str) -> str:
#         """Run the stable diffusion tool."""
#         return pipe(prompt).images[0]

# %%
# stable_diffusion_tool = StableDiffusionTool()
# stable_diffusion_tool.use('A photo of a black cat lounging around')

# %%
# sawyer = Agent(llm=ChatLLM(), tools=[
#     PythonREPLTool(),
#     SerpAPITool(),
#     SimplyRespond(),
#     Inquire(),
#     StableDiffusionTool(),
#     check_stock_balance
# ])


# %%
# chat_with(sawyer)

# %% [markdown]
# ## RAG using a lookup API
#

# %%
# from pinecone import Pinecone

# class LookupTool(ToolInterface):
#     """A tool for performing semantic searches using Pinecone and OpenAI embeddings."""

#     # Pinecone setup
#     pinecone_key: str = userdata.get('PINECONE_API_KEY')
#     INDEX_NAME: str = 'semantic-search-rag'
#     ENGINE: str = 'text-embedding-3-small'
#     NAMESPACE: str = 'default'

#     name: str = "Semantic Search Tool"
#     description: str = (
#         "A tool for performing information lookup. Look something up if you "
#         "are being asked about a fact. Even if the retrieved information is "
#         "irrelevant, mention that in the thought and answer the question."
#     )

#     def __init__(self, **data):
#         super().__init__(**data)

#     def get_embeddings(self, texts, engine=ENGINE):
#         response = OpenAI(api_key=userdata.get('OPENAI_API_KEY')).embeddings.create(
#             input=texts,
#             model=engine
#         )
#         return [d.embedding for d in list(response.data)]

#     def get_embedding(self, text, engine=ENGINE):
#         return self.get_embeddings([text], engine)[0]

#     def query_from_pinecone(self, query, top_k=3, include_metadata=True):
#         # get embedding from THE SAME embedder as the documents
#         query_embedding = self.get_embedding(query, engine=self.ENGINE)
#         index = Pinecone(api_key=userdata.get('PINECONE_API_KEY')).Index(name=self.INDEX_NAME)

#         return index.query(
#           vector=query_embedding,
#           top_k=top_k,
#           namespace=self.NAMESPACE,
#           include_metadata=include_metadata   # gets the metadata (dates, text, etc)
#         ).get('matches')[0]['metadata']['text']

#     def use(self, query):
#         return self.query_from_pinecone(query)


# %%
# l = LookupTool()
# l.use('I lost my medicare card')

# %%
# agent = Agent(llm=ChatLLM(), tools=[PythonREPLTool(), LookupTool()])
# result = agent.run("What is the hash of the number I need to call to replace my medicare card")

# print(f"Final answer is {result}")

# %%
# for a in agent.ai_responses:
#     print(a)
#     print('---')

# %% [markdown]
# ## Basic Cost Projection
#
# Looking at the input/output tokens in our data warehouse (Supabase)

# %%
import pandas as pd

response = supabase.table('cost_projecting').select("*").eq('app', 'AGENT').execute()
completions_df = pd.DataFrame(response.data)
completions_df.index = pd.to_datetime(completions_df['created_at'])

completions_df.head()

# %%
