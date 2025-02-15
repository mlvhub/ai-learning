import re


class Agent:
    def __init__(self, client, system=None, llm_name="gpt-3.5-turbo"):
        self.client = client
        self.system = system
        self.llm_name = llm_name

        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()

        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            # don't be creative, give me a straight answer
            temperature=0.0,
            messages=self.messages,
        )
        return response.choices[0].message.content


def interactive_agent(client, prompt, llm_name):
    max_turns = int(input("Enter the maximum number of turns: "))

    bot = Agent(client, system=prompt, llm_name=llm_name)

    action_re = re.compile(r"^Action: (\w+): (.*)$")

    i = 0

    while i < max_turns:
        next_prompt = input("Enter your prompt: ")
        i += 1
        response = bot(next_prompt)
        print(f"Bot: {response}")

        actions = [
            action_re.match(a) for a in response.split("\n") if action_re.match(a)
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}")
            print(f"-- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print(f"Observation: {observation}")
            next_prompt = "Observation: {}".format(observation)
        else:
            print("No actions found")
            return


def auto_agent(client, prompt, llm_name, question, max_turns=10):
    bot = Agent(client, system=prompt, llm_name=llm_name)

    action_re = re.compile(r"^Action: (\w+): (.*)$")

    i = 0

    next_prompt = question

    while i < max_turns:
        i += 1
        response = bot(next_prompt)
        print(response)

        actions = [
            action_re.match(a) for a in response.split("\n") if action_re.match(a)
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}")
            print(f"-- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print(f"Observation: {observation}")
            next_prompt = "Observation: {}".format(observation)
        else:
            print("No actions found")
            return


def calculate(expression: str) -> float:
    return eval(expression)


def planet_mass(planet: str) -> float:
    masses = {
        "Mercury": 0.33011,
        "Venus": 4.8675,
        "Earth": 5.972,
        "Mars": 0.64171,
        "Jupiter": 1898.6,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.43,
    }
    return f"{planet} has a mass of {masses[planet]} × 10^24 kg"


known_actions = {
    "calculate": calculate,
    "planet_mass": planet_mass,
}
