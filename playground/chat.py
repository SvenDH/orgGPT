from langchain.globals import set_llm_cache
from langchain.chat_models.base import BaseChatModel
from langchain.cache import InMemoryCache

from playground.agents import Agent
from playground.utils import content_str

set_llm_cache(InMemoryCache())


class GroupChat:
    def __init__(self, model: BaseChatModel, agents: list[Agent] = []) -> None:
        self.model = model
        self.agents = agents
        self._step = 0
        self.messages = []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def reset(self):
        self.messages = []
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self, agent: Agent) -> tuple[str, str]:
        message = agent.send()
        for receiver in self.agents:
            receiver.receive(agent.name, message)
        self._step += 1
        return agent.name, message
    
    def append(self, message: dict, agent: Agent):
        if message["role"] != "function":
            message["name"] = agent.name
        message["content"] = content_str(message["content"])
        self.messages.append(message)
