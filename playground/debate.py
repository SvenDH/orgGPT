import numpy as np
import tenacity
from langchain.globals import set_llm_cache
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import RegexParser
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from langchain_community.llms import LlamaCpp
from langchain_experimental.chat_models import Llama2Chat
from langchain.cache import InMemoryCache

from playground.agents import BiddingAgent

set_llm_cache(InMemoryCache())

MODEL_PATH = "C:/Users/denha/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"


llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=32,
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
    max_tokens=512
)

model = Llama2Chat(
    llm=llm,
    sys_beg="<|im_start|>system\n",
    sys_end="<|im_end|>\n",
    ai_n_beg="<|im_start|>assistant\n",
    ai_n_end="<|im_end|>",
    usr_n_beg="<|im_start|>user\n",
    usr_n_end="<|im_end|>\n",
    usr_0_beg="<|im_start|>user\n",
    usr_0_end="<|im_end|>\n<|im_start|>assistant\n"
)


DESCRIPTION_PROMPT = """{topic}
Please reply with a creative description of the candidate, {name}, in {word_limit} words or less, that emphasizes their personalities. 
Speak directly to {name}.
Do not add anything else.
"""

TEMPLATE_PROMPT = """{topic}
Your are {name}.
You are a chat candidate.
Your description is as follows: {description}
You are debating the topic: {topic}.
Your goal is to be as creative as possible and make the voters think you are the best candidate.
"""

AGENT_SYSTEM_PROMPT = """{header}
You will speak in the style of {name}.
You will come up with creative ideas related to {topic}.
Do not say the same things over and over again.
Speak in the first person from the perspective of {name}
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
"""

BIDDING_PROMPT = """{header}

```
{{message_history}}
```

On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory, rate how contradictory the following message is to your ideas.

```
{{recent_message}}
```

{bid_instructions}
Do nothing else.
"""


class BidOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."


class GroupChat:
    bid_parser = BidOutputParser(
        regex=r"<(\d+)>", output_keys=["bid"], default_output_key="bid"
    )

    def __init__(self, model: BaseChatModel, topic: str, word_limit: int = 50) -> None:
        self.model = model
        self.topic = topic
        self.agents = []
        self._step = 0
        self._word_limit = word_limit

    def add_agent(self, name: str, description: str | None = None):
        if not description:
            description = model([
                SystemMessage(content="You can add detail to the description of each candidate."),
                HumanMessage(
                    content=DESCRIPTION_PROMPT.format(name=name, topic=self.topic, word_limit=self._word_limit)
                )
            ]).content
        header = TEMPLATE_PROMPT.format(topic=self.topic, name=name, description=description)
        self.agents.append(
            BiddingAgent(
                name=name,
                model=self.model,
                system_message=SystemMessage(
                    content=AGENT_SYSTEM_PROMPT.format(
                        name=name, topic=self.topic, header=header, word_limit=self._word_limit
                    )
                ),
                bidding_template=BIDDING_PROMPT.format(
                    header=header, bid_instructions=self.bid_parser.get_format_instructions()
                )
            )
        )

    def reset(self):
        for agent in self.agents:
            agent.reset()
        
    def start(self):
        character_names = []
        for agent in self.agents:
            character_names.append(agent.name)
        
        game_description = f"""Here is the topic for the chat: {self.topic}.
        The candidates are: {', '.join(character_names)}."""

        topic_specifier_prompt = [
            SystemMessage(content="You can make a task more specific."),
            HumanMessage(
                content=f"""{game_description}

You are the debate moderator.
Please make the debate topic more specific. 
Frame the debate topic as a problem to be solved.
Be creative and imaginative.
Please reply with the specified topic in {self._word_limit} words or less. 
Speak directly to the candidates: {*character_names,}.
Do not add anything else."""
            )
        ]
        specified_topic = model(topic_specifier_prompt).content
        self.inject("Chat Moderator", specified_topic)
        print(f"(Chat Moderator): {specified_topic}")
        print("\n")

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self) -> tuple[str, str]:
        bids = []
        for agent in self.agents:
            bid = self._ask_for_bid(agent)
            bids.append(bid)

        max_value = np.max(bids)
        max_indices = np.where(bids == max_value)[0]
        idx = np.random.choice(max_indices)

        speaker = self.agents[idx]
        message = speaker.send()
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1
        return speaker.name, message

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
        retry_error_callback=lambda retry_state: 0,
    )
    def _ask_for_bid(self, agent):
        bid_string = agent.bid()
        return int(self.bid_parser.parse(bid_string)["bid"])



characters = [
    "Outlaw",
    "Magician",
    "Hero",
    "Lover",
    "Jester",
    "Everyman",
    "Caregiver",
    "Ruler",
    "Creator",
    "Innocent",
    "Sage",
    "Explorer"
]
character_names = [f"The {c} Archetype" for c in characters]
character_names = [
    "Robin Hood",
    "Merlin",
    "King Arthur",
    "Juliet",
    "Puck",
    "Forrest Gump",
    "Mother Teresa",
    "Julius Caesar",
    "Leonardo da Vinci",
    "Alice from Wonderland",
    "Socrates",
    "Christopher Columbus"
]

topic = "First you should think about a alias for yourselves. Then talk about who the new gods are after the death of God in the age of technology?"

simulator = GroupChat(model=model, topic=topic)
for name in character_names:
    simulator.add_agent(name)

simulator.reset()
simulator.start()

n = 0
while n < 100:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print("\n")
    n += 1

