import inspect
import copy
from typing import Callable
from collections import defaultdict

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.memory.prompt import SUMMARY_PROMPT


class Agent:
    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
    """

    DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        system_message: str | None = None,
        description: str | None = None,
        max_token_limit: int = 2000,
    ) -> None:
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        self.model = model
        self.prefix = f"{self.name}: "
        self.description = description or self.DEFAULT_DESCRIPTION
        self.max_token_limit = max_token_limit
        self.moving_summary_buffer = ""
        self.summary_chain = LLMChain(llm=self.model, prompt=SUMMARY_PROMPT)
        self.reset()
        
        self._reply_func_list = []

        self.reply_at_receive = defaultdict(bool)
        self.register_reply([Agent, None], Agent.generate_reply)
        self.register_reply([Agent, None], Agent.generate_code_execution_reply)
        self.register_reply([Agent, None], Agent.generate_tool_calls_reply)
        self.register_reply([Agent, None], Agent.generate_function_call_reply)
        self.register_reply([Agent, None], Agent.check_termination_and_human_reply)

        self.hook_lists = {self.process_last_message: []}

    @property
    def message_history(self):
        buffer = self.buffer
        if self.moving_summary_buffer != "":
            buffer = [self.moving_summary_buffer] + buffer
        return buffer

    def reset(self):
        self.buffer = ["Here is the conversation so far."]
    
    def generate_reply(self, messages, sender):
        all_messages = []
        for message in messages:
            tool_responses = message.get("tool_responses", [])
            if tool_responses:
                all_messages += tool_responses
                if message.get("role") != "tool":
                    all_messages.append({key: message[key] for key in message if key != "tool_responses"})
            else:
                all_messages.append(message)

        response = self.model([
            SystemMessage(self.system_message),
            HumanMessage(content="\n\n".join(self.message_history + [self.prefix])),
        ])

        response = client.create(
            context=messages[-1].pop("context", None),
            messages=self._oai_system_message + all_messages,
            cache=self.client_cache,
        )

        extracted_response = client.extract_text_or_completion_object(response)[0]

        # ensure function and tool calls will be accepted when sent back to the LLM
        if not isinstance(extracted_response, str):
            extracted_response = model_dump(extracted_response)
        if isinstance(extracted_response, dict):
            if extracted_response.get("function_call"):
                extracted_response["function_call"]["name"] = self._normalize_name(
                    extracted_response["function_call"]["name"]
                )
            for tool_call in extracted_response.get("tool_calls") or []:
                tool_call["function"]["name"] = self._normalize_name(tool_call["function"]["name"])
        return True, extracted_response

    def send(self) -> str:
        message = self.model([
            SystemMessage(self.system_message),
            HumanMessage(content="\n\n".join(self.message_history + [self.prefix])),
        ])
        return message.content
    
    def prune(self):
        buffer = self.buffer
        curr_buffer_length = self.model.llm.get_num_tokens("\n\n".join(buffer))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.model.llm.get_num_tokens("\n\n".join(buffer))
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )

    def receive(self, name, message) -> None:
        self.buffer.append(f"{name}: {message}")
        self.prune()

    def predict_new_summary(self, messages, existing_summary) -> str:
        return self.summary_chain.predict(summary=existing_summary, new_lines="\n\n".join(messages))

    def register_reply(self, trigger, reply_func, position=0, config=None, reset_config=None):
        if not isinstance(trigger, (type, str, Agent, Callable, list)):
            raise ValueError("trigger must be a class, a string, an agent, a callable or a list.")
        self._reply_func_list.insert(position, {
            "trigger": trigger,
            "reply_func": reply_func,
            "config": copy.copy(config),
            "init_config": config,
            "reset_config": reset_config,
        })
    


class BiddingAgent(Agent):
    def __init__(
        self,
        name,
        bidding_template: PromptTemplate,
        model: BaseChatModel,
        system_message: str | None = None,
        description: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(name, model, system_message, description, **kwargs)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        prompt = PromptTemplate(
            input_variables=["message_history", "recent_message"],
            template=self.bidding_template,
        ).format(
            message_history="\n\n".join(self.message_history),
            recent_message=self.message_history[-1],
        )
        return self.model([HumanMessage(content=prompt)]).content
