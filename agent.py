from typing import Any, Dict, List, Optional

from langchain_core.pydantic_v1 import PrivateAttr, Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables.config import RunnableConfig
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult
from langchain.memory import ConversationSummaryMemory
from langchain.tools import StructuredTool, BaseTool
from langchain.tools.render import render_text_description_and_args
from langchain_core.runnables.config import RunnableConfig
from crewai.agents import CrewAgentExecutor, CrewAgentOutputParser, ToolsHandler
from crewai.utilities import Prompts
from crewai import Agent
from llama_cpp import LlamaGrammar

from grammar import schema_to_grammar


OUTPUT_GRAMMAR = '''
root ::= "Thought: " thought "\\n" ( action | final )
action ::= "Action: " tool-input "\\nObservation"
final ::= "Final Answer: " thought "\\n"

{tool_input}

thought ::= [^\\n]*
'''


class ToolCalling(BaseChatModel):
    model: BaseChatModel
    tools: List[StructuredTool] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tool-call-support"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self.model._generate(
            messages,
            stop=stop,
            grammar=self._get_grammar(self.tools),
            run_manager=run_manager,
            **kwargs
        )
    
    def _get_grammar(self, tools):
        tools_schema = tools_to_schema(tools)
        prop_order = ["tool", "arguments"]
        input_grammar = schema_to_grammar(tools_schema, prop_order=prop_order)
        input_grammar = input_grammar.replace("root ::= ", "tool-root ::= ")
        input_grammar += "\ntool-input ::= " + " | ".join([
            f'("{tool.name}\\nAction Input: " ' + f'{i}-arguments' + ')'
            if len(tools_schema["oneOf"][i]["properties"]["arguments"]["properties"]) > 1 else
            f'("{tool.name}\\nAction Input: " [^\\n]*)'
            for i, tool in enumerate(tools)
        ])
        grammar_string = OUTPUT_GRAMMAR.format(tool_input=input_grammar)
        return LlamaGrammar.from_string(grammar_string, verbose=False)


def tools_to_schema(tools: list[BaseTool]) -> Dict[str, Any]:
    return {"oneOf": [tool_to_schema(tool) for tool in tools]}


def tool_to_schema(tool: BaseTool) -> Dict[str, Any]:
    if not isinstance(tool, StructuredTool):
        return {
            "type": "object",
            "properties": {
                "tool": {"const": tool.name},
                "arguments": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}}
                }
            }
        }
    
    schema = tool.args_schema.schema()
    properties = {}
    for k, v in schema["properties"].items():
        properties[k] = {"type": v["type"]}

    return {
        "type": "object",
        "properties": {
            "tool": {"const": tool.name},
            "arguments": {
                "type": "object",
                "properties": properties
            }
        }
    }



class StructuredAgent(Agent):
    _original_llm: Optional[Any] = PrivateAttr()

    def execute_task(self, task: str, context: str = None, tools=None) -> str:
        if context:
            task = self.i18n.slice("task_with_context").format(
                task=task, context=context
            )

        tools = tools or self.tools
        self.agent_executor.tools = tools
        self.llm.tools = tools 
        result = self.agent_executor.invoke(
            {
                "input": task,
                "tool_names": ", ".join([t.name for t in tools]),
                "tools": render_text_description_and_args(tools),
            },
            RunnableConfig(callbacks=[self.tools_handler]),
        )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result
    
    def set_cache_handler(self, cache_handler) -> None:
        self.cache_handler = cache_handler
        self.tools_handler = ToolsHandler(cache=self.cache_handler)
        self.__create_agent_executor()

    def set_rpm_controller(self, rpm_controller) -> None:
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.__create_agent_executor()
    
    def __create_agent_executor(self) -> CrewAgentExecutor:
        if not isinstance(self.llm, ToolCalling):
            # Added this for toolcalling using grammars and tool schemas
            self._original_llm = self.llm
            self.llm = ToolCalling(model=self.llm)
        
        agent_args = {
            "input": lambda x: x["input"],
            "tools": lambda x: x["tools"],
            "tool_names": lambda x: x["tool_names"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        executor_args = {
            "i18n": self.i18n,
            "tools": self.tools,
            "verbose": self.verbose,
            "handle_parsing_errors": True,
            "max_iterations": self.max_iter,
        }

        if self._rpm_controller:
            executor_args[
                "request_within_rpm_limit"
            ] = self._rpm_controller.check_or_wait

        if self.memory:
            summary_memory = ConversationSummaryMemory(
                llm=self._original_llm or self.llm, input_key="input", memory_key="chat_history"
            )
            executor_args["memory"] = summary_memory
            agent_args["chat_history"] = lambda x: x["chat_history"]
            prompt = Prompts(i18n=self.i18n).task_execution_with_memory()
        else:
            prompt = Prompts(i18n=self.i18n).task_execution()

        execution_prompt = prompt.partial(
            goal=self.goal,
            role=self.role,
            backstory=self.backstory,
        )

        bind = self.llm.bind(stop=[self.i18n.slice("observation")])
        inner_agent = (
            agent_args
            | execution_prompt
            | bind
            | CrewAgentOutputParser(
                tools_handler=self.tools_handler,
                cache=self.cache_handler,
                i18n=self.i18n,
            )
        )
        self.agent_executor = CrewAgentExecutor(agent=inner_agent, **executor_args)
