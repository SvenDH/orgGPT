from typing import Any, Dict, List, Optional

from langchain_core.pydantic_v1 import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult
from langchain.tools import StructuredTool, BaseTool

from llama_cpp import LlamaGrammar
from grammar import schema_to_grammar


OUTPUT_GRAMMAR = '''
root ::= "{ai_beg}Thought: " thought "\\n" ( action | final ) "{ai_end}"
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
        
        tools = self.tools
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
        grammar = OUTPUT_GRAMMAR.format(
            ai_beg=self.model.ai_n_beg,
            ai_end=self.model.ai_n_end,
            tool_input=input_grammar,
        )
        grammar = LlamaGrammar.from_string(grammar, verbose=False)

        return self.model._generate(
            messages,
            stop=stop,
            grammar=grammar,
            run_manager=run_manager,
            **kwargs
        )


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
