from llm_agent.coala.tools.coala_tool import CoalaTool
from langchain_core.tools import BaseTool
from typing import Dict, Any


class LangchainTool(CoalaTool):
    def __init__(self, langchain_tool: BaseTool, name: str):
        super().__init__(name)
        self.langchain_tool = langchain_tool

    @property
    def description(self):
        return self.langchain_tool.description

    async def ainvoke(self, tool_input: Dict[str, Any]) -> Any:
        return await self.langchain_tool.ainvoke(tool_input)
