from abc import ABC, abstractmethod
from typing import Dict, Any


class CoalaTool(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    async def ainvoke(self, tool_input: Dict[str, Any]) -> Any:
        """Execute the tool with the given input parameters.

        Args:
            tool_input: A dictionary of input parameters for the tool.

        Returns:
            The result of the tool execution.
        """
        pass
