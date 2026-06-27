from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from schema.message import ToolCall, ToolDefinition, ToolResult


class BaseTool(ABC):
    @abstractmethod
    def name(self):
        pass

class Registry:
    
    def __init__(self, tool: BaseTool) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        pass

    def get_available_tools(self) -> Optional[List[ToolDefinition]]:
        pass

    def execute(self, call: ToolCall) -> Optional[ToolResult]:
        pass
    pass