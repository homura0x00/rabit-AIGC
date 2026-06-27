from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

from schema.message import Message, ToolDefinition

class LLMProviderError(Exception):
    """Provider 基础异常"""
    pass

class APIError(LLMProviderError):
    """API 调用错误"""
    pass

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[Message], available_tools: Optional[List[ToolDefinition]]=None) -> Message:
        """
        生成响应

        args:
            messages: 必须的消息列表
            available_tools: 可选的工具列表

        returns:
            Message: 成功时返回
            None: 失败时返回
        """
        pass