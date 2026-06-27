from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from enum import Enum


class Role(str, Enum):
    """消息的角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ToolCall(BaseModel):
    """LLM 请求调用某个具体的工具"""
    id: str = Field(description="工具调用的唯一ID")
    name: str = Field(description="被调用的工具名称（例如'bash'）")
    arguments: Union[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="存放的JSON参数，可以是JSON字符串或字典"
    )

class ToolResult(BaseModel):
    tool_call_id: str
    output: str
    is_error: bool

class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: str

class Message(BaseModel):
    role: Role = Field(description="消息角色Role")
    content: str = Field(description="Session content")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description=""
    )
    tool_call_id: Optional[str]