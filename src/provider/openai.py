import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from provider.interface import LLMProvider, LLMProviderError
from schema.message import Message, Role, ToolCall, ToolDefinition


class DeepSeekProvider(LLMProvider):
    """DeepSeek Provider（基于 OpenAI 兼容格式，可复用于其他 OpenAI 兼容服务）"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")

        if base_url is None:
            base_url = os.getenv(
                "DEEPSEEK_BASE_URL",
                os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/"),
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _translate_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """将 ToolDefinition 转换为 OpenAI tools 参数格式"""
        openai_tools = []
        for tool in tools:
            # input_schema 是 JSON 字符串，解析为 dict
            try:
                parameters = (
                    json.loads(tool.input_schema)
                    if isinstance(tool.input_schema, str)
                    else tool.input_schema
                )
            except json.JSONDecodeError:
                parameters = {"type": "object", "properties": {}}

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            })
        return openai_tools

    def _translate_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """将内部 Message 列表转换为 OpenAI API 的消息格式"""
        openai_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                openai_messages.append({
                    "content": msg.content,
                    "role": "system",
                })
            elif msg.role == Role.USER:
                if msg.tool_call_id:
                    # tool 调用的结果回传
                    openai_messages.append({
                        "content": msg.content,
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                    })
                else:
                    openai_messages.append({
                        "content": msg.content,
                        "role": "user",
                    })
            elif msg.role == Role.ASSISTANT:
                ast_msg: Dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    ast_msg["content"] = msg.content

                if msg.tool_calls:
                    ast_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": (
                                    json.dumps(tc.arguments, ensure_ascii=False)
                                    if isinstance(tc.arguments, dict)
                                    else tc.arguments
                                ),
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                openai_messages.append(ast_msg)

        return openai_messages

    def generate(
        self,
        messages: List[Message],
        available_tools: Optional[List[ToolDefinition]] = None,
    ) -> Optional[Message]:
        try:
            openai_messages = self._translate_messages(messages)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
            }

            if available_tools:
                kwargs["tools"] = self._translate_tools(available_tools)

            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            openai_msg = choice.message

            # 构造返回的 Message
            result = Message(role=Role.ASSISTANT, content=openai_msg.content or "", tool_call_id=None)

            if openai_msg.tool_calls:
                result.tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in openai_msg.tool_calls
                ]

            return result

        except Exception as e:
            raise LLMProviderError(f"OpenAI API 调用失败: {e}") from e
