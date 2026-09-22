import getpass
import os
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
import pymupdf

# 生产环境时启动
# def _set_env(key: str) -> None:
#     if key not in os.environ:
#         os.environ[key] = getpass.getpass(f"{key}:")

# _set_env("DEEPSEEK_API_KEY")

# 开发时使用
with open("config.yaml", "r", encoding="utf-8") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print("解析 YAML 失败: {exc}")
        exit(1)

secret_key = config["deepseek_api_key"]

llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=secret_key
)

@tool
def read_pdf_tool():
    """获取PDF格式的简历内容"""
    doc = pymupdf.open('test.pdf')
    page_text = ""
    for page in doc:
        page_text = page.get_text()

    return page_text

@tool
def is_baseline():
    """简历基线过滤器"""
    
    pass

tools = [read_pdf_tool]
llm_with_tools = llm.bind_tools(tools)

class MessagesState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(MessagesState)

def chatbot(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=""
                    )
                ]
                + state["messages"]
            )
        ],

    }
    # return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
