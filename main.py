import os
from typing import Annotated

from dotenv import load_dotenv, dotenv_values
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import TypedDict

load_dotenv(override=True)

config = dotenv_values(".env")
deepseek_api_key = config.get("DEEPSEEK_API_KEY")

if deepseek_api_key is None:
    raise ValueError("DEEPSEEK_API_KEY is not found in .env file")

llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=deepseek_api_key,
)


class MessagesState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(MessagesState)

def chatbot(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [llm.invoke(state["messages"])]
    }

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
