from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    ToolCall,
    AIMessageChunk,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7
from langchain_core.runnables import RunnableConfig
from rich import print as rprint
from langchain.tools import tool


@tool
def add_numbers(a: int, b: int):
    """Adds 2 numbers"""
    return a + b


memory_checkpointer = InMemorySaver()

llm = init_chat_model(model="nvidia:stepfun-ai/step-3.7-flash")

agent = create_agent(
    model=llm,
    system_prompt=SystemMessage(content="You are a humuorous agent"),
    # checkpointer=memory_checkpointer,
    tools=[add_numbers],
)
config: RunnableConfig = {"configurable": {"thread_id": uuid7()}}

for message, metadata in agent.stream(
    {"messages": [HumanMessage(content="Add 10000 and 299247")]},
    config=config,
    stream_mode="messages",
):
    if isinstance(message, AIMessageChunk):

        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning:
            print(reasoning, end="", flush=True)

        for tc in message.tool_call_chunks:
            if tc.get("name"):
                print(f"\nCalling tool: {tc['name']}")

        if message.content:
            print(message.content, end="", flush=True)

    elif isinstance(message, ToolMessage):
        print(f"\nTool returned: {message.content}")
