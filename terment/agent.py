from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7
from langchain_core.runnables import RunnableConfig
from langchain.tools import BaseTool


class Agent:
    def __init__(
        self,
        model: str,
        provider: str,
        tools: list[BaseTool],
        system_prompt: str,
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = SystemMessage(system_prompt)
        self.memory_checkpointer = InMemorySaver()
        self.model_provider = provider
        self.llm = init_chat_model(model=model, model_provider=self.model_provider)
        self.config: RunnableConfig = {"configurable": {"thread_id": uuid7()}}

        self.agent = create_agent(
            model=self.llm,
            system_prompt=self.system_prompt,
            checkpointer=self.memory_checkpointer,
            tools=self.tools,
        )

    def _generate_streaming_response(self, prompt: str):
        for message, metadata in self.agent.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=self.config,
            stream_mode="messages",
        ):
            if message:
                yield message
