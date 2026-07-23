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

YELLOW_COLOR = "#fab387"
SAPPHIRE_COLOR = "#74c7ec"


@tool
def add_numbers(a: int, b: int):
    """Adds 2 numbers"""
    return a + b


show_reasoning = True


class Agent:
    def __init__(
        self, model: str, provider: str, tools: list, system_prompt: SystemMessage
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
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

    def _render_cli_response(self, prompt):
        for message in self._generate_streaming_response(prompt=prompt):
            if isinstance(message, AIMessageChunk):
                reasoning = message.additional_kwargs.get("reasoning_content")
                if message.content:
                    rprint(
                        f"[{SAPPHIRE_COLOR}]{message.content}[{SAPPHIRE_COLOR}]",
                        end="",
                        flush=True,
                    )
                if reasoning and show_reasoning:
                    rprint(
                        f"[{YELLOW_COLOR}]{reasoning}[/{YELLOW_COLOR}]",
                        end="",
                    )
                for tc in message.tool_call_chunks:
                    if tc.get("name"):
                        rprint(f"\nCalling tool: {tc['name']}")
            elif isinstance(message, ToolMessage):
                rprint(f"\nTool returned: {message.content}")


Terment = Agent(
    model="stepfun-ai/step-3.7-flash",
    provider="nvidia",
    tools=[add_numbers],
    system_prompt=SystemMessage("You are a funny agent"),
)

Terment._render_cli_response("Indie game devs")
