from agent import Agent
from tools import tool_list


from rich import print as rprint
from langchain.messages import (
    ToolMessage,
    AIMessageChunk,
)

YELLOW_COLOR = "#fab387"
SAPPHIRE_COLOR = "#74c7ec"


class CliAgent(Agent):
    def _render_cli_response(self, prompt: str, show_reasoning: bool = True):
        for message in self._generate_streaming_response(prompt=prompt):
            if isinstance(message, AIMessageChunk):
                reasoning = message.additional_kwargs.get("reasoning_content")
                if message.content:
                    rprint(
                        f"[{SAPPHIRE_COLOR}]{message.content}[/{SAPPHIRE_COLOR}]",
                        end="",
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

    def chat(self):
        try:
            while True:
                prompt = input("\nYOU : ")
                self._render_cli_response(prompt=prompt)
        except KeyboardInterrupt:
            print("Exiting")


Terment = CliAgent(
    model="stepfun-ai/step-3.7-flash",
    provider="nvidia",
    tools=tool_list,
    system_prompt="You are a helpful funny agent",
)

Terment.chat()
