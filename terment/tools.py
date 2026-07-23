from langchain.tools import tool


@tool
def add_numbers(a: int, b: int):
    """Adds 2 integers"""
    return a + b


tool_list = [add_numbers]
