from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""

    if b == 0:
        return "Cannot divide by zero"

    return a / b


# ============================================================
# 2. TOOL MAP
# ============================================================

tools = [
    add,
    multiply,
    subtract,
    divide
]

tool_map = {
    "add": add,
    "multiply": multiply,
    "subtract": subtract,
    "divide": divide
}


# ============================================================
# 3. LOAD OLLAMA
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 4. BIND TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 5. USER QUESTION
# ============================================================

question = """
Calculate 24 multiplied by 40, then add 100.
"""

print("\nUSER:")
print(question)


# ============================================================
# 6. INITIAL MESSAGE
# ============================================================

messages = [
    HumanMessage(content=question)
]


# ============================================================
# 7. AGENT LOOP
# ============================================================

while True:

    print("\n" + "=" * 60)
    print("LLM THINKING / DECIDING")
    print("=" * 60)


    # --------------------------------------------------------
    # LLM decides next action
    # --------------------------------------------------------

    response = llm_with_tools.invoke(messages)


    # --------------------------------------------------------
    # Store LLM response
    # --------------------------------------------------------

    messages.append(response)


    # ========================================================
    # 8. CHECK IF LLM WANTS TO USE A TOOL
    # ========================================================

    if not response.tool_calls:

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(response.content)

        break


    # ========================================================
    # 9. PROCESS TOOL CALLS
    # ========================================================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        arguments = tool_call["args"]

        print("\nSelected tool:")
        print(tool_name)

        print("\nArguments:")
        print(arguments)


        # ====================================================
        # 10. HANDLE NONE ARGUMENT
        # ====================================================

        if tool_name == "add" and arguments.get("a") is None:

            previous_result = None

            # Find the latest tool result
            for message in reversed(messages):

                if isinstance(message, ToolMessage):

                    try:
                        previous_result = float(message.content)
                        break

                    except ValueError:
                        continue


            # Use previous result
            if previous_result is not None:

                arguments["a"] = int(previous_result)

                # For our current example
                arguments["b"] = 100

                print("\nFixed arguments:")
                print(arguments)


        # ====================================================
        # 11. FIND TOOL
        # ====================================================

        selected_tool = tool_map.get(tool_name)


        if selected_tool is None:

            result = "Unknown tool"

        else:

            result = selected_tool.invoke(arguments)


        # ====================================================
        # 12. TOOL RESULT
        # ====================================================

        print("\nTool result:")
        print(result)


        # ====================================================
        # 13. SEND RESULT BACK TO LLM
        # ====================================================

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )


        print("\nTool result sent back to LLM.")
        print("Continuing agent loop...")