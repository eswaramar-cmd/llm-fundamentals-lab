from langchain_ollama import ChatOllama
from langchain_core.tools import tool


# ============================================================
# 1. CREATE TOOL
# ============================================================

@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers together."""

    return a + b


# ============================================================
# 2. LOAD LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 3. BIND TOOL
# ============================================================

llm_with_tools = llm.bind_tools(
    [calculator]
)


# ============================================================
# 4. USER QUESTION
# ============================================================

question = "What is 25 + 40?"

print("\nUser:")
print(question)


# ============================================================
# 5. ASK LLM
# ============================================================

response = llm_with_tools.invoke(question)

print("\nLLM requested tool:")
print(response.tool_calls)

.
# ============================================================
# 6. EXECUTE TOOL
# ============================================================

tool_call = response.tool_calls[0]

if tool_call["name"] == "calculator":

    result = calculator.invoke(
        tool_call["args"]
    )

    print("\nTool result:")
    print(result)


# ============================================================
# 7. SEND RESULT BACK TO LLM
# ============================================================

final_response = llm.invoke(
    f"""
The user asked:

{question}

The calculator tool returned:

{result}

Give the final answer to the user.
"""
)


# ============================================================
# 8. FINAL ANSWER
# ============================================================

print("\nFinal answer:")
print(final_response.content)