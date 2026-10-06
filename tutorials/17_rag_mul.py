from langchain_ollama import ChatOllama
from langchain_core.tools import tool


# ============================================================
# 1. TOOL 1 - ADD
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""

    return a + b


# ============================================================
# 2. TOOL 2 - MULTIPLY
# ============================================================

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""

    return a * b


# ============================================================
# 3. LOAD OLLAMA
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 4. GIVE MULTIPLE TOOLS TO LLM
# ============================================================

llm_with_tools = llm.bind_tools(
    [add, multiply]
)


# ============================================================
# 5. USER QUESTION
# ============================================================

question = "What is 25 multiplied by 40?"

print("\n" + "=" * 60)
print("USER QUESTION")
print("=" * 60)

print(question)


# ============================================================
# 6. LLM ANALYZES QUESTION
# ============================================================

response = llm_with_tools.invoke(question)


print("\n" + "=" * 60)
print("LLM TOOL CALL")
print("=" * 60)

print(response.tool_calls)


# ============================================================
# 7. GET TOOL CALL
# ============================================================

tool_call = response.tool_calls[0]

tool_name = tool_call["name"]

arguments = tool_call["args"]


print("\nSelected tool:")
print(tool_name)

print("\nArguments:")
print(arguments)


# ============================================================
# 8. EXECUTE SELECTED TOOL
# ============================================================

if tool_name == "add":

    result = add.invoke(arguments)

elif tool_name == "multiply":

    result = multiply.invoke(arguments)

else:

    result = "Unknown tool"


# ============================================================
# 9. TOOL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TOOL RESULT")
print("=" * 60)

print(result)