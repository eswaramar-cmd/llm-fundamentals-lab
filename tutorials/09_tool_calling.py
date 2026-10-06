from ollama import chat


# ============================================================
# 1. TOOLS
# ============================================================

def add(a: float, b: float):
    return a + b


def multiply(a: float, b: float):
    return a * b


# ============================================================
# 2. TOOL DEFINITIONS
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]


# ============================================================
# 3. CONVERSATION HISTORY
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
You are a helpful AI assistant.

Understand what the user is asking.

If the user asks for a calculation that requires one
of the available tools, call the appropriate tool.

If the user is just chatting, greeting you, or asking
something that does not require a tool, answer normally.

Do NOT call a tool unnecessarily.
"""
    }
]


# ============================================================
# 4. CONTINUOUS CHAT
# ============================================================

print("=" * 60)
print("LLM TOOL CALLING CHAT")
print("=" * 60)
print("The LLM decides when to use a tool.")
print("Type 'quit' to exit.")
print()


while True:

    question = input("Ask something: ").strip()

    if question.lower() in ["quit", "exit", "bye"]:
        print()
        print("Goodbye!")
        break

    # Add user message
    messages.append({
        "role": "user",
        "content": question
    })


    # ========================================================
    # 5. ASK LLM
    # ========================================================

    response = chat(
        model="llama3.2:3b",
        messages=messages,
        tools=tools
    )


    # ========================================================
    # 6. CHECK IF LLM REQUESTED A TOOL
    # ========================================================

    if response.message.tool_calls:

        # Save assistant's tool-call message
        messages.append(response.message)

        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print()
            print("Tool selected:", tool_name)
            print("Arguments:", arguments)


            # =================================================
            # 7. EXECUTE THE TOOL
            # =================================================

            if tool_name == "add":

                result = add(
                    float(arguments["a"]),
                    float(arguments["b"])
                )

            elif tool_name == "multiply":

                result = multiply(
                    float(arguments["a"]),
                    float(arguments["b"])
                )

            else:

                result = "Unknown tool"


            print("Tool result:", result)


            # =================================================
            # 8. SEND TOOL RESULT BACK TO LLM
            # =================================================

            messages.append({
                "role": "tool",
                "content": str(result)
            })


        # =====================================================
        # 9. ASK LLM FOR FINAL NATURAL RESPONSE
        # =====================================================

        final_response = chat(
            model="llama3.2:3b",
            messages=messages,
            tools=tools
        )

        messages.append(final_response.message)

        print()
        print("Answer:")
        print(final_response.message.content)
        print()

    else:

        # =====================================================
        # 10. NORMAL CONVERSATION
        # =====================================================

        messages.append(response.message)

        print()
        print("Tool selected: None")
        print()
        print("Answer:")
        print(response.message.content)
        print()