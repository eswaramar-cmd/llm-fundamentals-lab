from ollama import chat
import mysql.connector


# ============================================================
# 1. CONNECT TO MYSQL
# ============================================================

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Root@1234",
    database="hackathon_db"
)

cursor = connection.cursor()

print("MySQL connected successfully!")


# ============================================================
# 2. SAVE CHAT TO MYSQL
# ============================================================

def save_chat(user_query, ai_response):

    cursor.execute(
        """
        INSERT INTO chat_history (user_query, ai_response)
        VALUES (%s, %s)
        """,
        (user_query, ai_response)
    )

    connection.commit()

    print("Chat saved to MySQL.")


# ============================================================
# 3. CONTINUOUS CHAT
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
You are a helpful AI assistant.
Answer the user's questions naturally.
"""
    }
]


print()
print("=" * 60)
print("AI CHAT WITH MYSQL MEMORY")
print("=" * 60)
print("Type 'quit' to exit.")
print()


while True:

    question = input("Ask something: ").strip()

    if question.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break


    # ========================================================
    # 4. SEND QUESTION TO OLLAMA
    # ========================================================

    messages.append({
        "role": "user",
        "content": question
    })

    response = chat(
        model="llama3.2:3b",
        messages=messages
    )

    answer = response.message.content


    # ========================================================
    # 5. SAVE AI RESPONSE TO CHAT HISTORY
    # ========================================================

    messages.append({
        "role": "assistant",
        "content": answer
    })


    # ========================================================
    # 6. DISPLAY RESPONSE
    # ========================================================

    print()
    print("AI:", answer)
    print()


    # ========================================================
    # 7. AUTOMATICALLY SAVE QUESTION + RESPONSE TO MYSQL
    # ========================================================

    save_chat(
        user_query=question,
        ai_response=answer
    )


# ============================================================
# 8. CLOSE MYSQL CONNECTION
# ============================================================

cursor.close()
connection.close()

print("MySQL connection closed.")