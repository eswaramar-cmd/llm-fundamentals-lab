from ollama import chat
import mysql.connector
import chromadb


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
# 2. CONNECT TO CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_memory"
)

memory_collection = chroma_client.get_or_create_collection(
    name="user_memories"
)

print("ChromaDB connected successfully!")


# ============================================================
# 3. SAVE NORMAL CHAT TO CHAT_HISTORY
# ============================================================

def save_chat(question, answer):

    cursor.execute(
        """
        INSERT INTO chat_history (user_query, ai_response)
        VALUES (%s, %s)
        """,
        (question, answer)
    )

    connection.commit()


# ============================================================
# 4. CHECK IF USER MESSAGE CONTAINS LONG-TERM MEMORY
# ============================================================

def extract_memory(question):

    # Obvious casual messages should NEVER become memory
    casual = {
        "hi",
        "hii",
        "hiii",
        "hello",
        "helo",
        "heloo",
        "hey",
        "heyy",
        "okay",
        "ok",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "good morning",
        "good afternoon",
        "good evening",
        "good night"
    }

    if question.lower().strip() in casual:
        return None


    prompt = f"""
You are a strict memory extractor.

Look ONLY at the user's message.

We want to remember facts ABOUT THE USER that may be useful
in future conversations.

Examples that should be SAVED:

User: My name is Raju
Output:
SAVE|User's name is Raju.

User: I am learning AI and machine learning
Output:
SAVE|User is learning AI and machine learning.

User: I use C++ for DSA
Output:
SAVE|User uses C++ for DSA.

User: I am building an AI chatbot
Output:
SAVE|User is building an AI chatbot.

Examples that should be IGNORED:

User: hi
Output:
IGNORE

User: hello
Output:
IGNORE

User: What is RAG?
Output:
IGNORE

User: Give me C++ code
Output:
IGNORE

User: What is 5 + 5?
Output:
IGNORE

User: Explain Python
Output:
IGNORE

User: What is my name?
Output:
IGNORE

IMPORTANT:

- Questions are NOT memories.
- General knowledge is NOT memory.
- Coding requests are NOT memory.
- Greetings are NOT memory.
- Only facts/preferences/goals/projects about the USER are memory.

Return EXACTLY one of these formats:

SAVE|one short memory

or

IGNORE

User message:
{question}
"""


    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response.message.content.strip()


    # ========================================================
    # STRICT PARSING
    # ========================================================

    if result.startswith("SAVE|"):

        memory = result[5:].strip()

        if memory:
            return memory


    return None


# ============================================================
# 5. SAVE LONG-TERM MEMORY TO MYSQL
# ============================================================

def save_memory_to_mysql(memory):

    cursor.execute(
        """
        INSERT INTO user_memories (memory)
        VALUES (%s)
        """,
        (memory,)
    )

    connection.commit()

    return cursor.lastrowid


# ============================================================
# 6. SAVE MEMORY TO CHROMADB
# ============================================================

def save_memory_to_chromadb(memory_id, memory):

    chroma_id = f"memory_{memory_id}"

    # Avoid duplicate ChromaDB IDs
    existing = memory_collection.get(
        ids=[chroma_id]
    )

    if existing["ids"]:
        return


    memory_collection.add(
        ids=[chroma_id],
        documents=[memory],
        metadatas=[
            {
                "mysql_id": memory_id
            }
        ]
    )


# ============================================================
# 7. LOAD MYSQL MEMORIES INTO CHROMADB
# ============================================================

cursor.execute(
    """
    SELECT id, memory
    FROM user_memories
    ORDER BY id ASC
    """
)

existing_memories = cursor.fetchall()

print(
    f"Found {len(existing_memories)} long-term memories in MySQL."
)


for memory_id, memory in existing_memories:

    save_memory_to_chromadb(
        memory_id,
        memory
    )


print("Long-term memories synced with ChromaDB.")


# ============================================================
# 8. SEARCH CHROMADB MEMORY
# ============================================================

def search_memory(question):

    count = memory_collection.count()

    if count == 0:
        return []


    results = memory_collection.query(
        query_texts=[question],
        n_results=min(5, count)
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]


    return documents


# ============================================================
# 9. START CHAT
# ============================================================

print()
print("=" * 60)
print("AI CHAT WITH MYSQL + CHROMADB LONG-TERM MEMORY")
print("=" * 60)
print("Type 'quit' to exit.")
print()


# ============================================================
# 10. CONTINUOUS CHAT
# ============================================================

while True:

    question = input("Ask something: ").strip()


    if not question:
        continue


    if question.lower() in [
        "quit",
        "exit"
    ]:

        print("Goodbye!")
        break


    # ========================================================
    # 11. RETRIEVE POSSIBLE MEMORIES
    # ========================================================

    memories = search_memory(question)


    memory_context = ""


    if memories:

        memory_context = "\n".join(
            f"- {memory}"
            for memory in memories
        )


    # ========================================================
    # 12. ASK OLLAMA
    # ========================================================

    prompt = f"""
You are a helpful AI assistant.

You have access to long-term information about the user.

Long-term memory:
{memory_context}

Current user question:
{question}

Instructions:

1. Answer naturally.
2. Use the memory when it is relevant.
3. Do not invent information.
4. If the memory does not contain the answer, answer normally.
5. Do not mention MySQL, ChromaDB, embeddings, or memory
   retrieval to the user.
"""


    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    answer = response.message.content.strip()


    # ========================================================
    # 13. SHOW ANSWER
    # ========================================================

    print()
    print("AI:", answer)
    print()


    # ========================================================
    # 14. SAVE NORMAL CHAT
    # ========================================================

    save_chat(
        question,
        answer
    )

    print("Chat saved to chat_history.")


    # ========================================================
    # 15. CHECK FOR LONG-TERM MEMORY
    # ========================================================

    memory = extract_memory(question)


    if memory:

        print("Long-term memory detected:")
        print(memory)


        # ----------------------------------------------------
        # Save memory to MySQL
        # ----------------------------------------------------

        memory_id = save_memory_to_mysql(
            memory
        )

        print("Memory saved to user_memories.")


        # ----------------------------------------------------
        # Save memory to ChromaDB
        # ----------------------------------------------------

        save_memory_to_chromadb(
            memory_id,
            memory
        )

        print("Memory embedding saved to ChromaDB.")


    else:

        print("No long-term memory to save.")


    print()


# ============================================================
# 16. CLOSE MYSQL
# ============================================================

cursor.close()
connection.close()

print("MySQL connection closed.")