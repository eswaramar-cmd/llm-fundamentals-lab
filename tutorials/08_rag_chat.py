import chromadb
from ollama import chat


# ============================================================
# 1. CREATE CHROMADB
# ============================================================

client = chromadb.Client()

collection = client.create_collection(
    name="rag_chat_documents"
)


# ============================================================
# 2. ADD OUR KNOWLEDGE
# ============================================================

documents = [
    "Apex Care is open from 9 AM to 6 PM.",
    "Apex Care provides dental and wellness services.",
    "Appointments can be cancelled 24 hours in advance.",
    "Apex Care is located in Hyderabad."
]

collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

print("=" * 60)
print("INTERACTIVE RAG CHAT")
print("=" * 60)
print("Type 'quit' to exit.")
print()


# ============================================================
# 3. CHAT LOOP
# ============================================================

while True:

    question = input("Ask a question: ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    print()


    # ========================================================
    # 4. RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    results = collection.query(
        query_texts=[question],
        n_results=2
    )

    retrieved_documents = results["documents"][0]


    # ========================================================
    # 5. CREATE CONTEXT
    # ========================================================

    context = "\n".join(retrieved_documents)


    # ========================================================
    # 6. SEND CONTEXT + QUESTION TO LLM
    # ========================================================

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."

Answer:
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


    # ========================================================
    # 7. PRINT ANSWER
    # ========================================================

    print("Answer:")
    print(response["message"]["content"])
    print()