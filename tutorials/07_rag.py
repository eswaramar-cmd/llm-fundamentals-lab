import chromadb
from ollama import chat

# ============================================================
# 1. CREATE CHROMADB
# ============================================================

client = chromadb.Client()

collection = client.create_collection(
    name="rag_documents"
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

print("Documents stored!")
print()


# ============================================================
# 3. USER QUESTION
# ============================================================

question = "loacation of apex care?"

print("Question:")
print(question)
print()


# ============================================================
# 4. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

results = collection.query(
    query_texts=[question],
    n_results=2
)

retrieved_documents = results["documents"][0]

print("Retrieved Documents:")
print("--------------------")

for document in retrieved_documents:
    print(document)

print()


# ============================================================
# 5. CREATE CONTEXT
# ============================================================

context = "\n".join(retrieved_documents)


# ============================================================
# 6. SEND CONTEXT + QUESTION TO LLM
# ============================================================

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

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

# ============================================================
# 7. FINAL ANSWER
# ============================================================

print("LLM Answer:")
print("-----------")
print(response["message"]["content"])