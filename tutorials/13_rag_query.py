# ============================================================
# 13_rag_query.py
# RAG QUERY - Ask questions about the PDF
# ============================================================

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model ready!")


# ============================================================
# 2. CONNECT TO EXISTING CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_rag"
)

print("ChromaDB connected!")


# ============================================================
# 3. CHECK DOCUMENT COUNT
# ============================================================

try:
    collection_data = vectorstore._collection.get()

    total_documents = len(collection_data["ids"])

    print(f"Documents stored in ChromaDB: {total_documents}")

except Exception as e:
    print("Could not check document count.")
    print("Error:", e)


# ============================================================
# 4. ASK USER QUESTION
# ============================================================

query = input("\nAsk a question about the PDF: ")

if not query.strip():
    print("Please enter a question.")
    exit()


# ============================================================
# 5. SEARCH SIMILAR DOCUMENTS
# ============================================================

print("\nSearching the PDF...")

try:
    results = vectorstore.similarity_search(
        query,
        k=4
    )

except Exception as e:
    print("\nError while searching ChromaDB:")
    print(e)
    exit()


# ============================================================
# 6. CHECK RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

if not results:
    print("No documents found.")
    print("\nMake sure you have run 12_pdfs.py first.")
    exit()


for i, doc in enumerate(results, start=1):

    print(f"\n--- Document {i} ---")

    print(doc.page_content[:1000])

    print("\nMetadata:")
    print(doc.metadata)


# ============================================================
# 7. COMBINE RETRIEVED DOCUMENTS
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc in results
)


# ============================================================
# 8. CREATE PROMPT
# ============================================================

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I don't know based on the provided PDF."

Do not make up information.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

USER QUESTION:

{query}

Give a clear and concise answer.
"""


# ============================================================
# 9. LOAD LLM
# ============================================================

print("\nLoading LLM...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("LLM ready!")


# ============================================================
# 10. SEND CONTEXT + QUESTION TO LLM
# ============================================================

print("\nGenerating answer...")

try:

    response = llm.invoke(prompt)

except Exception as e:

    print("\nError while generating answer:")
    print(e)
    exit()


# ============================================================
# 11. DISPLAY FINAL ANSWER
# ============================================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)

print("\n" + "=" * 60)