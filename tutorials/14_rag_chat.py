from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

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

print(
    "Documents stored:",
    vectorstore._collection.count()
)


# ============================================================
# 3. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever ready!")


# ============================================================
# 4. LOAD OLLAMA LLM
# ============================================================

print("\nLoading LLM...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("LLM ready!")


# ============================================================
# 5. RAG FUNCTION
# ============================================================

def ask_question(question):

    # Retrieve relevant PDF chunks
    documents = retriever.invoke(question)

    if not documents:
        return "I don't know based on the provided PDF."

    # Combine retrieved text
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the PDF context below.

If the answer is not present in the PDF context,
say:

"I don't know based on the provided PDF."

Do not make up information.

PDF CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}

ANSWER:
"""

    # Send to Llama
    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 6. CONTINUOUS CHAT
# ============================================================

print("\n")
print("=" * 60)
print("             PDF RAG CHATBOT")
print("=" * 60)

print("Ask questions about your PDF.")
print("Type 'exit' to stop.")
print()


while True:

    question = input("You: ").strip().lower()

    # Exit
    if question.lower() in ["exit", "quit", "bye"]:
        print("\nGoodbye!")
        break

    # Ignore empty questions
    if not question:
        continue

    print("\nSearching PDF...")

    try:

        answer = ask_question(question)

        print("\nAssistant:")
        print(answer)

    except Exception as e:

        print("\nError:")
        print(e)

    print("\n" + "-" * 60)