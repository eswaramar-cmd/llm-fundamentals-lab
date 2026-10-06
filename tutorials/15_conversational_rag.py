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
# 2. CONNECT TO CHROMADB
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
# 4. LOAD OLLAMA
# ============================================================

print("\nLoading LLM...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("LLM ready!")


# ============================================================
# 5. CONVERSATION HISTORY
# ============================================================

chat_history = []


# ============================================================
# 6. RAG + CONVERSATION FUNCTION
# ============================================================

def ask_question(question):

    # --------------------------------------------------------
    # Retrieve relevant PDF documents
    # --------------------------------------------------------

    documents = retriever.invoke(question)

    # --------------------------------------------------------
    # Create PDF context
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # --------------------------------------------------------
    # Create conversation history text
    # --------------------------------------------------------

    history = ""

    for user_message, assistant_message in chat_history:

        history += f"""
User: {user_message}
Assistant: {assistant_message}
"""


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant that answers questions
using a PDF.

You have access to:

1. Previous conversation
2. Relevant PDF context

Use the previous conversation to understand
follow-up questions.

Use the PDF context to answer factual questions.

Do NOT make up information.

If the answer cannot be found in the PDF,
say:

"I don't know based on the provided PDF."

================ PREVIOUS CONVERSATION ================

{history}

================ PDF CONTEXT ================

{context}

================ CURRENT QUESTION ================

{question}

================ ANSWER ================
"""


    # --------------------------------------------------------
    # Send to LLM
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    answer = response.content

    # --------------------------------------------------------
    # Save conversation
    # --------------------------------------------------------

    chat_history.append(
        (question, answer)
    )

    return answer


# ============================================================
# 7. START CHATBOT
# ============================================================

print("\n")
print("=" * 60)
print("        CONVERSATIONAL RAG CHATBOT")
print("=" * 60)

print("Ask questions about your PDF.")
print("The chatbot remembers this conversation.")
print("Type 'exit' to stop.")
print()


# ============================================================
# 8. CHAT LOOP
# ============================================================

while True:

    question = input("You: ").strip().lower()

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if question in ["exit", "quit", "bye"]:
        print("\nGoodbye!")
        break


    # --------------------------------------------------------
    # Ignore empty input
    # --------------------------------------------------------

    if not question:
        continue


    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    print("\nSearching PDF...")

    try:

        answer = ask_question(question)

        print("\nAssistant:")
        print(answer)

    except Exception as e:

        print("\nError:")
        print(e)


    print("\n" + "-" * 60)