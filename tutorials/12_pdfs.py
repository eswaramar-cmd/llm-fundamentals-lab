import os
import chromadb

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. PDF LOCATION
# ============================================================

PDF_PATH = "./documents/amar.pdf"


# ============================================================
# 2. CHECK PDF
# ============================================================

if not os.path.exists(PDF_PATH):
    print("PDF not found!")
    print("Put your PDF inside the documents folder.")
    exit()


# ============================================================
# 3. READ PDF
# ============================================================

reader = PdfReader(PDF_PATH)

text = ""

for page_number, page in enumerate(reader.pages):

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print("PDF loaded successfully!")
print("Number of pages:", len(reader.pages))
print("Characters extracted:", len(text))


# ============================================================
# 4. SPLIT TEXT INTO CHUNKS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))


# ============================================================
# 5. CREATE EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model ready!")
print("Embedding model: sentence-transformers/all-MiniLM-L6-v2")
print("Embedding dimension: 384")


# ============================================================
# 6. CONNECT TO CHROMADB
# ============================================================

CHROMA_PATH = os.path.abspath("./chroma_rag")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name="documents"
)

print("ChromaDB path:", CHROMA_PATH)
print("ChromaDB collection:", collection.name)
print("ChromaDB connected!")

try:
    existing_count = collection.count()
    if existing_count > 0:
        print("Existing collection found with", existing_count, "documents")
        print("Deleting old collection to ensure correct embedding dimensions...")
        client.delete_collection(name="documents")
        collection = client.get_or_create_collection(
            name="documents"
        )
        print("Old collection deleted. New empty collection created.")
except Exception as e:
    print("Note:", e)


# ============================================================
# 7. CREATE EMBEDDINGS + STORE IN CHROMADB
# ============================================================

for i, chunk in enumerate(chunks):

    embedding = embeddings.embed_query(chunk)

    collection.upsert(
        ids=[f"pdf_chunk_{i}"],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[
            {
                "source": PDF_PATH,
                "chunk": i
            }
        ]
    )

    print(
        f"Stored chunk {i + 1}/{len(chunks)}"
    )


# ============================================================
# 8. FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("RAG INGESTION COMPLETE")
print("=" * 60)

print("PDF:", PDF_PATH)
print("Pages:", len(reader.pages))
print("Chunks:", len(chunks))
print("ChromaDB path:", CHROMA_PATH)
print("ChromaDB collection:", "documents")
print("Embedding dimension: 384")

print()
print("PDF text has been converted into embeddings")
print("and stored in ChromaDB.")