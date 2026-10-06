import chromadb

# Create a local ChromaDB database
client = chromadb.Client()

# Create a collection
collection = client.create_collection(
    name="my_documents"
)

# Documents
documents = [
    "The cat drinks milk.",
    "I drive my car every day.",
    "The weather is very hot today.",
    "A kitten likes drinking milk."
]

# Add documents to the vector database
collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

print("Documents stored successfully!")

# Search the database
query = "Which animal likes milk?"

results = collection.query(
    query_texts=[query],
    n_results=4
)

print()
print("Search Results:")
print("----------------")

documents_found = results["documents"][0]
distances = results["distances"][0]

for document, distance in zip(documents_found, distances):
    print(f"Distance: {distance:.4f} -> {document}")

print()
print("Top Result:")
print(documents_found[0])