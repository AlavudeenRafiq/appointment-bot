import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# --- Initialize model ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Initialize Pinecone client ---
pc = Pinecone(api_key="pcsk_2uWykL_S9X7zDokB7jZhHAQTRApCLrBkiYEipZFruNATCKLNttLFUdg8GDonnvhK4N9hUa")


# --- Ensure index exists ---
index_name = "rag-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # Hugging Face MiniLM embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")  # adjust region if needed
    )

index = pc.Index(index_name)

def insert_text(item_id: str, text: str):
    """Insert a single text chunk into Pinecone"""
    embedding = model.encode(text).tolist()
    index.upsert(vectors=[{"id": item_id, "values": embedding, "metadata": {"text": text}}])

def query_text(query: str, top_k: int = 5):
    """Query Pinecone for relevant context"""
    embedding = model.encode(query).tolist()
    result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [m.metadata["text"] for m in result.matches]