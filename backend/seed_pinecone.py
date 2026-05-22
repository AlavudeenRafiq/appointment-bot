import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader

# --- Initialize model ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Initialize Pinecone client ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pcsk_2uWykL_S9X7zDokB7jZhHAQTRApCLrBkiYEipZFruNATCKLNttLFUdg8GDonnvhK4N9hUa"))

index_name = "rag-index"
namespace = os.getenv("PINECONE_NAMESPACE", "triage-v1")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # MiniLM embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

index = pc.Index(index_name)

# --- Step 1: Read PDF ---
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# --- Step 2: Chunk text ---
def chunk_text(text, chunk_size=220):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# --- Step 3: Insert into Pinecone ---
def seed_pinecone(file_path):
    text = read_pdf(file_path)
    chunks = list(chunk_text(text))
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        index.upsert(
            vectors=[{"id": f"TRIAGE{i}", "values": embedding, "metadata": {"text": chunk}}],
            namespace=namespace
        )
    print(f"Inserted {len(chunks)} chunks from {file_path} into Pinecone namespace {namespace}.")

if __name__ == "__main__":
    seed_pinecone("backend/data/basic_medical_triage_rag_dataset.pdf")
