import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_path):
    """Extract text from a PDF."""
    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text


def split_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    """Generate embeddings for text chunks."""
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings


def create_faiss_index(embeddings):
    """Create a FAISS vector database."""
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def search_similar_chunks(question, chunks, index, top_k=3):
    """Retrieve the most relevant chunks for a question."""

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.array(question_embedding),
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks
