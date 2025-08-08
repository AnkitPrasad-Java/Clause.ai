import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Chunks text into smaller pieces with a specified overlap."""
    chunks = []
    if not text:
        return chunks
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def create_embeddings(chunks):
    """Creates embeddings for a list of text chunks using SentenceTransformer."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype('float32')

def create_faiss_index(embeddings):
    """Creates a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
