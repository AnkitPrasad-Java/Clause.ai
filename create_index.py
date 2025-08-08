import os
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# --- LangChain Imports ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Configuration ---
# Configure logging to print status updates to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file (for the API key)
load_dotenv()

# --- Script Constants (modify these to change behavior) ---
SOURCE_DOCUMENTS_PATH = "full_contract_txt"
FAISS_INDEX_SAVE_PATH = "faiss_index"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def create_faiss_index():
    """
    Reads documents, creates embeddings, builds a FAISS index, and saves it to disk.
    """
    # 1. --- Loading & Chunking ---
    logging.info(f"Looking for source documents in '{SOURCE_DOCUMENTS_PATH}'...")
    source_path = Path(SOURCE_DOCUMENTS_PATH)
    if not source_path.exists() or not any(source_path.iterdir()):
        logging.error(f"Source directory '{SOURCE_DOCUMENTS_PATH}' is missing or empty. Aborting.")
        return

    files = list(source_path.glob("*.txt"))
    logging.info(f"Found {len(files)} documents to process.")

    # Load all documents with a progress bar
    docs = []
    for file in tqdm(files, desc="📚 Loading Documents"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())

    # Split the documents into smaller chunks
    logging.info(f"Splitting {len(docs)} loaded pages into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Successfully created {len(chunks)} text chunks.")

    # 2. --- Embedding ---
    logging.info("Initializing Google Generative AI Embeddings model...")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize embeddings model: {e}")
        return

    # 3. --- Indexing & Saving ---
    logging.info("Starting the creation of the FAISS vector index. This may take several minutes...")
    start_time = time.time()

    try:
        # This is the core step that creates the index from all chunks and their embeddings
        db = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        logging.error(f"An error occurred during FAISS index creation: {e}")
        return

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"FAISS index created successfully in {elapsed_time:.2f} seconds.")

    # Save the newly created index to the local disk
    logging.info(f"Saving index to disk at '{FAISS_INDEX_SAVE_PATH}'...")
    db.save_local(FAISS_INDEX_SAVE_PATH)
    logging.info("✅ Index saved successfully. You can now use this in your Streamlit app.")


if __name__ == "__main__":
    create_faiss_index()