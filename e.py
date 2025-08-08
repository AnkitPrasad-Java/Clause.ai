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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Script Constants ---
SOURCE_DOCUMENTS_PATH = "full_contract_txt"
SAMPLE_SIZE = 5  # <--- We will only process the first 5 files
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def estimate_indexing_time():
    """
    Processes a sample of documents to estimate the total time required for full indexing.
    """
    # --- 1. File Discovery ---
    logging.info("Scanning source directory to count total files...")
    source_path = Path(SOURCE_DOCUMENTS_PATH)
    if not source_path.exists():
        logging.error(f"Source directory '{SOURCE_DOCUMENTS_PATH}' is missing. Aborting.")
        return

    all_files = sorted(list(source_path.glob("*.txt"))) # Sort for consistency
    total_files = len(all_files)

    if total_files == 0:
        logging.error(f"No .txt files found in '{SOURCE_DOCUMENTS_PATH}'. Aborting.")
        return
        
    # Adjust sample size if there are fewer files than the desired sample size
    actual_sample_size = min(total_files, SAMPLE_SIZE)
    sample_files = all_files[:actual_sample_size]
    
    logging.info(f"Found {total_files} total documents. Processing a sample of {len(sample_files)} files.")

    # --- 2. Loading & Chunking Sample ---
    docs = []
    for file in tqdm(sample_files, desc=f"📚 Loading {len(sample_files)} Sample Files"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())

    logging.info(f"Splitting sample documents into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Created {len(chunks)} text chunks from the sample.")

    # --- 3. Embedding & Indexing Sample ---
    logging.info("Initializing Google Generative AI Embeddings model...")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize embeddings model: {e}")
        return

    logging.info(f"Starting FAISS index creation for the {len(sample_files)}-file sample...")
    start_time = time.time()

    try:
        # Create the index in-memory (we will not save it)
        FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        logging.error(f"An error occurred during sample index creation: {e}")
        return

    end_time = time.time()

    # --- 4. Calculation & Estimation ---
    time_for_sample = end_time - start_time
    avg_time_per_file = time_for_sample / len(sample_files)
    estimated_total_time_seconds = avg_time_per_file * total_files
    
    # Convert seconds to a more readable format (minutes and seconds)
    est_minutes, est_seconds = divmod(estimated_total_time_seconds, 60)

    logging.info("--- ⏱️ Time Estimation Complete ⏱️ ---")
    logging.info(f"Time to process {len(sample_files)} files: {time_for_sample:.2f} seconds.")
    logging.info(f"Average time per file: {avg_time_per_file:.2f} seconds.")
    logging.info(f"Total files in directory: {total_files}.")
    logging.info(f"ESTIMATED TOTAL TIME for all {total_files} files: {int(est_minutes)} minutes and {int(est_seconds)} seconds.")
    logging.info("Note: This script did NOT save the index. Run 'create_index.py' to build the full index.")

if __name__ == "__main__":
    estimate_indexing_time()