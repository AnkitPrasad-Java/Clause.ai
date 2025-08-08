import os
import re
import logging
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# FIX: Manually set an asyncio event loop for the current thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- LangChain Imports ---
try:
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"A required library is not installed. Please check your requirements.txt. Error: {e}", icon="🚨")
    st.stop()

# --- Basic Setup & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Core RAG Functions ---

def init_llm():
    """Initializes the LLM and embeddings models."""
    logger.info("Initializing Gemini LLM and Embeddings...")
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        st.error("Your Gemini API key is missing. Please add it to your .env file.", icon="🔑")
        st.stop()
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.8)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing the Gemini model: {e}", icon="🔥")
        st.stop()

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vector_store(_embeddings):
    """
    This function is now much simpler. It ONLY loads the pre-built index from disk.
    """
    INDEX_PATH = "faiss_index"
    logger.info(f"Checking for pre-built index at '{INDEX_PATH}'...")

    if not os.path.exists(INDEX_PATH):
        logger.error("FAISS index not found!")
        st.error(
            f"The '{INDEX_PATH}' folder was not found. Please run the `create_index.py` script first to build your knowledge base.",
            icon="❌"
        )
        st.stop()
    
    try:
        vector_store = FAISS.load_local(INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
        logger.info("Successfully loaded knowledge base from disk.")
        return vector_store
    except Exception as e:
        logger.error(f"An error occurred while loading the FAISS index: {e}", exc_info=True)
        st.error(f"Could not load the knowledge base. Error: {e}", icon="🔥")
        st.stop()


def generate_clauses(query, vector_store, llm):
    """Runs the RAG chain to generate and parse clauses."""
    prompt_template = """
    You are a specialized AI assistant for drafting legal contracts. Your task is to generate 5 distinct and professional legal clauses based on the user's request, using only the provided context from a knowledge base of existing contracts.

    CONTEXT FROM KNOWLEDGE BASE:
    {context}

    USER'S REQUEST:
    {question}

    INSTRUCTIONS:
    1.  Generate exactly 5 legal clauses.
    2.  Each clause must be well-written, clear, and directly address the user's request.
    3.  Rely strictly on the provided context. Do not invent information.
    4.  The clauses should be diverse, covering different aspects or phrasings related to the request.
    5.  Begin each clause with 'CLAUSE X:' (e.g., 'CLAUSE 1:', 'CLAUSE 2:').
    6.  Do not add any introductory or concluding text outside of the clauses themselves.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 7}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    with st.spinner("AI is drafting the clauses..."):
        result = qa_chain({"query": query})
    
    text = result.get("result", "")
    sources = [doc.metadata.get('source', 'Unknown') for doc in result.get("source_documents", [])]

    clause_pattern = r'CLAUSE\s+\d+:\s*(.*?)(?=CLAUSE\s+\d+:|$)'
    clauses = [m.strip() for m in re.findall(clause_pattern, text, re.DOTALL | re.IGNORECASE) if m.strip()]
    
    return clauses, list(set(sources))


# --- Streamlit UI ---

st.set_page_config(page_title="Legal Clause Generator", layout="wide", page_icon="⚖️")

st.title("AI-Powered Legal Clause Generator ⚖️")
st.markdown("This tool helps you draft legal clauses by leveraging a knowledge base of existing contract documents. Describe the clause you need, and the AI will generate several options based on relevant examples.")

# --- Initialization Step ---
llm, embeddings = init_llm()
vector_store = load_vector_store(embeddings)

st.sidebar.success("Knowledge base is ready.", icon="✅")
st.sidebar.markdown("---")
st.sidebar.info("This application is for demonstration purposes and does not constitute legal advice.", icon="ℹ️")

# --- User Interaction ---
user_query = st.text_input(
    "**Describe the legal clause you need to draft:**",
    placeholder="e.g., a strong confidentiality clause for a financial services SaaS agreement"
)

if user_query:
    if len(user_query) < 15:
        st.warning("Please provide a more detailed description for better results.", icon="⚠️")
    else:
        st.markdown("---")
        clauses, sources = generate_clauses(user_query, vector_store, llm)
        
        if clauses:
            st.subheader("Generated Clauses")
            for i, clause in enumerate(clauses):
                st.markdown(f"**Option {i+1}**")
                st.text_area(label=f"Clause {i+1}", value=clause, height=150, key=f"clause_{i}")
        else:
            st.error("The AI could not generate clauses based on your request. Please try rephrasing your query.", icon="❌")
            
        with st.expander("View Context Sources"):
            if sources:
                st.json([{"source": src.replace('\\', '/')} for src in sources])
            else:
                st.info("No specific source documents were retrieved for this query.")