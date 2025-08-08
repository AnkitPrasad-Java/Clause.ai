# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from gtts import gTTS
import uuid
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

# --- MODIFIED: Added LangChain Serper wrapper ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper

from rag_pipeline import RAGPipeline
from risk_assessor import RiskAssessor
from test import text_to_pdf

# --- 1. Initial Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(title="Integrated Legal AI Assistant", version="3.0.0") # Version bump

# --- Middleware ---
origins = ["*"]  # Restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files ---
# Create temp directory for audio files
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Mount the 'static' folder to serve HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- AI Model and Components Initialization ---
try:
    # Gemini Model
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)

    # Serper API Wrapper
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        raise ValueError("SERPER_API_KEY not found in environment variables.")
    # The wrapper automatically uses the SERPER_API_KEY environment variable
    search = GoogleSerperAPIWrapper()

except Exception as e:
    logger.error(f"Failed to initialize AI components: {e}")
    gemini_model = None
    search = None # Handle case where Serper fails to load

rag_pipeline = RAGPipeline(faiss_index_path="faiss_index")
risk_assessor = RiskAssessor()


# --- 2. Helper Class for PDF Processing ---
class PDFProcessor:
    @staticmethod
    def extract_text(pdf_content: bytes) -> str:
        try:
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                return "\n".join(page.get_text("text") for page in doc)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise HTTPException(status_code=400, detail="Error processing PDF file.")

    @staticmethod
    def summarize(text: str) -> str:
        if not gemini_model:
            raise HTTPException(status_code=503, detail="AI Summarizer is currently unavailable.")
        try:
            prompt = f"Summarize the following legal document text clearly and concisely, highlighting the key points, obligations, and any potential areas of concern:\n\n{text}"
            response = gemini_model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing with Gemini: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate summary.")

    @staticmethod
    def to_speech(text: str) -> str:
        try:
            tts = gTTS(text=text, lang="en")
            filename = f"{uuid.uuid4()}.mp3"
            filepath = TEMP_DIR / filename
            tts.save(str(filepath))
            return filename
        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate audio file.")

pdf_processor = PDFProcessor()

# --- 3. Pydantic Models ---
class ClauseRequest(BaseModel):
    prompt: str

class PdfTextRequest(BaseModel):
    clause: str

class EvaluationResponse(BaseModel):
    clause: str
    risk: str
    classification: str
    source: str

class WebSearchResponse(BaseModel):
    generated_text: str
    source_link: str

class CombinedResponse(BaseModel):
    rag_clause: str
    risk: str
    classification: str
    rag_source: str
    web_text: str
    web_source_link: str

# --- 4. API Endpoints ---

# Root endpoint redirects to the main page
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/landing.html")


# --- ADDED: Endpoint for PDF Summarization ---
@app.post("/upload-pdf/")
async def upload_pdf_and_summarize(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        pdf_content = await file.read()
        pdf_text = pdf_processor.extract_text(pdf_content)
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        
        summary = pdf_processor.summarize(pdf_text)
        audio_filename = pdf_processor.to_speech(summary)
        
        return {"summary": summary, "audio_filename": audio_filename, "status": "success"}
    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI's HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in /upload-pdf/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# Endpoint for RAG-based clause evaluation
@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_clause(request: ClauseRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    retrieved_chunks = rag_pipeline.retrieve(request.prompt)
    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="No relevant information found to generate the clause.")

    generated_clause = rag_pipeline.generate(request.prompt, retrieved_chunks)
    risk = risk_assessor.assess_risk(generated_clause)
    classification = risk_assessor.classify_clause(generated_clause)
    _, source = rag_pipeline.get_metadata_and_source(retrieved_chunks)

    return EvaluationResponse(
        clause=generated_clause,
        risk=risk,
        classification=classification,
        source=source,
    )

# Endpoint for clause generation with web search
@app.post("/generate-with-web-search", response_model=WebSearchResponse)
async def generate_with_web_search(request: ClauseRequest):
    if not gemini_model or not search:
         raise HTTPException(status_code=503, detail="AI Search is currently unavailable.")
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        search_results = search.results(request.prompt)
        
        if not search_results or "organic" not in search_results or not search_results["organic"]:
            raise HTTPException(status_code=404, detail="No relevant web pages found for the query.")
            
        top_result = search_results["organic"][0]
        context = top_result.get("snippet", "No snippet available.")
        source_link = top_result.get("link", "#")
        title = top_result.get("title", "Untitled")

        prompt_for_gemini = f"""
        User Query: "{request.prompt}"
        Based on the following content from the web page titled "{title}", please provide a detailed answer to the user's query.
        --- Web Content Snippet ---\n{context}\n--- End of Snippet ---
        Generated Answer:
        """
        
        response = gemini_model.invoke(prompt_for_gemini)
        generated_text = response.content.strip()

        return WebSearchResponse(
            generated_text=generated_text,
            source_link=source_link,
        )

    except HTTPException as http_exc:
        raise http_exc # Re-raise known exceptions
    except Exception as e:
        logger.error(f"Error in web search generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during web search generation.")

# Endpoint to generate both RAG and Web results
@app.post("/generate-combined", response_model=CombinedResponse)
def generate_combined(request: ClauseRequest):
    if not gemini_model or not search:
         raise HTTPException(status_code=503, detail="AI services are currently unavailable.")
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # --- Part 1: RAG Pipeline ---
    rag_clause, risk, classification, rag_source = "", "N/A", "N/A", "N/A"
    try:
        retrieved_chunks = rag_pipeline.retrieve(request.prompt)
        if retrieved_chunks:
            rag_clause = rag_pipeline.generate(request.prompt, retrieved_chunks)
            risk = risk_assessor.assess_risk(rag_clause)
            classification = risk_assessor.classify_clause(rag_clause)
            _, rag_source = rag_pipeline.get_metadata_and_source(retrieved_chunks)
        else:
            rag_clause = "Could not generate a clause from the internal knowledge base."
    except Exception as e:
        logger.error(f"Error in RAG part of combined generation: {e}")
        rag_clause = "An error occurred while generating from the internal knowledge base."

    # --- Part 2: Web Search Pipeline ---
    web_text, web_source_link = "", "#"
# --- Part 2: Web Search Pipeline (with AI-driven query refinement) ---
    web_text, web_source_link = "", "#"
    try:
        # 1. Use Gemini to create an optimized search query
        query_refinement_prompt = f"""
        Based on the following user request, generate a concise and effective search engine query.
        The query should be perfect for finding a relevant legal document, article, or clause.
        Return only the search query itself, without any extra text, labels, or quotation marks.

        User Request: "{request.prompt}"
        
        Optimized Search Query:
        """
        
        search_query_response = gemini_model.invoke(query_refinement_prompt)
        optimized_search_query = search_query_response.content.strip()
        logger.info(f"Original Prompt: '{request.prompt}' | Optimized Query: '{optimized_search_query}'")

        # 2. Use the optimized query to search the web
        search_results = search.results(optimized_search_query)
        
        if search_results and "organic" in search_results and search_results["organic"]:
            top_result = search_results["organic"][0]
            context = top_result.get("snippet", "No snippet available.")
            web_source_link = top_result.get("link", "#")
            title = top_result.get("title", "Untitled")
            
            # 3. Use the search results to generate the final answer to the ORIGINAL user prompt
            prompt_for_gemini = f"""
            Based on the following information from the web page titled "{title}", answer the user's original query.
            
            Original User Query: "{request.prompt}"
            
            --- WEB CONTENT SNIPPET ---
            {context}
            --- END OF SNIPPET ---
            
            Generated Answer:
            """
            response = gemini_model.invoke(prompt_for_gemini)
            web_text = response.content.strip()
        else:
            web_text = "Could not find a relevant source on the web for the refined query."
            
    except Exception as e:
        logger.error(f"Error in Web Search part of combined generation: {e}", exc_info=True)
        web_text = "An error occurred during the web search."

    return CombinedResponse(
        rag_clause=rag_clause,
        risk=risk,
        classification=classification,
        rag_source=rag_source,
        web_text=web_text,
        web_source_link=web_source_link,
    )

# Endpoint to download a clause as a PDF
@app.post("/download_pdf")
async def download_as_pdf(request: PdfTextRequest):
    if not request.clause or not request.clause.strip():
        raise HTTPException(status_code=400, detail="Clause text cannot be empty.")
    
    output_filename = "generated_clause.pdf"
    text_to_pdf(request.clause, output_filename)
    
    return FileResponse(
        output_filename,
        media_type='application/pdf',
        filename=output_filename
    )

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)