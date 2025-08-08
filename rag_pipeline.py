import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import your custom RiskAssessor class
from risk_assessor import RiskAssessor

load_dotenv()

class RAGPipeline:
    def __init__(self, faiss_index_path="../faiss_index"):
        self.faiss_index_path = faiss_index_path
        self.index = None
        self.embeddings_model = None
        self.llm = None
        # Instantiate the RiskAssessor to be used by the pipeline
        self.risk_assessor = RiskAssessor()
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initializes the RAG pipeline by loading the FAISS index and models."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")
            
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_index_path}. Please run create_index.py first.")
        
        self.index = FAISS.load_local(self.faiss_index_path, self.embeddings_model, allow_dangerous_deserialization=True)

    def retrieve(self, query, k=5):
        """Retrieves the top k most relevant chunks for a given query."""
        docs = self.index.similarity_search(query, k=k)
        return docs

    def generate(self, query, retrieved_chunks):
        """
        [MODIFIED] Generates a clause, using the external RiskAssessor, and regenerates up to 3 times if risk is 'High'.
        """
        base_prompt = f"**User Intent:** {query}\n\n"
        for i, doc in enumerate(retrieved_chunks):
            base_prompt += f"{i+1}. {doc.page_content}\n"
        
        base_prompt += """
**Instructions:**
- Based solely on the user’s intent and the examples provided above, draft a clear, concise, and legally sound clause.
- Retrieve supporting examples from your RAG index **only if** they directly illustrate or inform the user’s specific request.
- Do **not** include any irrelevant or tangential fragments.
- Do not give irrelevant references or contexts which are not related to the prompt/topic or arent covered in the corpus.
- Attribute any borrowed language or structure from retrieved examples by noting “[Adapted from Example X]” in brackets.
"""
        max_retries = 3
        generated_clause = ""
        current_prompt = base_prompt

        for attempt in range(max_retries):
            response = self.llm.invoke([HumanMessage(content=current_prompt)])
            generated_clause = response.content.strip()
            
            # Use the external RiskAssessor instance to assess the clause
            risk = self.risk_assessor.assess_risk(generated_clause)

            if risk in ["Medium", "Low"]:
                return generated_clause # Success: Exit loop if risk is acceptable

            # If risk is High, and it's not the last attempt, prepare to regenerate
            if attempt < max_retries - 1:
                regeneration_instruction = f"""
---
**Previous Attempt (Risk: {risk}):**
{generated_clause}

**Feedback:** The previous clause was assessed as having a high legal risk. Please revise it to create a more balanced and safer alternative that still fulfills the user's original intent based on the provided examples.
"""
                current_prompt = base_prompt + regeneration_instruction
        
        # Return the last attempt even if risk is still high
        return generated_clause

    def get_metadata_and_source(self, retrieved_chunks):
        """Gets metadata and source from the first retrieved chunk."""
        if not retrieved_chunks:
            return None, None
        first_chunk = retrieved_chunks[0]
        metadata = first_chunk.metadata if hasattr(first_chunk, 'metadata') else {}
        source = metadata.get('source', 'Unknown')
        return metadata, source

    def summarize_text(self, text: str) -> str:
        """Summarizes the given text using the LLM."""
        prompt = f"""
        Please summarize the following text concisely and accurately.
        Text:
        {text}
        Summary:
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content