import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the engines we built earlier
from retrieval_engine import LegalRetrievalEngine
from generation_engine import LegalGenerationEngine

# Initialize the FastAPI app
app = FastAPI(
    title="LegalBot Legal Assistant API",
    description="Production-ready backend for retrieving and generating verified legal answers.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows your Next.js frontend to safely call this backend
# Create a specific list of allowed URLs
origins = [
    "http://localhost:3000", # Keeps local testing working
    "https://https://legal-assistant-frontend-gamma.vercel.app/", # MUST be your actual Vercel URL!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Pass the list here instead of ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our core components once during startup
try:
    retriever = LegalRetrievalEngine()
    generator = LegalGenerationEngine()
    
    # Pre-populating some baseline legal chunks for demonstration

except Exception as e:
    print(f"Initialization Error: Ensure GROQ_API_KEY is set. Details: {e}")

# Define the data structure for incoming requests using Pydantic
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# Define the data structure for the outgoing response
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]

@app.get("/")
def health_check():
    """Simple route to check if the server is running properly."""
    return {"status": "healthy", "service": "Legal Assistant Backend"}

@app.post("/api/ask", response_model=QueryResponse)
async def ask_legal_bot(payload: QueryRequest):
    """
    Primary endpoint for processing user legal queries through the complete RAG loop.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Retrieve the most relevant legal context from Qdrant
        matched_chunks = retriever.hybrid_search(payload.question, top_k=payload.top_k)
        
        # 2. Extract sources payload to send back to user for verification transparency
        sources = [
            {
                "section": chunk.payload.get("section", "N/A"),
                "act": chunk.payload.get("act", "N/A"),
                "snippet": chunk.payload.get("document", "")[:150] + "..."
            }
            for chunk in matched_chunks
        ]
        
        # 3. Generate the highly-guarded response via LLaMA 3.3
        ai_answer = generator.generate_answer(payload.question, matched_chunks)
        
        return QueryResponse(
            query=payload.question,
            answer=ai_answer,
            sources=sources
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # <--- THIS FORCES THE TERMINAL TO SPILL THE BEANS
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")