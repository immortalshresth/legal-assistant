# app.py
from retrieval_engine import LegalRetrievalEngine
from generation_engine import LegalGenerationEngine

def run_legal_bot(user_query: str):
    retriever = LegalRetrievalEngine()
    generator = LegalGenerationEngine()
    
    # 1. Fetch relevant legal texts using Hybrid + Reranker
    matched_chunks = retriever.hybrid_search(user_query, top_k=3)
    
    # 2. Feed those verified texts to LLaMA 3.3 via Groq
    final_answer = generator.generate_answer(user_query, matched_chunks)
    
    return final_answer