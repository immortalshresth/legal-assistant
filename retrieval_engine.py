import os
import google.generativeai as genai
from qdrant_client import QdrantClient

class LegalRetrievalEngine:
    def __init__(self):
        # Configure the stable Google SDK
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Initialize Qdrant Client
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
        # Collection name configuration
        self.collection_name = "legal_documents"

    def get_embedding(self, text: str) -> list:
        # Generate lightweight embeddings via API call instead of local RAM
        result = genai.embed_content(
            model="models/text-embedding-004",
            contents=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def search_context(self, query: str, top_k: int = 3):
        # Get query embedding using the same API model
        query_vector = genai.embed_content(
            model="models/text-embedding-004",
            contents=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Search Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return search_result