import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

class LegalRetrievalEngine:
    def __init__(self):
        # FIX: This must be named self.embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "legal_collection"
        
        self.client = QdrantClient(
            url=os.getenv("QDRANT_CLOUD_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )

    def _setup_collection(self):
        """Creates the collection ONLY if it doesn't already exist."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                sparse_vectors_config={"text-sparse": SparseVectorParams()}
            )
            print(f"Created new persistent database: '{self.collection_name}'.")
        else:
            print(f"Successfully connected to existing database: '{self.collection_name}'.")

    def add_legal_chunks(self, chunks: list[dict]):
        """Indexes data into Qdrant using real dense embeddings."""
        points = []
        for chunk in chunks:
            unique_id = uuid.uuid4().hex 
            
            real_vector = self.embedder.encode(chunk["text"]).tolist()
            
            points.append(
                PointStruct(
                    id=unique_id, 
                    vector=real_vector,
                    payload={
                        "document": chunk["text"],
                        "section": chunk["metadata"]["section"],
                        "act": chunk["metadata"]["act"]
                    }
                )
            )
            
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Successfully indexed {len(chunks)} verified legal chunks into Qdrant.")

    def hybrid_search(self, query: str, top_k: int = 5):
        """Executes a direct semantic search, bypassing the strict reranker."""
        query_vector = self.embedder.encode(query).tolist()
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k, 
            with_payload=True
        )
        
        return response.points