from retrieval_engine import LegalRetrievalEngine

def peek_into_brain():
    db = LegalRetrievalEngine()
    
    # 1. Count the total number of laws currently saved in Qdrant
    total_laws = db.client.count(collection_name=db.collection_name).count
    print(f"\n📊 TOTAL LAWS IN DATABASE: {total_laws}")
    
    if total_laws < 500:
        print("🚨 WARNING: You are missing thousands of laws! The database is mostly empty.")
        
    # 2. Perform a RAW semantic search, bypassing all AI logic
    query = "divorce mutual consent hindu marriage act"
    print(f"\n🔍 RAW DATABASE SEARCH FOR: '{query}'")
    
    query_vector = db.embedder.encode(query).tolist()
    response = db.client.query_points(
        collection_name=db.collection_name,
        query=query_vector,
        limit=10,
        with_payload=True
    )
    
    for i, point in enumerate(response.points, 1):
        act = point.payload.get('act', 'N/A')
        section = point.payload.get('section', 'N/A')
        print(f"  {i}. {act} - Section {section}")

if __name__ == "__main__":
    peek_into_brain()