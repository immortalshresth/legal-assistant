import json
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- 1. EXPLICIT CLOUD CONNECTION ---
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300.0
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "legal_collection"

# Ensure the collection exists in the cloud before uploading
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Created new cloud collection!")

def ingest_multiple_jsons(file_list: list):
    # 1. Create a massive master bucket to hold everything
    all_master_points = [] 
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} - File not found.")
            continue
            
        print(f"\n--- Cracking open {file_path} ---")
        act_name = file_path.replace('.json', '').upper()
        
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            
            # Handle nested dict formats if they sneak in
            if isinstance(raw_data, dict) and "acts" in raw_data:
                 raw_data = list(raw_data["acts"].values())[0]

            if not isinstance(raw_data, list):
                print(f"Warning: {file_path} is not a valid list. Skipping.")
                continue
            
            for idx, row in enumerate(raw_data):
                text = row.get('description') or row.get('Description') or row.get('section_desc') or row.get('text') or row.get('content') or ""
                title = row.get('title') or row.get('Title') or row.get('section_title') or ""
                section = row.get('section') or row.get('Section') or row.get('section_number') or f"Sec {idx+1}"
                
                full_text = f"Act: {act_name}\nSection: {section}\nTitle: {title}\nDescription: {text}".strip()
                
                if text and full_text != ".": 
                    # Create the vector embedding
                    vector = embedding_model.encode(full_text).tolist()
                    
                    # Create a bulletproof unique ID
                    string_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{act_name}_{section}_{idx}"))
                    
                    point = PointStruct(
                        id=string_uuid,
                        vector=vector,
                        payload={
                            "act": act_name,
                            "section": str(section),
                            "title": title,
                            "document": full_text
                        }
                    )
                    all_master_points.append(point)
                    
        print(f"Prepared laws from {act_name}.")

    # 3. Call the Cloud Database in safe BATCHES
    batch_size = 50
    total_points = len(all_master_points)
    print(f"\nPreparing to inject a massive total of {total_points} laws directly into Qdrant Cloud...")
    
    if total_points == 0:
        print("No valid laws found to upload.")
        return

    for i in range(0, total_points, batch_size):
        batch = all_master_points[i : i + batch_size]
        print(f"Injecting batch {i//batch_size + 1} (Laws {i} to {i + len(batch)})...")
        
        # Pushing directly to the Cloud Client!
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME, 
            points=batch
        )
        
    print("\n✅ MASSIVE CLOUD BULK LOAD COMPLETE!")

if __name__ == "__main__":
    my_law_books = [
        "ipc.json", 
        "crpc.json", 
        "cpc.json", 
        "hma.json",
        "mva.json",
        "nia.json",
        "iea.json",
        "ida.json"
    ]
    
    ingest_multiple_jsons(my_law_books)