import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
environment="us-east-1"
index_name="hackrx-vector-db"

your_dimension = 384
from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=PINECONE_API_KEY, environment=environment)

existing_indexes = pc.list_indexes()
print(f"Existing indexes: {existing_indexes}")

index_exists = False
for index in existing_indexes:
    if index.name == index_name:
        index_exists = True
        break

if index_exists:
    print(f"Index '{index_name}' exists. Deleting...")
    pc.delete_index(index_name)
    print(f"Index '{index_name}' deleted successfully.")
    
    import time
    print("Waiting for deletion to complete...")
    # time.sleep(20)

spec = ServerlessSpec(cloud="aws", region="us-east-1")  

try:
    pc.create_index(
        name=index_name,
        spec=spec, 
        dimension=your_dimension,  
        metric="cosine" 
    )
    print(f"New index '{index_name}' created successfully!")
except Exception as e:
    print(f"Error creating index: {e}")
    if "already exists" in str(e).lower():
        print("Index still exists. Pinecone may need more time to complete deletion.")
        print("Try running this script again in a few minutes.")
    else:
        raise

try:
    index = pc.Index(index_name)
    print(f"Successfully connected to index '{index_name}'")
except Exception as e:
    print(f"Error connecting to index: {e}")