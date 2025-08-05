#upsert.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
import re

# --- Configuration ---
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
INDEX_NAME = "hackrx-vector-db"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
SOURCE_FILE_PATH = r"C:\Users\omtil\Downloads\zeroGPU\policy.txt"

# --- Functions (These are all correct and unchanged) ---

def load_document_from_txt(file_path):
    """Loads content from a .txt file."""
    print(f"Loading document from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        document = {
            "id": os.path.basename(file_path),
            "text": text,
            "metadata": {"source": file_path}
        }
        return [document]
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def initialize_pinecone():
    """Initializes and returns a Pinecone client and index object."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)

def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into dynamic, semantically coherent chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def generate_embeddings(chunks, model):
    """Generates vector embeddings for a list of text chunks."""
    print(f"Generating embeddings for {len(chunks)} chunks...")
    return model.encode(chunks)

def upsert_to_pinecone(index, vectors_to_upsert, namespace):
    """Upserts a list of vectors into a specific namespace in the Pinecone index."""
    if not vectors_to_upsert:
        print("No vectors to upsert.")
        return
    print(f"Upserting {len(vectors_to_upsert)} vectors to namespace '{namespace}'...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    print("Upsert complete.")

def generate_namespace(filename: str) -> str:
    """Helper function to create a clean namespace from a filename."""
    name_without_ext = os.path.splitext(filename)[0]
    namespace ="policy"
    return namespace if namespace else "default_namespace"

# --- Main Execution Logic ---

def main():
    """Main function to run the data processing and upserting pipeline."""
    print("--- Starting Pinecone Upsert Process ---")

    # 1. Load data from the text file
    documents = load_document_from_txt(SOURCE_FILE_PATH)
    if not documents:
        print("Halting process due to file loading error.")
        return

    # 2. Generate the namespace from the filename
    source_filename = os.path.basename(SOURCE_FILE_PATH)
    namespace = generate_namespace(source_filename)
    print(f"Targeting Pinecone namespace: '{namespace}'")

    # 3. Initialize Pinecone
    index = initialize_pinecone()

    # 4. Load the embedding model
    print(f"Loading embedding model: '{EMBEDDING_MODEL}'")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    all_vectors = []

    # 5. Process each document (THIS IS THE CORRECTED PART)
    for doc in documents:
        print(f"\nProcessing document: {doc['id']}...")

        # 5a. Create dynamic chunks
        chunks = create_chunks(doc['text'])
        print(f"  - Split into {len(chunks)} chunks.")

        # 5b. Generate embeddings for the chunks
        embeddings = generate_embeddings(chunks, embedding_model)

        # 5c. Prepare vectors for Pinecone
        for i, chunk in enumerate(chunks):
            vector_id = f"{doc['id']}-chunk-{i}"
            vector = {
                "id": vector_id,
                "values": embeddings[i].tolist(),
                "metadata": {
                    **doc['metadata'],
                    "original_text": chunk,
                    "chunk_number": i
                }
            }
            all_vectors.append(vector)

    # 6. Upsert all prepared vectors to Pinecone
    upsert_to_pinecone(index, all_vectors, namespace)
    
    # 7. Verify the upsert
    index_stats = index.describe_index_stats()
    vector_count_in_namespace = index_stats.namespaces.get(namespace, {}).get('vector_count', 0)
    print(f"\nVerification: Namespace '{namespace}' now contains {vector_count_in_namespace} vectors.")
    print("--- Process Finished ---")


if __name__ == "__main__":
    # Ensure you have set your PINECONE_API_KEY in a .env file
    if not PINECONE_API_KEY:
        print("ERROR: Please set your PINECONE_API_KEY in a .env file.")
    else:
        main()