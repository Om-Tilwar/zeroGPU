#vector_store.py
import os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from config.settings import settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from groq import Groq, APIError
# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)

# Initialize HuggingFace embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#     temperature=0.0,
#     top_k=1,
#     top_p=1.0,
#     do_sample=False,
#     repetition_penalty=1.0,
# )
# model = ChatHuggingFace(llm=llm)
 # The user's prompt. You can change this to test different inputs.
user_prompt = "Explain the importance of low-latency LLMs in 100 words."
completion = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }],
    temperature=0.0,
    top_p=1.0,
    stream=True  # or False if you want the full response at once
)


def split_text(text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

async def embed_and_upsert(chunks: list[str], namespace: str):
    print(f"Embedding and upserting {len(chunks)} chunks into namespace: {namespace}")
    try:
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        total_inserted = 0

        print(f"🧮 Total batches to process: {total_batches} (batch size = {batch_size})")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            current_batch_number = (i // batch_size) + 1
            print(f"📦 Processing batch {current_batch_number}/{total_batches}...")

            embeddings = embedding_model.embed_documents(batch)

            vectors = []
            for j, embedding in enumerate(embeddings):
                text = batch[j]
                metadata = {
                    "text": text,
                    "section": "unknown",
                    "page": -1,
                    "source": "",
                    "type": "paragraph",
                }

                vectors.append({
                    "id": f"{namespace}_{i + j}",
                    "values": embedding,
                    "metadata": metadata
                })

            print(f"⬆️ Upserting {len(vectors)} vectors from batch {current_batch_number}...")
            response = index.upsert(vectors=vectors, namespace=namespace)
            print(f"✅ Upsert for batch {current_batch_number} completed. Response: {response}")
            total_inserted += len(vectors)

        return {"status": "success", "inserted": total_inserted}

    except Exception as e:
        print(f"❌ Error in embed_and_upsert: {e}")
        return {"status": "error", "error": str(e)}

async def retrieve_from_kb(input_params):
    """
    Retrieves relevant PARENT chunks from the knowledge base.
    It searches using child chunk embeddings but returns the larger parent context.
    """
    try:
        query = input_params.get("query", "")
        agent_id = input_params.get("agent_id", "")  # agent_id is the namespace
        top_k = input_params.get("top_k", 5) # Retrieve more children to find diverse parents

        if not query or not agent_id:
            return {"chunks": [], "status": "error", "message": "Query and Agent ID are required"}

        print(f"Retrieving context for query: '{query[:50]}...' from namespace: '{agent_id}'")

        # 1. Get embedding for the user's query
        query_vector = embedding_model.embed_query(query)

        # 2. Search Pinecone for top_k similar CHILD vectors
        results = index.query(
            vector=query_vector,
            namespace=agent_id,
            top_k=top_k,
            include_metadata=True
        )

        # 3. Extract the PARENT text from metadata and de-duplicate
        unique_parent_chunks = set()
        for match in results.matches:
            # You can tune this score threshold based on your results
            if match.score > 0.3:
                metadata = match.metadata or {}
                parent_text = metadata.get("parent_text")
                if parent_text:
                    unique_parent_chunks.add(parent_text)
        
        content_blocks = list(unique_parent_chunks)

        if not content_blocks:
            print(f"⚠️ No relevant parent chunks found for namespace '{agent_id}' with the given query.")
        else:
            print(f"✅ Retrieved {len(content_blocks)} unique parent chunk(s).")
            
        return {"chunks": content_blocks}

    except Exception as e:
        print(f"❌ Error in retrieve_from_kb: {e}")
        return {"chunks": [], "status": "error", "error": str(e)}
# Function routing
FUNCTION_HANDLERS = {
    "retrieve_from_kb": retrieve_from_kb
}

FUNCTION_DEFINITIONS = [
    {
        "name": "retrieve_from_kb",
        "description": "Retrieves top-k chunks from the knowledge base using a query and agent_id (namespace).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's search query."
                },
                "agent_id": {
                    "type": "string",
                    "description": "The namespace or agent ID to search in."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 3
                }
            },
            "required": ["query", "agent_id"]
        }
    }
]
