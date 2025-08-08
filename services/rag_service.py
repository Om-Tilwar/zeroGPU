# services/rag_service.py

from pinecone import Pinecone
from services.vector_store import retrieve_from_kb
from services.hf_model import ask_gpt
import re
import asyncio
from config.settings import settings
import os
from urllib.parse import urlparse

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    print("✅ Initialized Pinecone connection.")
except Exception as e:
    print(f"❌ Error initializing Pinecone: {e}")
    raise

def generate_namespace_from_url(url: str) -> str:
    """
    Generates a namespace from a URL, exactly matching the offline script's logic.
    This version is intentionally simple to match existing data in Pinecone.
    """
    try:
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # This is the original, flawed logic that matches your existing data
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name_without_ext).lower()
        
        namespace = safe_name if safe_name else "default_namespace"
        return namespace

    except Exception as e:
        print(f"Error generating namespace from URL '{url}': {e}")
        return "default_namespace"

async def process_documents_and_questions(pdf_url: str, questions: list[str], namespace: str = None) -> dict:
    print(f"Processing questions for PDF URL: {pdf_url}")
    
    try:
        # Step 1: Determine the correct namespace
        if namespace:
            agent_id = namespace
            print(f"📂 Using provided namespace: '{agent_id}'")
        else:
            agent_id = generate_namespace_from_url(pdf_url)
            print(f"📂 Generated namespace from URL: '{agent_id}'")

        # Verify namespace exists in Pinecone
        stats = index.describe_index_stats()
        existing_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        if agent_id not in existing_namespaces:
            print(f"⚠️ Namespace '{agent_id}' not found in Pinecone index.")
            print(f"🔍 Available namespaces: {existing_namespaces}")
            # Raising an error is often better than proceeding with incorrect context
            raise Exception(f"Namespace '{agent_id}' does not exist in the Pinecone index.")

        # Step 2: Process questions in parallel
        semaphore = asyncio.Semaphore(3)  # Limit concurrency to avoid overwhelming the API

        async def process_single_question(q_index: int, question: str) -> tuple[int, str, str]:
            async with semaphore:
                try:
                    # Retrieve context from Pinecone
                    retrieval_input = {"query": question, "agent_id": agent_id, "top_k": 3}
                    retrieved = await retrieve_from_kb(retrieval_input)
                    retrieved_chunks = retrieved.get("chunks", [])
                    
                    if not retrieved_chunks:
                        print(f"⚠️ Q{q_index+1}: No chunks retrieved for question: '{question[:50]}...'")
                        return (q_index, question, "I couldn't find relevant information in the document to answer this question.")

                    context = "\n".join(retrieved_chunks)[:3000] # Limit context size
                    print(f"➡️ Q{q_index+1}: Passing context to model for question: '{question[:50]}...'")

                    # Get answer from the model
                    answer = await ask_gpt(context, question)
                    return (q_index, question, answer)
                
                except Exception as e:
                    print(f"❌ Q{q_index+1}: Failed to process with error: {e}")
                    return (q_index, question, "An error occurred while trying to answer this question.")

        print(f"🧠 Processing {len(questions)} questions in parallel...")
        if not questions:
            return {}
            
        tasks = [process_single_question(i, q) for i, q in enumerate(questions)]
        responses = await asyncio.gather(*tasks)

        # Step 3: Sort and return results
        sorted_responses = sorted(responses, key=lambda x: x[0])
        results = {q: ans for _, q, ans in sorted_responses}
        return results
        
    except Exception as e:
        print(f"❌ A critical error occurred in process_documents_and_questions: {e}")
        # Re-raise the exception to be handled by the calling function (e.g., in your API endpoint)
        raise