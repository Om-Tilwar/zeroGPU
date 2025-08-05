#rag_service.py
from pinecone import Pinecone
from services.vector_store import retrieve_from_kb
from services.hf_model import ask_gpt
import re
import asyncio
from config.settings import settings

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

def generate_namespace_from_url(url: str) -> str:
    """Generate namespace matching the embedding script logic"""
    import os
    from urllib.parse import urlparse
    
    try:
        # Parse URL to get the filename part
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1]
        
        # Remove query parameters if they're part of the filename
        if '?' in filename:
            filename = filename.split('?')[0]
            
        # Remove extension (matching your embedding script logic)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Replace non-alphanumeric characters with underscores and convert to lowercase
        # This matches exactly: re.sub(r'[^a-zA-Z0-9]+', '_', name_without_ext).strip('_').lower()
        namespace = "policy"
        
        # Ensure namespace is not empty (matching your embedding script)
        if not namespace:
            namespace = "default_namespace"
            
        return namespace
        
    except Exception as e:
        print(f"Error generating namespace: {e}")
        return "default_namespace"

async def list_available_namespaces() -> list[str]:
    """Helper function to list all available namespaces in the Pinecone index"""
    try:
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        return namespaces
    except Exception as e:
        print(f"Error retrieving namespaces: {e}")
        return []

async def process_documents_and_questions(pdf_url: str, questions: list[str], namespace: str = None) -> dict:
    print(f"Processing questions for PDF URL: {pdf_url}")
    
    try:
        # Step 1: Handle namespace determination
        if namespace:
            # Use provided namespace directly
            agent_id = namespace
            print(f"📂 Using provided namespace: '{agent_id}'")
        else:
            # Generate namespace using the same logic as your embedding scripts
            agent_id = generate_namespace_from_url(pdf_url)
            print(f"📂 Generated namespace: '{agent_id}'")
        
        # Debug: Check what namespaces actually exist
        try:
            stats = index.describe_index_stats()
            existing_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            print(f"🔍 Available namespaces: {existing_namespaces}")
            
            if agent_id not in existing_namespaces:
                print(f"⚠️ Namespace '{agent_id}' not found in existing namespaces")
                
                if not namespace:  # Only auto-select if namespace wasn't provided
                    # Try common patterns based on your embedding scripts
                    possible_namespaces = [
                        agent_id,  # Direct match
                        "extracted_text_embedding",  # For text files
                        f"{agent_id}_pdf",  # With suffix
                        f"doc_{agent_id}",  # With prefix
                    ]
                    
                    # Also try partial matches for existing namespaces
                    for existing_ns in existing_namespaces:
                        if agent_id in existing_ns.lower() or existing_ns.lower() in agent_id:
                            possible_namespaces.append(existing_ns)
                    
                    found_namespace = None
                    for candidate in possible_namespaces:
                        if candidate in existing_namespaces:
                            found_namespace = candidate
                            break
                    
                    if found_namespace:
                        agent_id = found_namespace
                        print(f"🔄 Found matching namespace: '{agent_id}'")
                    else:
                        print(f"❌ No matching namespace found.")
                        print(f"Expected: '{generate_namespace_from_url(pdf_url)}'")
                        print(f"Available: {existing_namespaces}")
                        raise Exception(f"No matching namespace found for PDF. Expected: '{generate_namespace_from_url(pdf_url)}', Available: {existing_namespaces}")
                else:
                    raise Exception(f"Provided namespace '{namespace}' not found. Available: {existing_namespaces}")
                
        except Exception as e:
            if "No matching namespace found" in str(e):
                raise e
            print(f"Error checking namespaces: {e}")
            # Continue with the generated namespace anyway

        # Step 2: Parallel question processing with reduced concurrency
        semaphore = asyncio.Semaphore(3)  # Reduced from 10 to 3 to avoid rate limits

        async def process_question(index: int, question: str) -> tuple[int, str, str]:
            async with semaphore:
                for attempt in range(3):
                    try:
                        retrieval_input = {"query": question, "agent_id": agent_id, "top_k": 3}
                        retrieved = await retrieve_from_kb(retrieval_input)
                        retrieved_chunks = retrieved.get("chunks", [])
                        
                        if not retrieved_chunks:
                            print(f"⚠️ Q{index}: No chunks retrieved for question: {question[:50]}...")
                            return (index, question, "I couldn't find relevant information to answer this question.")

                        max_context_chars = 3000
                        context = "\n".join(retrieved_chunks)[:max_context_chars]

                        print(f"✏️ Q{index}: Context preview: {context[:100]}...")
                        answer = await ask_gpt(context, question)
                        return (index, question, answer)
                        
                    except Exception as e:
                        print(f"⚠️ Q{index}: Attempt {attempt + 1} failed with error: {e}")
                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                return (index, question, "Sorry, I couldn't find relevant information to answer this question.")

        print(f"🧠 Parallel processing {len(questions)} questions...")
        
        if not questions:
            return {}
            
        # Add timeout for question processing
        try:
            tasks = [asyncio.create_task(process_question(i, q)) for i, q in enumerate(questions)]
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)  # 2 minute timeout
        except asyncio.TimeoutError:
            print("⚠️ Question processing timed out")
            raise Exception("Processing timed out. Please try with fewer questions or a smaller document.")

        # Step 3: Return sorted results
        results = {q: ans for _, q, ans in sorted(responses)}
        return results
        
    except Exception as e:
        print(f"❌ Error in process_documents_and_questions: {e}")
        raise