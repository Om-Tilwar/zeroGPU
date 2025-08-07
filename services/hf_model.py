# services/hf_model.py

import os
import asyncio
import re
from groq import Groq, APIError
from config.settings import settings

# Initialize multiple API keys from settings for round-robin usage
GROQ_API_KEYS = [
    settings.GROQ_API_KEY,
    settings.GROQ_API_KEY_1,
    settings.GROQ_API_KEY_2,
    settings.GROQ_API_KEY_3
]

# Filter out any keys that are not set
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    raise ValueError("At least one GROQ_API_KEY must be provided in the environment variables.")

print(f"✅ Initialized with {len(GROQ_API_KEYS)} Groq API key(s).")

# Define the model to be used
PRIMARY_MODEL = "llama3-8b-8192"

# Initialize a client for each API key
clients = [Groq(api_key=key) for key in GROQ_API_KEYS]

# --- MODIFICATION START ---

# Global counter and lock for thread-safe round-robin
request_counter = 0
client_lock = asyncio.Lock()

async def get_next_client():
    """
    Atomically gets the next client in the rotation. This is now thread-safe.
    """
    global request_counter
    async with client_lock:
        client_index = request_counter % len(clients)
        client = clients[client_index]
        current_request_num = request_counter # Capture for logging
        request_counter += 1
        return client, client_index, current_request_num

# --- MODIFICATION END ---


def is_rate_limit_error(error: Exception) -> bool:
    """Checks if an exception is a rate limit error."""
    error_str = str(error).lower()
    return (
        "rate limit" in error_str or
        "429" in error_str or
        "quota exceeded" in error_str or
        "too many requests" in error_str
    )

async def make_groq_request(client: Groq, client_index: int, request_num: int, system_prompt: str, user_prompt: str, max_tokens: int = 75):
    # Modified to accept request_num for accurate logging
    print(f"API key #{client_index + 1} (Request #{request_num + 1}) for model {PRIMARY_MODEL}")
    
    loop = asyncio.get_event_loop()
    
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            stream=False,
            max_tokens=max_tokens,
        )
    )
    
    return response.choices[0].message.content.strip()

async def ask_gpt(context: str, question: str) -> str:
    """
    Main function with round-robin, fallback, and backoff logic.
    """
    system_prompt = (
        "You are an insurance policy assistant. Using ONLY the provided text, give a comprehensive answer to the user's question. Make sure to include all relevant conditions, limits, waiting periods, and exceptions mentioned in the text. Do not add any information not present in the text. \n\nContext: [retrieved text]\n\nQuestion: [user question]\n\nComprehensive Answer:"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # --- MODIFICATION START ---
    # Get the next client atomically
    primary_client, primary_index, request_num = await get_next_client()
    # --- MODIFICATION END ---
    
    try:
        # First attempt with the primary client
        response = await make_groq_request(primary_client, primary_index, request_num, system_prompt, user_prompt)
        return response
        
    except Exception as e:
        print(f"API key #{primary_index + 1} failed: {e}")
        
        if is_rate_limit_error(e):
            print(f"Rate limit hit on key #{primary_index + 1}. Trying fallback keys...")
            
            # Try all other available API keys
            for i in range(len(clients)):
                if i == primary_index:
                    continue  # Skip the key that just failed
                
                # We need to get the next request number for fallback attempts as well
                fallback_client, fallback_index, fallback_req_num = await get_next_client()

                try:
                    print(f"Trying fallback API key #{fallback_index + 1}...")
                    response = await make_groq_request(fallback_client, fallback_index, fallback_req_num, system_prompt, user_prompt)
                    return response
                except Exception as fallback_error:
                    print(f"Fallback key #{fallback_index + 1} also failed: {fallback_error}")
                    if not is_rate_limit_error(fallback_error):
                        return "Sorry, a non-recoverable error occurred."

            # If all keys are rate-limited, start exponential backoff
            print("All API keys appear to be rate-limited. Starting exponential backoff...")
            for attempt in range(4): # Retry up to 4 times
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s, 16s
                print(f"Waiting for {wait_time}s before retry attempt {attempt + 1}...")
                await asyncio.sleep(wait_time)
                
                # Retry with the next available client from rotation
                retry_client, retry_index, retry_req_num = await get_next_client()
                try:
                    response = await make_groq_request(retry_client, retry_index, retry_req_num, system_prompt, user_prompt)
                    return response
                except Exception as retry_error:
                    print(f"Retry attempt {attempt + 1} failed: {retry_error}")

            return "I'm temporarily unable to process your question due to high API demand. Please try again in a few moments."
    
    # Handle other non-rate-limit errors
    return "Sorry, I couldn't process your question due to an unexpected error."