#hf_model.py
import os
import asyncio
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from groq import Groq, APIError

# Initialize model lazily to avoid startup issues
_model = None

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

user_prompt = "Explain the importance of low-latency LLMs in 100 words."


def get_model():
    global _model
    if _model is None:
        try:
        
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

        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    return _model

async def ask_gpt(context: str, question: str) -> str:
    try:
        system_prompt = (
            """You are an insurance policy assistant. Using ONLY the provided text, give a comprehensive answer to the user's question. Make sure to include all relevant conditions, limits, waiting periods, and exceptions mentioned in the text. Do not add any information not present in the text. \n\nContext: [retrieved text]\n\nQuestion: [user question]\n\nComprehensive Answer:"""
        )

        user_prompt = f"""
        Context:
        {context}

        Question: {question}
        """

        model = get_model()
        
        # Run the synchronous model call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                stream=False,
                max_tokens=512,
            )
        )

        return response.choices[0].message.content.strip()
        
        # response = await loop.run_in_executor(
        #     None,
        #     lambda: model.invoke([
        #         SystemMessage(content=system_prompt),
        #         HumanMessage(content=user_prompt)
        #     ])
        # )

        # return response.content.strip()
    
    except Exception as e:
        print(f"Error in ask_gpt: {e}")
        return "Sorry, I couldn't process your question at the moment."
