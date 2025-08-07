# config/settings.py

from dotenv import load_dotenv
import os
from typing import Optional, List

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Configuration settings loaded from environment variables.
    Handles validation and provides defaults where appropriate.
    """
    
    # Hugging Face Configuration
    HUGGINGFACEHUB_ACCESS_TOKEN: Optional[str] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    
    # Backend Authentication
    BACKEND_BEARER_TOKEN: Optional[str] = os.getenv("BACKEND_BEARER_TOKEN")
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: Optional[str] = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_HOST_NAME: Optional[str] = os.getenv("PINECONE_HOST_NAME")
    
    # Groq API Keys (Multiple for load balancing)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_API_KEY_1: Optional[str] = os.getenv("GROQ_API_KEY_1")
    GROQ_API_KEY_2: Optional[str] = os.getenv("GROQ_API_KEY_2")
    GROQ_API_KEY_3: Optional[str] = os.getenv("GROQ_API_KEY_3")
    
    # Optional: Additional Groq API Keys for high-volume usage
    GROQ_API_KEY_4: Optional[str] = os.getenv("GROQ_API_KEY_4")
    GROQ_API_KEY_5: Optional[str] = os.getenv("GROQ_API_KEY_5")
    
    # Model Configuration
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "75"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    TOP_P: float = float(os.getenv("TOP_P", "1.0"))
    
    # RAG Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "3000"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    
    # Retry and Backoff Configuration
    MAX_RETRY_ATTEMPTS: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "4"))
    BASE_BACKOFF_DELAY: int = int(os.getenv("BASE_BACKOFF_DELAY", "2"))  # seconds
    
    # Default namespace for Pinecone
    DEFAULT_NAMESPACE: str = os.getenv("DEFAULT_NAMESPACE", "default_namespace")
    
    def get_groq_api_keys(self) -> List[str]:
        """
        Returns a list of all configured Groq API keys, filtering out None values.
        """
        keys = [
            self.GROQ_API_KEY,
            self.GROQ_API_KEY_1,
            self.GROQ_API_KEY_2,
            self.GROQ_API_KEY_3,
            self.GROQ_API_KEY_4,
            self.GROQ_API_KEY_5,
        ]
        return [key for key in keys if key and key.strip()]
    
    def validate(self) -> None:
        """
        Validates that all required environment variables are set.
        Raises ValueError if any required variables are missing.
        """
        # Core required variables
        required_vars = {
            "HUGGINGFACEHUB_ACCESS_TOKEN": self.HUGGINGFACEHUB_ACCESS_TOKEN,
            "PINECONE_API_KEY": self.PINECONE_API_KEY,
            "PINECONE_INDEX_NAME": self.PINECONE_INDEX_NAME,
        }
        
        missing = [var for var, value in required_vars.items() if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Validate at least one Groq API key exists
        groq_keys = self.get_groq_api_keys()
        if not groq_keys:
            raise ValueError(
                "At least one GROQ_API_KEY must be provided. "
                "Set GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, etc."
            )
        
        # Validate numeric configurations
        if self.MAX_TOKENS <= 0:
            raise ValueError("MAX_TOKENS must be greater than 0")
        
        if not (0.0 <= self.TEMPERATURE <= 2.0):
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")
        
        if not (0.0 <= self.TOP_P <= 1.0):
            raise ValueError("TOP_P must be between 0.0 and 1.0")
        
        if self.DEFAULT_TOP_K <= 0:
            raise ValueError("DEFAULT_TOP_K must be greater than 0")
        
        if self.MAX_CONTEXT_LENGTH <= 0:
            raise ValueError("MAX_CONTEXT_LENGTH must be greater than 0")
        
        if self.MAX_CONCURRENT_REQUESTS <= 0:
            raise ValueError("MAX_CONCURRENT_REQUESTS must be greater than 0")
        
        if self.MAX_RETRY_ATTEMPTS < 0:
            raise ValueError("MAX_RETRY_ATTEMPTS must be non-negative")
        
        if self.BASE_BACKOFF_DELAY <= 0:
            raise ValueError("BASE_BACKOFF_DELAY must be greater than 0")
        
        print("✅ All configuration settings validated successfully")
    
    def get_config_summary(self) -> dict:
        """
        Returns a summary of current configuration (without sensitive data).
        Useful for debugging and logging.
        """
        groq_keys = self.get_groq_api_keys()
        
        return {
            "groq_model": self.GROQ_MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "groq_api_keys_count": len(groq_keys),
            "default_top_k": self.DEFAULT_TOP_K,
            "max_context_length": self.MAX_CONTEXT_LENGTH,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "max_retry_attempts": self.MAX_RETRY_ATTEMPTS,
            "base_backoff_delay": self.BASE_BACKOFF_DELAY,
            "default_namespace": self.DEFAULT_NAMESPACE,
            "pinecone_index_name": self.PINECONE_INDEX_NAME,
            "has_huggingface_token": bool(self.HUGGINGFACEHUB_ACCESS_TOKEN),
            "has_backend_bearer_token": bool(self.BACKEND_BEARER_TOKEN),
            "has_pinecone_config": bool(self.PINECONE_API_KEY and self.PINECONE_INDEX_NAME),
        }
    
    def print_config_summary(self) -> None:
        """
        Prints a formatted configuration summary.
        """
        summary = self.get_config_summary()
        print("\n" + "="*50)
        print("🔧 CONFIGURATION SUMMARY")
        print("="*50)
        print(f"🤖 Groq Model: {summary['groq_model']}")
        print(f"🔑 Groq API Keys: {summary['groq_api_keys_count']} configured")
        print(f"🎯 Max Tokens: {summary['max_tokens']}")
        print(f"🌡️  Temperature: {summary['temperature']}")
        print(f"📊 Top P: {summary['top_p']}")
        print(f"🔍 Default Top K: {summary['default_top_k']}")
        print(f"📏 Max Context Length: {summary['max_context_length']}")
        print(f"🚀 Max Concurrent Requests: {summary['max_concurrent_requests']}")
        print(f"🔄 Max Retry Attempts: {summary['max_retry_attempts']}")
        print(f"⏰ Base Backoff Delay: {summary['base_backoff_delay']}s")
        print(f"📂 Default Namespace: {summary['default_namespace']}")
        print(f"🗃️  Pinecone Index: {summary['pinecone_index_name']}")
        print(f"🤗 HuggingFace Token: {'✅' if summary['has_huggingface_token'] else '❌'}")
        print(f"🔐 Backend Bearer Token: {'✅' if summary['has_backend_bearer_token'] else '❌'}")
        print(f"📌 Pinecone Config: {'✅' if summary['has_pinecone_config'] else '❌'}")
        print("="*50 + "\n")

# Initialize settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
    print("🎉 Settings loaded and validated successfully!")
    
    # Print configuration summary if in debug mode
    if os.getenv("DEBUG", "false").lower() == "true":
        settings.print_config_summary()
        
except Exception as e:
    print(f"❌ Configuration error: {e}")
    raise

# Export commonly used values for convenience
GROQ_API_KEYS = settings.get_groq_api_keys()
GROQ_MODEL = settings.GROQ_MODEL
MAX_TOKENS = settings.MAX_TOKENS
DEFAULT_TOP_K = settings.DEFAULT_TOP_K
MAX_CONTEXT_LENGTH = settings.MAX_CONTEXT_LENGTH