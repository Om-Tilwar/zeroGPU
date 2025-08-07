# test_api_keys.py
import os
import asyncio
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

async def test_single_api_key(key: str, key_name: str) -> dict:
    """Test a single API key"""
    if not key:
        return {"key_name": key_name, "status": "not_set", "error": "API key not provided"}
    
    try:
        client = Groq(api_key=key)
        
        # Make a simple test request
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=5,
            temperature=0.0
        )
        
        return {
            "key_name": key_name,
            "status": "valid",
            "key_preview": f"{key[:8]}...{key[-4:]}",
            "response_preview": response.choices[0].message.content.strip()[:20]
        }
        
    except Exception as e:
        error_str = str(e)
        if "invalid_api_key" in error_str.lower():
            status = "invalid"
        elif "rate_limit" in error_str.lower():
            status = "rate_limited"
        else:
            status = "error"
            
        return {
            "key_name": key_name,
            "status": status,
            "key_preview": f"{key[:8]}...{key[-4:]}" if key else "None",
            "error": error_str
        }

async def test_all_api_keys():
    """Test all configured API keys"""
    api_keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GROQ_API_KEY_1": os.getenv("GROQ_API_KEY_1"),
        "GROQ_API_KEY_2": os.getenv("GROQ_API_KEY_2"),
        "GROQ_API_KEY_3": os.getenv("GROQ_API_KEY_3"),
        "GROQ_API_KEY_4": os.getenv("GROQ_API_KEY_4"),
        "GROQ_API_KEY_5": os.getenv("GROQ_API_KEY_5"),
    }
    
    print("🔍 Testing Groq API Keys...")
    print("=" * 50)
    
    results = []
    for key_name, key_value in api_keys.items():
        if key_value:  # Only test keys that are set
            result = await test_single_api_key(key_value, key_name)
            results.append(result)
            
            # Print immediate result
            status_emoji = {
                "valid": "✅",
                "invalid": "❌",
                "rate_limited": "⏰",
                "error": "⚠️",
                "not_set": "➖"
            }
            
            emoji = status_emoji.get(result["status"], "❓")
            print(f"{emoji} {key_name}: {result['status'].upper()}")
            
            if result["status"] == "valid":
                print(f"   Preview: {result['key_preview']}")
                print(f"   Response: {result.get('response_preview', 'N/A')}")
            elif "error" in result:
                print(f"   Error: {result['error']}")
            
            print()
    
    # Summary
    valid_keys = [r for r in results if r["status"] == "valid"]
    invalid_keys = [r for r in results if r["status"] == "invalid"]
    
    print("=" * 50)
    print(f"📊 SUMMARY:")
    print(f"✅ Valid keys: {len(valid_keys)}")
    print(f"❌ Invalid keys: {len(invalid_keys)}")
    print(f"🔑 Total tested: {len(results)}")
    
    if len(valid_keys) == 0:
        print("\n⚠️  WARNING: No valid API keys found!")
        print("Please check your API keys and ensure they are correct.")
    elif len(invalid_keys) > 0:
        print(f"\n🚨 Found {len(invalid_keys)} invalid API keys that need to be fixed:")
        for key_info in invalid_keys:
            print(f"   - {key_info['key_name']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_all_api_keys())