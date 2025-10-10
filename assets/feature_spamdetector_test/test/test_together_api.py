#!/usr/bin/env python3
"""
Test script to verify Together AI API key functionality
"""

import requests
import json

def test_together_ai_api(api_key):
    """Test if Together AI API key is working"""

    if not api_key or api_key.strip() == "":
        print("❌ No API key provided")
        return False

    # Test endpoint
    url = "https://api.together.xyz/v1/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Simple test payload
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": "Generate a simple spam message example:",
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        print("🔍 Testing Together AI API...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                print("✅ Together AI API is working!")
                print(f"📝 Sample response: {result['choices'][0]['text'][:100]}...")
                return True
            else:
                print("⚠️ API responded but no content generated")
                print(f"Response: {result}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ API request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Test the provided API key
    TOGETHER_AI_API_KEY = "a4910347ea0b1f86be877cd19899dd0bd3f855487a0b80eb611a64c0abf7a782"

    print("=== Together AI API Key Test ===")
    print(f"API Key: {TOGETHER_AI_API_KEY[:20]}...{TOGETHER_AI_API_KEY[-10:]}")

    if TOGETHER_AI_API_KEY == None or TOGETHER_AI_API_KEY == "":
        TOGETHER_AI_API_KEY = input("Nhập Together.ai API key (nhấn Enter để bỏ qua): ").strip()

    if TOGETHER_AI_API_KEY:
        is_working = test_together_ai_api(TOGETHER_AI_API_KEY)
        if is_working:
            print("\n🎉 API key is valid and working!")
        else:
            print("\n💥 API key test failed!")
    else:
        print("⏭️ Skipping API test (no key provided)")
