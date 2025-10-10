#!/usr/bin/env python3
"""
Script to fix Hugging Face authentication issues
"""

import os
import sys
from pathlib import Path

def clear_huggingface_auth():
    """Clear potentially problematic HuggingFace authentication"""
    print("🔧 Clearing HuggingFace authentication...")

    # Check common HuggingFace token locations
    hf_token_locations = [
        # HuggingFace CLI token
        Path.home() / ".huggingface" / "token",
        # Environment variables
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        os.environ.get("HUGGINGFACE_TOKEN"),
    ]
	
    # Clear environment variables
    env_vars_to_clear = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]
    for var in env_vars_to_clear:
        if var in os.environ:
            print(f"  ❌ Clearing environment variable: {var}")
            del os.environ[var]
        else:
            print(f"  ✅ Environment variable not set: {var}")

    # Check token file
    hf_token_file = Path.home() / ".huggingface" / "token"
    if hf_token_file.exists():
        print(f"  ⚠️  HuggingFace token file exists at: {hf_token_file}")
        print("     This might be causing authentication issues.")
        response = input("  Do you want to temporarily rename it? (y/n): ")
        if response.lower() == 'y':
            backup_file = hf_token_file.with_suffix('.backup')
            hf_token_file.rename(backup_file)
            print(f"  ✅ Token file renamed to: {backup_file}")
    else:
        print(f"  ✅ No HuggingFace token file found")

    print("\n🚀 Authentication cleared! Try running your app again.")
    print("💡 The multilingual-e5-base model should work without authentication.")

def test_model_access():
    """Test if we can access the model without authentication"""
    try:
        print("\n🧪 Testing model access...")
        from transformers import AutoTokenizer

        # Try to load tokenizer without authentication
        tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-base",
            use_auth_token=False,
            trust_remote_code=False
        )
        print("  ✅ Successfully loaded tokenizer!")
        return True

    except Exception as e:
        print(f"  ❌ Failed to load tokenizer: {e}")
        return False

def main():
    print("🛠️  HuggingFace Authentication Fix Tool")
    print("=" * 50)

    # Clear authentication
    clear_huggingface_auth()

    # Test model access
    test_success = test_model_access()

    if test_success:
        print("\n🎉 Success! The model should now work properly.")
    else:
        print("\n💭 Alternative solutions:")
        print("1. Check your internet connection")
        print("2. Try using a different model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')")
        print("3. Install/update transformers: pip install --upgrade transformers")

if __name__ == "__main__":
    main()
