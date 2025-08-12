#!/usr/bin/env python3
"""
Quick test script for the Spam Detector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if the model can be loaded without authentication errors"""
    try:
        print("🧪 Testing Spam Classifier model loading...")
        from spam_model import SpamClassifier

        # Initialize classifier
        classifier = SpamClassifier()
        print("✅ SpamClassifier initialized successfully")

        # Test model loading
        classifier._load_model()
        print("✅ Model loaded successfully!")

        print("\n🎉 All tests passed! Your spam detector should work now.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_streamlit_import():
    """Test if Streamlit app can be imported"""
    try:
        print("\n🧪 Testing Streamlit app import...")
        import app
        print("✅ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running Spam Detector Tests")
    print("=" * 50)

    success = True

    # Test 1: Model loading
    success &= test_model_loading()

    # Test 2: Streamlit app import
    success &= test_streamlit_import()

    if success:
        print("\n🎉 All tests passed!")
        print("\n💡 To run your Spam Detector:")
        print("   streamlit run app.py")
    else:
        print("\n😞 Some tests failed. Please check the errors above.")
