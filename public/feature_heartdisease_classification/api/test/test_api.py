"""
Test script for the Heart Disease Prediction API
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_prediction():
    """Test single prediction"""
    print("Testing single prediction...")
    payload = {
        "sample_index": 2,
        "video_index": 0
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\n✓ Prediction: {result['prediction_label']}")
        print(f"✓ Disease Probability: {result['disease_probability']:.4f}")
        print(f"✓ Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction...")
    payload = [
        {"sample_index": 0, "video_index": 0},
        {"sample_index": 1, "video_index": 1},
        {"sample_index": 2, "video_index": 2}
    ]
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Total processed: {result['total_processed']}")
        print(f"Total errors: {result['total_errors']}")

        if result['results']:
            print("\nResults:")
            for r in result['results']:
                print(f"  Sample {r['sample_index']}: {r['prediction_label']} (prob: {r['disease_probability']:.4f})")

        if result['errors']:
            print("\nErrors:")
            for e in result['errors']:
                print(f"  {e}")
    else:
        print(f"Error: {response.text}")
    print()

def test_invalid_request():
    """Test error handling with invalid request"""
    print("Testing error handling (invalid index)...")
    payload = {
        "sample_index": -1,  # Invalid negative index
        "video_index": 0
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()

if __name__ == "__main__":
    print("="*70)
    print("HEART DISEASE PREDICTION API - TEST SUITE")
    print("="*70)
    print()

    try:
        # Run all tests
        test_root()
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_invalid_request()

        print("="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API server")
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
