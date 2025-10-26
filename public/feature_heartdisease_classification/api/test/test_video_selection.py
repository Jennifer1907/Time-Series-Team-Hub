"""
Test script to demonstrate video selection features
"""
import requests
import json

API_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def test_health():
    """Test health endpoint"""
    print_section("1. TESTING HEALTH ENDPOINT")
    response = requests.get(f"{API_URL}/health")
    print(json.dumps(response.json(), indent=2))

def test_list_videos():
    """Test listing videos"""
    print_section("2. LISTING FIRST 10 VIDEOS")
    response = requests.get(f"{API_URL}/videos?limit=10&offset=0")
    data = response.json()

    print(f"Total videos available: {data['total_videos']}")
    print(f"Videos path: {data['videos_path']}\n")

    print("First 10 videos:")
    for video in data['videos']:
        status = "✓" if video['exists'] else "✗"
        print(f"  {status} [{video['index']:4d}] {video['filename_with_ext']:30s} "
              f"EF: {video['ejection_fraction']:5.1f}% "
              f"Frames: {video['number_of_frames']}")

    return data['videos']

def test_search_videos():
    """Test searching videos"""
    print_section("3. SEARCHING VIDEOS")
    search_query = "0X10"
    response = requests.get(f"{API_URL}/videos/search?query={search_query}&limit=5")
    data = response.json()

    print(f"Search query: '{search_query}'")
    print(f"Matches found: {data['total_matches']}\n")

    for video in data['matches']:
        status = "✓" if video['exists'] else "✗"
        print(f"  {status} [{video['index']:4d}] {video['filename_with_ext']:30s} "
              f"EF: {video['ejection_fraction']:5.1f}%")

    return data['matches']

def test_prediction_with_index(video_index):
    """Test prediction using video index"""
    print_section(f"4. PREDICTION WITH VIDEO INDEX ({video_index})")

    patient_data = {
        "id": 12345,
        "age": 50,  # in years (will be converted to days)
        "gender": 2,
        "height": 168,
        "weight": 62.0,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cardio": 0
    }

    request_data = {
        "patient_data": patient_data,
        "video_index": video_index
    }

    print("Request:")
    print(json.dumps(request_data, indent=2))

    response = requests.post(f"{API_URL}/predict", json=request_data)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction: {result['prediction_label']}")
        print(f"  Disease Probability: {result['disease_probability']:.2%}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Video Index Used: {result['video_index']}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.json())

def test_prediction_with_filename(video_filename):
    """Test prediction using video filename"""
    print_section(f"5. PREDICTION WITH VIDEO FILENAME ({video_filename})")

    patient_data = {
        "id": 67890,
        "age": 45,
        "gender": 1,
        "height": 165,
        "weight": 70.0,
        "ap_hi": 120,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 1,
        "alco": 0,
        "active": 1,
        "cardio": 1
    }

    request_data = {
        "patient_data": patient_data,
        "video_filename": video_filename
    }

    print("Request:")
    print(json.dumps(request_data, indent=2))

    response = requests.post(f"{API_URL}/predict", json=request_data)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction: {result['prediction_label']}")
        print(f"  Disease Probability: {result['disease_probability']:.2%}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Video Index Used: {result['video_index']}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.json())

def test_batch_prediction():
    """Test batch prediction with mixed video selection"""
    print_section("6. BATCH PREDICTION (MIXED INDEX & FILENAME)")

    patient1 = {
        "id": 1,
        "age": 50,
        "gender": 2,
        "height": 168,
        "weight": 62.0,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cardio": 0
    }

    patient2 = {
        "id": 2,
        "age": 60,
        "gender": 1,
        "height": 160,
        "weight": 75.0,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 1,
        "alco": 1,
        "active": 0,
        "cardio": 1
    }

    requests_data = [
        {
            "patient_data": patient1,
            "video_index": 0  # Using index
        },
        {
            "patient_data": patient2,
            "video_filename": "0X1005D03EED19C65B.avi"  # Using filename
        }
    ]

    print("Batch Request:")
    print(json.dumps(requests_data, indent=2))

    response = requests.post(f"{API_URL}/predict/batch", json=requests_data)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Total Processed: {result['total_processed']}")
        print(f"  Total Errors: {result['total_errors']}\n")

        for i, res in enumerate(result['results'], 1):
            print(f"  Result {i}:")
            print(f"    Patient ID: {res['patient_id']}")
            video_info = res.get('video_filename') or f"index {res['video_index']}"
            print(f"    Video: {video_info}")
            print(f"    Prediction: {res['prediction_label']}")
            print(f"    Probability: {res['disease_probability']:.2%}\n")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.json())


if __name__ == "__main__":
    try:
        print("="*70)
        print("VIDEO SELECTION API TEST SUITE")
        print("="*70)
        print("API URL:", API_URL)
        print("\nMake sure the API is running: python main.py")
        input("Press Enter to continue...")

        # Test 1: Health check
        test_health()

        # Test 2: List videos
        videos = test_list_videos()

        # Test 3: Search videos
        matches = test_search_videos()

        # Test 4: Prediction with video index
        if videos:
            test_prediction_with_index(videos[2]['index'])

        # Test 5: Prediction with video filename
        if videos:
            test_prediction_with_filename(videos[0]['filename_with_ext'])

        # Test 6: Batch prediction
        test_batch_prediction()

        print_section("ALL TESTS COMPLETED!")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API is running:")
        print('  cd "d:\\CODE\\Machine-Learning-Deep-Learning-2025\\AIO2025 - Mein Code\\Module5\\api"')
        print("  python main.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
