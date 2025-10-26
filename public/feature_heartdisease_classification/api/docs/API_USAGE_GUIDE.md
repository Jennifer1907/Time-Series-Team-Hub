# Heart Disease Prediction API - Usage Guide

## Overview
The API has been updated to accept patient medical data directly instead of requiring a `sample_index`. This makes the API production-ready for real-world deployment.

## Changes Made

### 1. Request Format
**OLD Format (sample_index based):**
```json
{
  "sample_index": 2,
  "video_index": 0
}
```

**NEW Format (patient_data based):**
```json
{
  "patient_data": {
    "id": 12345,
    "age": 18393,
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
  },
  "video_index": 0
}
```

### 2. Patient Data Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | int | Patient ID | Any integer |
| `age` | int | Age in days | >= 0 |
| `gender` | int | Gender | 1 (female) or 2 (male) |
| `height` | float | Height in cm | > 0 |
| `weight` | float | Weight in kg | > 0 |
| `ap_hi` | int | Systolic blood pressure | >= 0 |
| `ap_lo` | int | Diastolic blood pressure | >= 0 |
| `cholesterol` | int | Cholesterol level | 1 (normal), 2 (above normal), 3 (well above normal) |
| `gluc` | int | Glucose level | 1 (normal), 2 (above normal), 3 (well above normal) |
| `smoke` | int | Smoking status | 0 (no) or 1 (yes) |
| `alco` | int | Alcohol intake | 0 (no) or 1 (yes) |
| `active` | int | Physical activity | 0 (no) or 1 (yes) |
| `cardio` | int | Cardiovascular disease (ground truth) | 0 (no) or 1 (yes) |

## API Endpoints

### 1. Single Prediction - `/predict` (POST)

**Request:**
```json
{
  "patient_data": {
    "id": 12345,
    "age": 18393,
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
  },
  "video_index": 0
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Disease",
  "disease_probability": 0.1234,
  "confidence": 0.8766,
  "patient_id": 12345,
  "video_index": 0,
  "ground_truth": 0,
  "numerical_features_shape": [1, 28],
  "cnn_features_shape": [1, 64],
  "device": "cuda"
}
```

### 2. Batch Prediction - `/predict/batch` (POST)

**Request:**
```json
[
  {
    "patient_data": {
      "id": 12345,
      "age": 18393,
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
    },
    "video_index": 0
  },
  {
    "patient_data": {
      "id": 67890,
      "age": 20000,
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
    },
    "video_index": 1
  }
]
```

**Response:**
```json
{
  "results": [
    {
      "patient_id": 12345,
      "video_index": 0,
      "prediction": 0,
      "prediction_label": "No Disease",
      "disease_probability": 0.1234,
      "confidence": 0.8766,
      "ground_truth": 0
    },
    {
      "patient_id": 67890,
      "video_index": 1,
      "prediction": 1,
      "prediction_label": "Disease",
      "disease_probability": 0.8901,
      "confidence": 0.8901,
      "ground_truth": 1
    }
  ],
  "errors": [],
  "total_processed": 2,
  "total_errors": 0
}
```

## Usage Examples

### Python with requests
```python
import requests

# Single prediction
patient_data = {
    "id": 12345,
    "age": 18393,
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

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "patient_data": patient_data,
        "video_index": 0
    }
)

result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Disease Probability: {result['disease_probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "id": 12345,
      "age": 18393,
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
    },
    "video_index": 0
  }'
```

## Running the API

1. **Start the server:**
   ```bash
   cd "d:\CODE\Machine-Learning-Deep-Learning-2025\AIO2025 - Mein Code\Module5\api"
   python main.py
   ```

2. **Access interactive documentation:**
   - Open browser: `http://localhost:8000/docs`
   - Try out the API with the interactive Swagger UI

3. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

## Files Modified

1. **`main.py`**:
   - Added `PatientData` Pydantic model with validation
   - Updated `PredictionRequest` to use `patient_data` field
   - Updated `PredictionResponse` to use `patient_id` instead of `sample_index`
   - Modified `/predict` and `/predict/batch` endpoints to process patient data

2. **`utility/predict.py`**:
   - Changed `sample_index` parameter to `sample_json`
   - Updated function to pass patient data as dictionary to preprocessing pipeline

3. **`utility/preprocessing.py`**:
   - Added pandas import for DataFrame creation
   - Changed `sample_index` parameter to `sample_json`
   - Modified to convert JSON patient data to DataFrame for processing
   - Updated `sample_info` to use `patient_id` instead of `sample_index`

## Benefits of This Approach

1. **Production Ready**: No need to maintain a static dataset with indices
2. **Flexible**: Accept any patient data in real-time
3. **Validated**: Pydantic models ensure data integrity
4. **Type Safe**: Proper type hints and validation constraints
5. **Documented**: Auto-generated API docs at `/docs` endpoint
6. **Scalable**: Easy to integrate with databases and external systems

## Notes

- The `cardio` field in patient data is optional and represents the ground truth (if known)
- `video_index` still uses index-based selection from the EchoNet dataset
- All numerical features are automatically preprocessed and scaled using the trained scalers
- The API automatically handles feature engineering on the input data
