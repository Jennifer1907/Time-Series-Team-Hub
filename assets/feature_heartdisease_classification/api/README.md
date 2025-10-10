# Heart Disease Prediction API

A FastAPI-based REST API for predicting heart disease using a multimodal fusion model that combines numerical patient data with echocardiogram video features.

## Features

- **Single Prediction**: Predict heart disease for individual patient samples
- **Batch Prediction**: Process multiple predictions in one request
- **Health Check**: Monitor API status
- **Interactive Documentation**: Auto-generated Swagger UI

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn torch torchvision lightning numpy pandas scikit-learn imageio pillow
```

2. Ensure model files are present in `../models/`:
   - `best_fusion_model.pth`
   - `num_scaler_fold_1.pkl`
   - `cnn_scaler_fold_1.pkl`

3. Ensure data files are accessible:
   - Sulianova cardiovascular disease dataset
   - EchoNet features (saved as `data/echonet_features.npy`)

## Running the API

### Start the server:
```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
**GET** `/`
- Returns API information and available endpoints

### 2. Health Check
**GET** `/health`
- Check if the API and models are loaded correctly

### 3. Single Prediction
**POST** `/predict`

Request body:
```json
{
  "sample_index": 2,
  "video_index": 0
}
```

Response:
```json
{
  "prediction": 1,
  "prediction_label": "Disease",
  "disease_probability": 0.8745,
  "confidence": 0.8745,
  "sample_index": 2,
  "video_index": 0,
  "ground_truth": 1,
  "numerical_features_shape": [1, 28],
  "cnn_features_shape": [1, 64],
  "device": "cuda"
}
```

### 4. Batch Prediction
**POST** `/predict/batch`

Request body:
```json
[
  {"sample_index": 0, "video_index": 0},
  {"sample_index": 1, "video_index": 1},
  {"sample_index": 2, "video_index": 2}
]
```

Response:
```json
{
  "results": [
    {
      "sample_index": 0,
      "video_index": 0,
      "prediction": 0,
      "prediction_label": "No Disease",
      "disease_probability": 0.1234,
      "confidence": 0.8766,
      "ground_truth": 0
    },
    ...
  ],
  "errors": [],
  "total_processed": 3,
  "total_errors": 0
}
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

Visit `http://localhost:8000/redoc` for alternative documentation (ReDoc)

## Usage Examples

### Using cURL:

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sample_index": 2, "video_index": 0}'

# Health check
curl "http://localhost:8000/health"
```

### Using Python requests:

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"sample_index": 2, "video_index": 0}
)
result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['disease_probability']:.4f}")

# Batch prediction
batch_response = requests.post(
    "http://localhost:8000/predict/batch",
    json=[
        {"sample_index": 0, "video_index": 0},
        {"sample_index": 1, "video_index": 1}
    ]
)
batch_results = batch_response.json()
print(f"Processed: {batch_results['total_processed']}")
```

### Using JavaScript (fetch):

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    sample_index: 2,
    video_index: 0
  })
})
.then(response => response.json())
.then(data => console.log('Prediction:', data.prediction_label));
```

## API Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | int | Predicted class (0=No Disease, 1=Disease) |
| `prediction_label` | str | Human-readable label |
| `disease_probability` | float | Probability of disease (0.0-1.0) |
| `confidence` | float | Model confidence (max probability) |
| `sample_index` | int | Input patient sample index |
| `video_index` | int | Input video index |
| `ground_truth` | int/null | Actual label if available |
| `numerical_features_shape` | list | Shape of numerical features |
| `cnn_features_shape` | list/null | Shape of CNN features |
| `device` | str | Device used (cuda/cpu) |

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid indices)
- `404`: Not Found (data files missing)
- `500`: Internal Server Error (prediction failed)

## Configuration

Edit `main.py` to change:
- Host/port settings
- Model paths
- Feature extraction settings

## Troubleshooting

1. **"torch is not defined"**: Ensure `import torch` is in all required files
2. **Model size mismatch**: Check `numerical_dim` matches training (28 vs 13)
3. **Data not found**: Verify paths to datasets and saved features
4. **CUDA errors**: Set device to 'cpu' if GPU not available

## License

MIT License
