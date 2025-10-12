from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from utility import fusion_model, num_scaler, cnn_scaler, predict_single_sample

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using multimodal fusion model",
    version="1.0.0"
)

# Patient Data model
class PatientData(BaseModel):
    id: int = Field(description="Patient ID")
    age: int = Field(description="Age in days", ge=0)
    gender: int = Field(description="Gender (1=female, 2=male)", ge=1, le=2)
    height: float = Field(description="Height in cm", gt=0)
    weight: float = Field(description="Weight in kg", gt=0)
    ap_hi: int = Field(description="Systolic blood pressure", ge=0)
    ap_lo: int = Field(description="Diastolic blood pressure", ge=0)
    cholesterol: int = Field(description="Cholesterol level (1=normal, 2=above normal, 3=well above normal)", ge=1, le=3)
    gluc: int = Field(description="Glucose level (1=normal, 2=above normal, 3=well above normal)", ge=1, le=3)
    smoke: int = Field(description="Smoking (0=no, 1=yes)", ge=0, le=1)
    alco: int = Field(description="Alcohol intake (0=no, 1=yes)", ge=0, le=1)
    active: int = Field(description="Physical activity (0=no, 1=yes)", ge=0, le=1)
    cardio: int = Field(default=0, description="Target variable - presence of cardiovascular disease (0=no, 1=yes)", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
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
        }

# Request model
class PredictionRequest(BaseModel):
    patient_data: PatientData = Field(description="Patient medical data")
    video_index: int = Field(default=0, description="Index of video in EchoNet dataset", ge=0)

# Response model
class PredictionResponse(BaseModel):
    prediction: int = Field(description="Predicted class (0=No Disease, 1=Disease)")
    prediction_label: str = Field(description="Human-readable prediction label")
    disease_probability: float = Field(description="Probability of disease (0.0 to 1.0)")
    confidence: float = Field(description="Confidence score (max probability)")
    patient_id: int = Field(description="Patient ID")
    video_index: int
    ground_truth: Optional[int] = Field(description="Actual label if available")
    numerical_features_shape: list
    cnn_features_shape: Optional[list]
    device: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": fusion_model is not None,
        "scalers_loaded": num_scaler is not None and cnn_scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict heart disease for a given patient data and video.

    Args:
        request: PredictionRequest containing patient_data and video_index

    Returns:
        PredictionResponse with prediction results
    """
    try:
        # Convert PatientData to dict
        patient_json = request.patient_data.model_dump()

        # Make prediction
        prediction, probability, info = predict_single_sample(
            sample_json=patient_json,
            video_index=request.video_index,
            fusion_model=fusion_model,
            num_scaler=num_scaler,
            cnn_scaler=cnn_scaler
        )

        # Check if prediction was successful
        if prediction is None or probability is None:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed - model might not be loaded properly"
            )

        # Calculate confidence (max probability)
        confidence = max(float(probability), 1.0 - float(probability))

        # Prepare response
        response = PredictionResponse(
            prediction=prediction,
            prediction_label="Disease" if prediction == 1 else "No Disease",
            disease_probability=probability,
            confidence=confidence,
            patient_id=request.patient_data.id,
            video_index=request.video_index,
            ground_truth=info.get('ground_truth_label'),
            numerical_features_shape=list(info['numerical_features_shape']),
            cnn_features_shape=list(info['cnn_features_shape']) if info['cnn_features_shape'] is not None else None,
            device=info['device']
        )

        return response

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {str(e)}"
        )
    except IndexError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index provided: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """
    Make predictions for multiple samples.

    Args:
        requests: List of PredictionRequest objects, each containing patient_data and video_index

    Returns:
        List of prediction results
    """
    results = []
    errors = []

    for idx, request in enumerate(requests):
        try:
            # Convert PatientData to dict
            patient_json = request.patient_data.model_dump()

            prediction, probability, info = predict_single_sample(
                sample_json=patient_json,
                video_index=request.video_index,
                fusion_model=fusion_model,
                num_scaler=num_scaler,
                cnn_scaler=cnn_scaler
            )

            if prediction is not None and probability is not None:
                confidence = max(float(probability), 1.0 - float(probability))
                results.append({
                    "patient_id": request.patient_data.id,
                    "video_index": request.video_index,
                    "prediction": prediction,
                    "prediction_label": "Disease" if prediction == 1 else "No Disease",
                    "disease_probability": probability,
                    "confidence": confidence,
                    "ground_truth": info.get('ground_truth_label')
                })
            else:
                errors.append({
                    "index": idx,
                    "patient_id": request.patient_data.id,
                    "error": "Prediction failed"
                })
        except Exception as e:
            errors.append({
                "index": idx,
                "patient_id": request.patient_data.id,
                "error": str(e)
            })

    return {
        "results": results,
        "errors": errors,
        "total_processed": len(results),
        "total_errors": len(errors)
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )