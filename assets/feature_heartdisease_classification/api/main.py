from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union
import uvicorn
from utility import fusion_model, num_scaler, cnn_scaler, predict_single_sample
from utility.load_data import load_echonet_with_kagglehub

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using multimodal fusion model",
    version="1.0.0"
)

# Patient Data model
class PatientData(BaseModel):
    id: int = Field(description="Patient ID")
    age: int = Field(description="Age in year", ge=0)
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

    @field_validator('age')
    @classmethod
    def convert_age_to_days(cls, age_years: int) -> int:
        """
        Convert age from years to days (as required by the model).

        NOTE: This validator automatically converts user-provided age in YEARS
        to DAYS because the underlying ML model was trained on age in days.
        Example: 50 years → 18,250 days (50 * 365)

        Args:
            age_years: Age in years (0-120)

        Returns:
            Age in days (age_years * 365)
        """
        age_days = age_years * 365
        return age_days

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "age": 18,
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
    video_index: Optional[int] = Field(default=None, description="Index of video in EchoNet dataset (use either video_index or video_filename)", ge=0)
    video_filename: Optional[str] = Field(default=None, description="Filename of video (e.g., '0X1005D03EED19C65B.avi'). Use either video_index or video_filename")

    @field_validator('video_filename')
    @classmethod
    def validate_video_selection(cls, v, info):
        """
        Ensure either video_index or video_filename is provided, but not both.

        Validation Logic:
        - If both are None: defaults to video_index=0 (first video)
        - If video_filename provided: uses that (index will be looked up)
        - If video_index provided: uses that directly
        - If both provided: video_filename takes priority

        Args:
            v: The video_filename value being validated
            info: ValidationInfo containing other field values

        Returns:
            The validated video_filename (or None if using index)
        """
        video_index = info.data.get('video_index')

        # If neither is provided, default to video_index=0
        if v is None and video_index is None:
            info.data['video_index'] = 0
            return None

        return v

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

def get_video_index_from_filename(filename: str):
    """
    Get video index from filename by searching the EchoNet FileList.

    This function loads the EchoNet dataset metadata and searches for a matching
    filename, returning its corresponding index. This allows users to reference
    videos by meaningful filenames instead of memorizing numeric indices.

    Args:
        filename: Video filename (with or without .avi extension)
                 Examples: "0X1005D03EED19C65B.avi" or "0X1005D03EED19C65B"

    Returns:
        int: The index of the video in the EchoNet dataset (0-10029)

    Raises:
        HTTPException 500: If dataset fails to load
        HTTPException 404: If filename not found in dataset

    Example:
        >>> get_video_index_from_filename("0X1005D03EED19C65B.avi")
        0
    """
    try:
        echonet_filelist, _, _ = load_echonet_with_kagglehub()
        if echonet_filelist is None:
            raise HTTPException(status_code=500, detail="Failed to load EchoNet dataset")

        filename_without_ext = filename.replace('.avi', '')

        # Find the index by searching through the FileList
        # Note: iterrows() returns (index, row) tuples
        for idx, row in echonet_filelist.iterrows():
            if row['FileName'] == filename_without_ext or row['FileName'] == filename:
                return idx

        # If no match found, return None
        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding video: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/predict/batch": "POST - Make batch predictions",
            "/videos": "GET - List available videos",
            "/videos/search": "GET - Search videos by filename pattern",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        },
        "note": "You can use either 'video_index' (0-10029) or 'video_filename' (e.g., '0X1005D03EED19C65B.avi') in prediction requests"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": fusion_model is not None,
        "scalers_loaded": num_scaler is not None and cnn_scaler is not None
    }

@app.get("/videos")
async def list_videos(limit: int = 50, offset: int = 0):
    """
    List available EchoNet videos with pagination.

    Args:
        limit: Maximum number of videos to return (default: 50, max: 1000)
        offset: Number of videos to skip (default: 0)

    Returns:
        List of video information including index, filename, and metadata
    """
    try:
        # Validate limit to prevent excessive memory usage
        # Max 1000 videos per request to balance performance and usability
        if limit > 1000:
            limit = 1000

        echonet_filelist, _, videos_path = load_echonet_with_kagglehub()

        if echonet_filelist is None:
            raise HTTPException(status_code=500, detail="Failed to load EchoNet dataset")

        # Get subset of videos using pandas iloc for efficient slicing
        # This enables pagination: offset=0, limit=50 gets videos 0-49
        total_videos = len(echonet_filelist)
        videos_subset = echonet_filelist.iloc[offset:offset+limit]

        # Build response with video metadata
        # Each video includes clinical metadata from EchoNet-Dynamic dataset
        videos = []
        for idx, row in videos_subset.iterrows():
            video_filename = row['FileName']
            # Construct full path: videos_path/filename.avi
            video_path = videos_path / f"{video_filename}.avi"

            videos.append({
                "index": int(idx),  # Original index in full dataset
                "filename": video_filename,  # Without extension (as stored in CSV)
                "filename_with_ext": f"{video_filename}.avi",  # With extension (actual file)
                "ejection_fraction": float(row['EF']) if 'EF' in row else None,  # % of blood pumped out
                "frame_height": int(row['FrameHeight']) if 'FrameHeight' in row else None,
                "frame_width": int(row['FrameWidth']) if 'FrameWidth' in row else None,
                "fps": float(row['FPS']) if 'FPS' in row else None,  # Frames per second
                "number_of_frames": int(row['NumberOfFrames']) if 'NumberOfFrames' in row else None,
                "split": row['Split'] if 'Split' in row else None,  # TRAIN/VAL/TEST
                "exists": video_path.exists()  # Verify file actually exists on disk
            })

        return {
            "videos": videos,
            "total_videos": total_videos,
            "offset": offset,
            "limit": limit,
            "returned": len(videos),
            "videos_path": str(videos_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing videos: {str(e)}"
        )

@app.get("/videos/search")
async def search_videos(query: str, limit: int = 20):
    """
    Search for videos by filename pattern.

    Args:
        query: Search pattern (case-insensitive, partial match)
        limit: Maximum number of results (default: 20, max: 100)

    Returns:
        List of matching videos
    """
    try:
        if limit > 100:
            limit = 100

        echonet_filelist, _, videos_path = load_echonet_with_kagglehub()

        if echonet_filelist is None:
            raise HTTPException(status_code=500, detail="Failed to load EchoNet dataset")

        # Search for matching filenames using case-insensitive substring matching
        # Example: query="0X10" matches "0X1005D03EED19C65B", "0X10A9C59476F90E1B", etc.
        query_lower = query.lower()
        matches = []

        # Iterate through all videos and find matches
        # Stops early once limit is reached for performance
        for idx, row in echonet_filelist.iterrows():
            video_filename = row['FileName']
            if query_lower in video_filename.lower():
                video_path = videos_path / f"{video_filename}.avi"
                matches.append({
                    "index": int(idx),
                    "filename": video_filename,
                    "filename_with_ext": f"{video_filename}.avi",
                    "ejection_fraction": float(row['EF']) if 'EF' in row else None,
                    "exists": video_path.exists()
                })

                # Stop searching once we've found enough matches
                # This improves performance for common queries
                if len(matches) >= limit:
                    break

        return {
            "query": query,
            "matches": matches,
            "total_matches": len(matches),
            "limit": limit
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching videos: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict heart disease for a given patient data and video.

    Args:
        request: PredictionRequest containing patient_data and video_index
        special case: 0X10094BA0A028EAC3

    Returns:
        PredictionResponse with prediction results
    """
    try:
        # Convert PatientData to dict for processing
        # model_dump() is Pydantic v2's method (replaces dict() from v1)
        patient_json = request.patient_data.model_dump()

        # Determine video index from either filename or index
        # Priority: video_filename > video_index > default(0)
        if request.video_filename is not None:
            # Get index from filename using FileList lookup
            video_idx = get_video_index_from_filename(request.video_filename)
        else:
            # Use provided index (defaults to 0 if not provided)
            video_idx = request.video_index if request.video_index is not None else 0

        # Make prediction
        prediction, probability, info = predict_single_sample(
            sample_json=patient_json,
            video_index=video_idx,
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
        # Model outputs probability for class 1 (Disease)
        # Confidence is the max of: P(Disease) or P(No Disease) = 1 - P(Disease)
        # Example: if P(Disease)=0.8, confidence=0.8; if P(Disease)=0.2, confidence=0.8
        confidence = max(float(probability), 1.0 - float(probability))

        # Prepare response
        response = PredictionResponse(
            prediction=prediction,
            prediction_label="Disease" if prediction == 1 else "No Disease",
            disease_probability=probability,
            confidence=confidence,
            patient_id=request.patient_data.id,
            video_index=video_idx,
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

    # Process each prediction request independently
    # Failed predictions don't stop the batch - they're added to errors list
    for idx, request in enumerate(requests):
        try:
            # Convert PatientData to dict
            patient_json = request.patient_data.model_dump()

            # Determine video index (same logic as single prediction)
            if request.video_filename is not None:
                video_idx = get_video_index_from_filename(request.video_filename)
            else:
                video_idx = request.video_index if request.video_index is not None else 0

            prediction, probability, info = predict_single_sample(
                sample_json=patient_json,
                video_index=video_idx,
                fusion_model=fusion_model,
                num_scaler=num_scaler,
                cnn_scaler=cnn_scaler
            )

            if prediction is not None and probability is not None:
                confidence = max(float(probability), 1.0 - float(probability))
                # Add successful prediction to results
                # Includes video_filename if provided (for traceability)
                results.append({
                    "patient_id": request.patient_data.id,
                    "video_index": video_idx,  # Actual index used (even if filename was provided)
                    "video_filename": request.video_filename if request.video_filename else None,
                    "prediction": prediction,
                    "prediction_label": "Disease" if prediction == 1 else "No Disease",
                    "disease_probability": probability,
                    "confidence": confidence,
                    "ground_truth": info.get('ground_truth_label')
                })
            else:
                # Prediction returned None - model issue
                errors.append({
                    "index": idx,  # Position in batch
                    "patient_id": request.patient_data.id,
                    "error": "Prediction failed"
                })
        except Exception as e:
            # Catch any exception during prediction (file not found, invalid data, etc.)
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