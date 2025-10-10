import torch
from typing import Optional
from .preprocessing import preprocessing_pipeline
from .load_model import fusion_model, num_scaler, cnn_scaler

def predict_single_sample(
    sample_json: Optional[dict] = None,
    video_index=0,
    fusion_model=None,
    num_scaler=None,
    cnn_scaler=None
):
    """
    Complete pipeline: preprocess and predict for a single sample.

    Args:
        sample_json: Dictionary with patient data containing keys:
                    {"id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
					"cholesterol", "gluc", "smoke", "alco", "active", "cardio"}
        video_index: Index of video in EchoNet dataset
        fusion_model: Trained FeatureFusionModel
        num_scaler: Numerical scaler from training
        cnn_scaler: CNN scaler from training

    Returns:
        prediction: Predicted class (0 or 1)
        probability: Probability of disease (class 1)
        sample_info: Information about the sample
    """

    # Preprocess
    numerical_input, cnn_input, sample_info = preprocessing_pipeline(
        sample_json=sample_json,
        video_index=video_index,
        num_scaler=num_scaler,
        cnn_scaler=cnn_scaler,
        use_saved_echonet_features=True  # Set to False to extract on-the-fly
    )

    # Predict
    if fusion_model is not None:
        fusion_model.eval()

        with torch.no_grad():
            outputs, _ = fusion_model(numerical_input, cnn_input)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(dim=1).item()
            print('prediction:', prediction)
            disease_probability = probabilities[0, 1].item()

        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"Predicted class: {prediction} ({'Disease' if prediction == 1 else 'No Disease'})")
        print(f"Disease probability: {disease_probability:.4f}")
        print(f"Ground truth: {sample_info['ground_truth_label']} ({'Disease' if sample_info['ground_truth_label'] == 1 else 'No Disease'})")

        return prediction, disease_probability, sample_info
    else:
        print("⚠ No model provided - returning preprocessed inputs only")
        return None, None, sample_info


if __name__ == '__main__':
    # Example patient data
    sample_patient = {
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

    # Predict for patient with video 0
    prediction, probability, info = predict_single_sample(
        sample_json=sample_patient,
        video_index=0,
        fusion_model=fusion_model,
        num_scaler=num_scaler,
        cnn_scaler=cnn_scaler
    )