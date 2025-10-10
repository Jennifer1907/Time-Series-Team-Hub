import os
import imageio
import numpy as np
from PIL import Image
import torch
import pandas as pd
from typing import Optional
from .feature_engineer import preprocess_and_engineer_features
from .load_data import sulianova_data, load_echonet_with_kagglehub
from .load_model import cnn_model, transform

def extract_single_video_features(video_path, cnn_model, transform, max_frames=10):
    """
    Extract CNN features from a single video file.

    Args:
        video_path: Path to the video file
        cnn_model: Trained CNN feature extractor
        transform: Image preprocessing transform
        max_frames: Maximum number of frames to extract from video

    Returns:
        features: NumPy array of shape (1, 64) with averaged features
    """
    try:
        # Read video
        video = imageio.get_reader(str(video_path))

        # Extract frames (uniformly sampled)
        total_frames = video.count_frames()
        frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)

        frame_features = []

        for idx in frame_indices:
            frame = video.get_data(idx)

            # Convert to PIL Image and apply transform
            frame_pil = Image.fromarray(frame)
            frame_tensor = transform(frame_pil).unsqueeze(0)  # Add batch dimension

            # Move to device
            device = next(cnn_model.parameters()).device
            frame_tensor = frame_tensor.to(device)

            # Extract features
            with torch.no_grad():
                features = cnn_model(frame_tensor)

            frame_features.append(features.cpu().numpy())

        video.close()

        # Average features across frames
        avg_features = np.mean(frame_features, axis=0)

        return avg_features

    except Exception as e:
        print(f"  ✗ Error extracting features from video: {e}")
        return None


def preprocessing_pipeline(
    sample_json: Optional[dict] = None,
    video_index: int = 0,
    num_scaler=None,
    cnn_scaler=None,
    use_saved_echonet_features=False,
    echonet_features_path='data/echonet_features.npy',
):
    """
    Preprocessing pipeline for a single patient prediction.

    Args:
        sample_json: Dictionary with patient data containing keys:
                    {"id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
                     "cholesterol", "gluc", "smoke", "alco", "active", "cardio"}
        video_index: Index of the video from EchoNet dataset to use for CNN features
        num_scaler: Fitted numerical scaler (RobustScaler) from training
        cnn_scaler: Fitted CNN scaler (StandardScaler) from training
        use_saved_echonet_features: If True, load pre-extracted features instead of re-extracting
        echonet_features_path: Path to saved EchoNet features

    Returns:
        numerical_input: Tensor ready for model input (1, n_features)
        cnn_input: Tensor ready for model input (1, 64) or None
        sample_info: Dictionary with patient information
    """

    print("="*70)
    print("PREPROCESSING PIPELINE FOR SINGLE SAMPLE PREDICTION")
    print("="*70)

    # ============================================
    # STEP 1: Process Numerical Data (Cleveland/Sulianova)
    # ============================================
    print(f"\n[1/3] Processing numerical data from patient JSON...")

    # Convert JSON to DataFrame
    single_sample = pd.DataFrame([sample_json])

    # Preprocess and engineer features
    cleveland_X_processed, cleveland_y, _ = preprocess_and_engineer_features(single_sample)

    print(f"  ✓ Processed features shape: {cleveland_X_processed.shape}")
    print(f"  ✓ Ground truth label: {cleveland_y.values[0] if len(cleveland_y) > 0 else 'Unknown'}")

    # Apply scaler if provided (required for inference with trained model)
    if num_scaler is not None:
        cleveland_X_scaled = num_scaler.transform(cleveland_X_processed)
        print(f"  ✓ Applied numerical scaler")
    else:
        cleveland_X_scaled = cleveland_X_processed.values
        print(f"  ⚠ No scaler provided - using raw features")

    # Convert to PyTorch tensor
    numerical_input = torch.FloatTensor(cleveland_X_scaled)


    # ============================================
    # STEP 2: Process CNN Features (EchoNet)
    # ============================================
    print(f"\n[2/3] Processing CNN features for video index {video_index}...")

    cnn_input = None

    if use_saved_echonet_features and os.path.exists(echonet_features_path):
        # Load pre-extracted features
        print(f"  ✓ Loading pre-extracted EchoNet features from {echonet_features_path}...")
        echonet_features = np.load(echonet_features_path)

        if video_index < len(echonet_features):
            print('echo data video_index:', video_index)
            single_cnn_feature = echonet_features[video_index:video_index+1]
            print(f"  ✓ Extracted feature shape: {single_cnn_feature.shape}")
        else:
            print(f"  ⚠ Video index {video_index} out of range (max: {len(echonet_features)-1})")
            single_cnn_feature = None
    else:
        # Extract features from video on-the-fly
        print(f"  ✓ Extracting CNN features from EchoNet video...")

        # Load EchoNet data
        echonet_filelist, echonet_volume, videos_path = load_echonet_with_kagglehub()

        # Get specific video filename
        if video_index < len(echonet_filelist):
            video_filename = f"{echonet_filelist.iloc[video_index]['FileName']}.avi"
            video_path = videos_path / video_filename

            print(f"  ✓ Processing video: {video_filename} from {video_path}")

            # Extract features for this single video
            single_cnn_feature = extract_single_video_features(
                video_path, cnn_model, transform
            )

            if single_cnn_feature is not None:
                print(f"  ✓ Extracted feature shape: {single_cnn_feature.shape}")
        else:
            print(f"  ⚠ Video index {video_index} out of range")
            single_cnn_feature = None

    # Apply scaler if provided and features exist
    if single_cnn_feature is not None:
        if cnn_scaler is not None:
            single_cnn_feature_scaled = cnn_scaler.transform(single_cnn_feature)
            print(f"  ✓ Applied CNN scaler")
        else:
            single_cnn_feature_scaled = single_cnn_feature
            print(f"  ⚠ No CNN scaler provided - using raw features")

        # Convert to PyTorch tensor
        cnn_input = torch.FloatTensor(single_cnn_feature_scaled)
    else:
        print(f"  ⚠ No CNN features available - will use numerical features only")


    # ============================================
    # STEP 3: Prepare Final Inputs
    # ============================================
    print(f"\n[3/3] Preparing final inputs for model...")

    # Move to device if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  ✓ Output device: {device}')
    numerical_input = numerical_input.to(device)
    if cnn_input is not None:
        cnn_input = cnn_input.to(device)

    print(f"  ✓ Moved tensors to device: {device}")

    # Gather sample information
    sample_info = {
        'patient_id': sample_json.get('id', 'unknown'),
        'video_index': video_index,
        'numerical_features_shape': numerical_input.shape,
        'cnn_features_shape': cnn_input.shape if cnn_input is not None else None,
        'ground_truth_label': cleveland_y.values[0] if len(cleveland_y) > 0 else None,
        'device': device
    }

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Numerical input shape: {numerical_input.shape}")
    print(f"CNN input shape: {cnn_input.shape if cnn_input is not None else 'None'}")
    print(f"Ground truth label: {sample_info['ground_truth_label']}")
    print(f"Device: {device}")

    return numerical_input, cnn_input, sample_info