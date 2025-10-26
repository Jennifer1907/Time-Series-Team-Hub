import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from .model import FeatureFusionModel
import pickle
import os

def initialize_fixed_cnn_model():
    """Initialize FIXED ResNet-50 CNN for image feature extraction"""
    print("Initializing FIXED ResNet-50 CNN...")

    # Load pre-trained ResNet-50
    resnet = models.resnet50(pretrained=True)

    # FIXED: Proper feature extractor architecture
    class FixedCNNFeatureExtractor(L.LightningModule): #? apply pytorch lightning
        def __init__(self):
            super(FixedCNNFeatureExtractor, self).__init__()
            # Use ResNet backbone without the final FC layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_layer = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 64)  # Match MLP output dimension
            )

        def forward(self, x):
            # Input shape: (batch_size, 3, 224, 224)
            x = self.backbone(x)  # Shape: (batch_size, 2048, 7, 7)
            x = self.avgpool(x)   # Shape: (batch_size, 2048, 1, 1)
            x = torch.flatten(x, 1)  # Shape: (batch_size, 2048)
            x = self.feature_layer(x)  # Shape: (batch_size, 64)
            return x

    cnn_model = FixedCNNFeatureExtractor()
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    print(f"Model moved to device: {device}")

    # Transform for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    print("FIXED CNN feature extractor initialized!")
    print("Architecture: Input(3,224,224) -> ResNet -> 2048 -> 512 -> 64")
    return cnn_model, transform


def create_fusion_model(numerical_dim):
    fusion_model = FeatureFusionModel(numerical_dim=numerical_dim, fusion_dim=128, num_classes=2)
    return fusion_model

# Recreate the model architecture (must match the original)
fusion_model = create_fusion_model(numerical_dim=28)  # Use the same numerical_dim as in training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load weights
base_path = '../models'
model_path = f'{base_path}/best_fusion_model.pth'  # Or any fold-specific path
# Check if paths are valid
if not os.path.exists(model_path):
	raise FileNotFoundError(f"Model path does not exist: {model_path}")
if not os.path.exists(f'{base_path}/num_scaler_fold_1.pkl'):
	raise FileNotFoundError(f"Numerical scaler path does not exist: {base_path}/num_scaler_fold_1.pkl")
if not os.path.exists(f'{base_path}/cnn_scaler_fold_1.pkl'):
	raise FileNotFoundError(f"CNN scaler path does not exist: {base_path}/cnn_scaler_fold_1.pkl")


fusion_model.load_state_dict(torch.load(model_path))
fusion_model.to(device) # to 'cuda'
fusion_model.eval()  # Set to evaluation mode


with open(f'{base_path}/num_scaler_fold_1.pkl', 'rb') as f:
    num_scaler = pickle.load(f)
with open(f'{base_path}/cnn_scaler_fold_1.pkl', 'rb') as f:
    cnn_scaler = pickle.load(f)


cnn_model, transform = initialize_fixed_cnn_model()