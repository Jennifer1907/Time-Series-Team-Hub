import torch
import torch.nn as nn

class NumericalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims = [128, 64, 32], output_dim = 64):
        super(NumericalMLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FeatureFusionModel(nn.Module):
    """CNN + MLP Feature Fusion Model - FIXED"""
    def __init__(self, numerical_dim, fusion_dim=128, num_classes=2):
        super(FeatureFusionModel, self).__init__()

        # MLP branch for numerical data
        self.numerical_mlp = NumericalMLP(
            input_dim=numerical_dim,
            hidden_dims=[128, 64],
            output_dim=64
        )

        # Fusion layers - handles both numerical-only and multimodal cases
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 64, fusion_dim),  # 64 from MLP + 64 from CNN
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, numerical_data, image_features=None):
        # Extract MLP features from numerical data
        mlp_features = self.numerical_mlp(numerical_data)

        if image_features is not None:
            # Concatenate MLP and CNN features
            fused_features = torch.cat([mlp_features, image_features], dim=1)
        else:
            # Use only MLP features if no image data - pad with zeros
            fused_features = torch.cat([mlp_features, torch.zeros_like(mlp_features)], dim=1)

        # Apply fusion layers
        fused_features = self.fusion_layer(fused_features)

        # Classification
        output = self.classifier(fused_features)

        return output, mlp_features
