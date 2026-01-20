"""
Swin Transformer-based Age Estimation Model

This module defines the neural network architecture for predicting age from face images.
It uses a Swin Transformer backbone (pretrained on ImageNet) with a custom classification head.

The model treats age estimation as a classification problem with 101 classes (ages 0-100).
This approach leverages ordinal structure through expectation calculation during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SwinAgeEstimator(nn.Module):
    """
    Age estimation model using Swin Transformer backbone.

    Architecture:
    - Swin Transformer encoder (pretrained on ImageNet)
    - Global average pooling
    - Classification head → 101 age classes

    During inference, age is computed as the expected value of the softmax distribution,
    which implicitly captures the ordinal nature of age.

    Args:
        num_age_classes (int): Number of age classes (default: 101 for ages 0-100)
        backbone_name (str): Name of Swin Transformer variant from timm library
        pretrained (bool): Whether to load ImageNet pretrained weights
    """

    def __init__(self, num_age_classes=101,
                 backbone_name='swin_tiny_patch4_window7_224',
                 pretrained=True):
        """
        Initialize age estimation model.
        """
        super(SwinAgeEstimator, self).__init__()

        self.num_classes = num_age_classes
        self.backbone_architecture = backbone_name

        # ====================================================================
        # BACKBONE: SWIN TRANSFORMER
        # ====================================================================

        # Load Swin Transformer from timm library
        # This provides state-of-the-art vision transformer architecture
        # Pretrained=True loads ImageNet weights for transfer learning
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classification head
            global_pool=''  # We'll add custom pooling
        )

        # Get output feature dimension from backbone
        # Different Swin variants have different feature dimensions
        self.feature_dimension = self.feature_extractor.num_features

        # ====================================================================
        # CUSTOM CLASSIFICATION HEAD
        # ====================================================================

        # Global average pooling reduces spatial dimensions to single vector
        # This is more robust than using only the [CLS] token
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Final classification layer maps features to age classes
        # Output: probability distribution over ages 0-100
        self.age_classifier = nn.Linear(self.feature_dimension, num_age_classes)

        # Initialize classification head with proper weights
        self._initialize_classification_head()

    def _initialize_classification_head(self):
        """
        Initialize weights of the classification head.

        Uses Xavier/Glorot initialization for better convergence.
        The backbone weights are already initialized from ImageNet pretraining.
        """
        nn.init.xavier_uniform_(self.age_classifier.weight)
        nn.init.zeros_(self.age_classifier.bias)

    def forward(self, input_images):
        """
        Forward pass through the network.

        Args:
            input_images (torch.Tensor): Batch of images [B, 3, 224, 224]

        Returns:
            torch.Tensor: Age class logits [B, num_age_classes]
        """
        # Extract features using Swin Transformer backbone
        # Output shape: [B, feature_dim, H', W']
        feature_maps = self.feature_extractor(input_images)

        # Flatten spatial dimensions for pooling
        # Shape: [B, feature_dim, H'*W']
        batch_size = feature_maps.shape[0]
        features_flattened = feature_maps.view(batch_size, self.feature_dimension, -1)

        # Apply global average pooling
        # Shape: [B, feature_dim, 1] → [B, feature_dim]
        pooled_features = self.global_pooling(features_flattened).squeeze(-1)

        # Apply dropout for regularization (only during training)
        pooled_features = self.dropout(pooled_features)

        # Classify into age categories
        # Shape: [B, num_age_classes]
        age_logits = self.age_classifier(pooled_features)

        return age_logits

    def predict_age(self, input_images):
        """
        Predict continuous age values from images.

        This method:
        1. Computes softmax probabilities over age classes
        2. Calculates expected age as weighted sum

        This approach implicitly handles the ordinal nature of age
        and produces smooth predictions.

        Args:
            input_images (torch.Tensor): Batch of images [B, 3, 224, 224]

        Returns:
            torch.Tensor: Predicted ages as floats [B]
        """
        # Get class logits
        age_logits = self.forward(input_images)

        # Convert logits to probabilities
        # Shape: [B, num_age_classes]
        age_probabilities = F.softmax(age_logits, dim=1)

        # Create age range tensor [0, 1, 2, ..., 100]
        age_values = torch.arange(
            self.num_classes,
            dtype=torch.float32,
            device=input_images.device
        )

        # Calculate expected age (weighted average)
        # E[age] = sum(age * P(age))
        predicted_ages = (age_probabilities * age_values).sum(dim=1)

        return predicted_ages


def create_age_estimation_model(config):
    """
    Factory function to create age estimation model from config.

    Args:
        config: Configuration object with model parameters

    Returns:
        SwinAgeEstimator: Initialized model
    """
    print("=" * 70)
    print("BUILDING AGE ESTIMATION MODEL")
    print("=" * 70)

    model = SwinAgeEstimator(
        num_age_classes=config.NUM_AGE_CLASSES,
        backbone_name=config.BACKBONE_ARCHITECTURE,
        pretrained=config.USE_PRETRAINED
    )

    # Move model to appropriate device (GPU/CPU)
    model = model.to(config.DEVICE)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model created successfully:")
    print(f"   Architecture: {config.BACKBONE_ARCHITECTURE}")
    print(f"   Age classes: {config.NUM_AGE_CLASSES}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {config.DEVICE}")
    print("=" * 70)

    return model


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Test model architecture and forward pass.
    """
    from config import Config

    # Initialize config
    cfg = Config()

    # Create model
    model = create_age_estimation_model(cfg)

    # Create dummy input batch
    dummy_images = torch.randn(4, 3, 224, 224).to(cfg.DEVICE)

    # Test forward pass
    print("\n🧪 Testing forward pass...")
    with torch.no_grad():
        logits = model(dummy_images)
        predicted_ages = model.predict_age(dummy_images)

    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Predicted ages: {predicted_ages.cpu().numpy()}")
    print("✅ Model test successful!")