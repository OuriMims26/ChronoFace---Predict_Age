"""
Inference Script for Age Estimation

This module provides utilities for:
- Loading trained models
- Making predictions on single images or batches
- Visualizing results
- Batch processing entire directories
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

from config import Config
from age_model import SwinAgeEstimator


class AgePredictor:
    """
    Wrapper class for convenient age prediction from trained models.

    This class handles:
    - Model loading from checkpoints
    - Image preprocessing
    - Prediction generation
    - Batch processing
    """

    def __init__(self, checkpoint_path, config=None, device=None):
        """
        Initialize predictor with trained model.

        Args:
            checkpoint_path (str): Path to saved model checkpoint
            config (Config, optional): Configuration object
            device (str, optional): Device to run inference on
        """
        # Use provided config or create default
        self.cfg = config if config is not None else Config()

        # Determine device (GPU/CPU)
        if device is not None:
            self.computation_device = torch.device(device)
        else:
            self.computation_device = self.cfg.DEVICE

        # ====================================================================
        # LOAD MODEL
        # ====================================================================

        print("=" * 70)
        print("LOADING AGE ESTIMATION MODEL")
        print("=" * 70)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.computation_device}")

        # Create model instance
        self.model = SwinAgeEstimator(
            num_age_classes=self.cfg.NUM_AGE_CLASSES,
            backbone_name=self.cfg.BACKBONE_ARCHITECTURE,
            pretrained=False  # We're loading trained weights
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.computation_device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_mae = checkpoint.get('best_mae', 'Unknown')
            print(f"Best training MAE: {best_mae}")
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)

        # Move model to device and set to evaluation mode
        self.model.to(self.computation_device)
        self.model.eval()

        print("✅ Model loaded successfully")
        print("=" * 70)

        # ====================================================================
        # SETUP IMAGE PREPROCESSING
        # ====================================================================

        # Define transformation pipeline for inference
        # No augmentation - only resize and normalize
        self.preprocessing_pipeline = transforms.Compose([
            transforms.Resize((self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.cfg.NORMALIZATION_MEAN,
                std=self.cfg.NORMALIZATION_STD
            ),
        ])


    def preprocess_image(self, image_input):
        """
        Preprocess image for model input.

        Args:
            image_input: Can be PIL Image or file path (str)

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, H, W]
        """
        # Load image if path is provided
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError("Input must be PIL Image or file path")

        # Apply transformations
        image_tensor = self.preprocessing_pipeline(image)

        # Add batch dimension [1, 3, H, W]
        image_batch = image_tensor.unsqueeze(0)

        return image_batch


    def predict_single_image(self, image_input, return_distribution=False):
        """
        Predict age from a single image.

        Args:
            image_input: PIL Image or file path
            return_distribution (bool): Whether to return full probability distribution

        Returns:
            float or tuple: Predicted age, optionally with probability distribution
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_input)
        image_tensor = image_tensor.to(self.computation_device)

        # Perform inference (no gradient computation needed)
        with torch.no_grad():
            # Get model output
            age_logits = self.model(image_tensor)

            # Convert to probabilities
            age_probabilities = F.softmax(age_logits, dim=1)

            # Calculate expected age
            age_range = torch.arange(
                self.cfg.NUM_AGE_CLASSES,
                dtype=torch.float32,
                device=self.computation_device
            )
            predicted_age = (age_probabilities * age_range).sum(dim=1).item()

        if return_distribution:
            # Return both age and probability distribution
            distribution = age_probabilities.cpu().numpy()[0]
            return predicted_age, distribution
        else:
            return predicted_age


    def predict_batch(self, image_paths):
        """
        Predict ages for multiple images efficiently.

        Args:
            image_paths (list): List of file paths to images

        Returns:
            list: Predicted ages for each image
        """
        print(f"Processing batch of {len(image_paths)} images...")

        predicted_ages = []

        for img_path in image_paths:
            try:
                age = self.predict_single_image(img_path)
                predicted_ages.append(age)
            except Exception as error:
                print(f"⚠️  Error processing {img_path}: {error}")
                predicted_ages.append(None)

        return predicted_ages


    def process_directory(self, directory_path, output_csv=None):
        """
        Process all images in a directory and optionally save results.

        Args:
            directory_path (str): Path to directory containing images
            output_csv (str, optional): Path to save results as CSV

        Returns:
            dict: Dictionary mapping filenames to predicted ages
        """
        print("=" * 70)
        print(f"PROCESSING DIRECTORY: {directory_path}")
        print("=" * 70)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []

        for file_path in Path(directory_path).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))

        print(f"Found {len(image_files)} images")

        # Process all images
        results = {}

        for img_path in image_files:
            try:
                age = self.predict_single_image(img_path)
                filename = os.path.basename(img_path)
                results[filename] = age
                print(f"✓ {filename}: {age:.1f} years")
            except Exception as error:
                print(f"✗ {os.path.basename(img_path)}: Error - {error}")

        # Save to CSV if requested
        if output_csv:
            import pandas as pd
            df = pd.DataFrame.from_dict(
                results,
                orient='index',
                columns=['predicted_age']
            )
            df.index.name = 'filename'
            df.to_csv(output_csv)
            print(f"\n💾 Results saved to: {output_csv}")

        print("=" * 70)
        return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Simple CLI for age prediction.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Predict age from face images')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output CSV file (for directory mode)')

    args = parser.parse_args()

    # Load predictor
    config = Config()
    predictor = AgePredictor(args.checkpoint, config)

    # Single image mode
    if args.image:
        print("\n" + "=" * 70)
        print("SINGLE IMAGE PREDICTION")
        print("=" * 70)
        print(f"Image: {args.image}")

        age, distribution = predictor.predict_single_image(
            args.image,
            return_distribution=True
        )

        print(f"\n🎯 Predicted Age: {age:.1f} years")

        # Show confidence (probability mass around prediction)
        import numpy as np
        top_k = 5
        top_indices = np.argsort(distribution)[-top_k:][::-1]
        print(f"\nTop {top_k} most likely ages:")
        for idx in top_indices:
            print(f"  Age {idx}: {distribution[idx]*100:.1f}%")

    # Directory mode
    elif args.directory:
        results = predictor.process_directory(args.directory, args.output)

        if results:
            import numpy as np
            ages = [v for v in results.values() if v is not None]
            print(f"\n📊 Statistics:")
            print(f"  Images processed: {len(ages)}")
            print(f"  Average age: {np.mean(ages):.1f} years")
            print(f"  Min age: {np.min(ages):.1f} years")
            print(f"  Max age: {np.max(ages):.1f} years")

    else:
        print("Error: Specify either --image or --directory")


if __name__ == '__main__':
    main()