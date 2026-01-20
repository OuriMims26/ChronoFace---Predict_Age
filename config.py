"""
Configuration Module for Age Estimation Project

This module centralizes all hyperparameters, paths, and settings
to facilitate experimentation and deployment across different environments.

Usage:
    from config import Config
    cfg = Config()
    model = create_model(cfg.NUM_AGE_CLASSES)
"""

import os
import torch


class ProjectConfiguration:
    """
    Centralized configuration for age estimation model training and inference.

    This class manages:
    - File paths for datasets and model checkpoints
    - Model architecture parameters
    - Training hyperparameters
    - Data augmentation settings
    - Hardware configuration (CPU/GPU)

    Attributes can be modified for experimentation without changing core code.
    """

    def __init__(self, use_google_drive=False, drive_project_path=None):
        """
        Initialize configuration with environment detection.

        Args:
            use_google_drive (bool): Whether running in Google Colab with Drive mounted
            drive_project_path (str): Path to project folder on Google Drive
        """

        # ====================================================================
        # ENVIRONMENT DETECTION
        # ====================================================================

        self.IS_COLAB = use_google_drive
        self.DRIVE_BASE = drive_project_path if use_google_drive else None

        # ====================================================================
        # DATASET PATHS
        # ====================================================================

        if self.IS_COLAB and self.DRIVE_BASE:
            # Google Colab with Drive mounted
            self.DATASET_ARCHIVE = f"{self.DRIVE_BASE}/dataset.zip"
            self.EXTRACTED_DATA_DIR = "/content/dataset_local"  # Fast local storage
            self.CHECKPOINT_DIR = f"{self.DRIVE_BASE}/checkpoints"
        else:
            # Local development environment
            self.DATASET_ARCHIVE = "./data/utkface.zip"
            self.EXTRACTED_DATA_DIR = "./data/utkface_extracted"
            self.CHECKPOINT_DIR = "./saved_models"

        # Output paths for trained models
        self.BEST_MODEL_PATH = f"{self.CHECKPOINT_DIR}/best_age_estimator.pth"
        self.FINAL_MODEL_PATH = f"{self.CHECKPOINT_DIR}/final_age_estimator.pth"

        # ====================================================================
        # MODEL ARCHITECTURE PARAMETERS
        # ====================================================================

        # Swin Transformer variant to use
        # Options: 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'
        self.BACKBONE_ARCHITECTURE = 'swin_tiny_patch4_window7_224'

        # Number of age classes (0 to 100 years inclusive)
        self.NUM_AGE_CLASSES = 101

        # Input image dimensions (height, width)
        # Swin Transformer expects 224x224 inputs
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224

        # Whether to use pretrained ImageNet weights for initialization
        self.USE_PRETRAINED = True

        # ====================================================================
        # TRAINING HYPERPARAMETERS
        # ====================================================================

        # Number of complete passes through the training dataset
        self.NUM_EPOCHS = 20

        # Batch size (number of images processed simultaneously)
        # Reduce if running out of GPU memory
        self.BATCH_SIZE = 32

        # Initial learning rate for optimizer
        self.LEARNING_RATE = 5e-5

        # Weight decay for L2 regularization (prevents overfitting)
        self.WEIGHT_DECAY = 0.01

        # Validation split ratio (0.2 = 20% for validation)
        self.VALIDATION_SPLIT = 0.2

        # Random seed for reproducibility
        self.RANDOM_SEED = 42

        # ====================================================================
        # DATA AUGMENTATION SETTINGS
        # ====================================================================

        # Probability of horizontal flip during training
        self.HORIZONTAL_FLIP_PROB = 0.5

        # ImageNet normalization statistics
        # These values are standard for models pretrained on ImageNet
        self.NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZATION_STD = [0.229, 0.224, 0.225]

        # ====================================================================
        # HARDWARE CONFIGURATION
        # ====================================================================

        # Automatically detect GPU availability
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Number of worker processes for data loading
        # Set to 0 for debugging, increase for faster data loading
        self.NUM_WORKERS = 4 if not self.IS_COLAB else 2

        # Enable cuDNN benchmarking for faster training (if using GPU)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # ====================================================================
        # LOGGING AND CHECKPOINTING
        # ====================================================================

        # How often to print training statistics (in batches)
        self.LOG_INTERVAL = 50

        # Save checkpoint every N epochs
        self.CHECKPOINT_FREQUENCY = 5

        # Whether to save only the best model (based on validation MAE)
        self.SAVE_BEST_ONLY = True

        # Early stopping patience (stop if no improvement for N epochs)
        # Set to None to disable early stopping
        self.EARLY_STOPPING_PATIENCE = 5

        # ====================================================================
        # AGE EXTRACTION PATTERNS
        # ====================================================================

        # Expected filename format for UTKFace dataset
        # Example: "123_1995-01-15_2015-06-20.jpg"
        # Pattern: ID_BirthDate_PhotoDate.jpg
        self.FILENAME_PATTERN = r"(\d+)_(\d{4})-\d{2}-\d{2}_(\d{4})-\d{2}-\d{2}\.(jpg|png|jpeg)"

        # Valid age range for filtering outliers
        self.MIN_VALID_AGE = 0
        self.MAX_VALID_AGE = 100

    def create_directories(self):
        """
        Create necessary directories for data and checkpoints.

        This method ensures all required folders exist before training begins.
        """
        os.makedirs(self.EXTRACTED_DATA_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        print(f"✅ Directories created: {self.EXTRACTED_DATA_DIR}, {self.CHECKPOINT_DIR}")

    def print_configuration(self):
        """
        Display current configuration settings.

        Useful for debugging and verifying hyperparameters before training.
        """
        print("=" * 70)
        print("AGE ESTIMATION MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Device: {self.DEVICE}")
        print(f"Backbone: {self.BACKBONE_ARCHITECTURE}")
        print(f"Age Classes: {self.NUM_AGE_CLASSES}")
        print(f"Image Size: {self.IMAGE_HEIGHT}x{self.IMAGE_WIDTH}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Epochs: {self.NUM_EPOCHS}")
        print(f"Validation Split: {self.VALIDATION_SPLIT}")
        print(f"Checkpoint Dir: {self.CHECKPOINT_DIR}")
        print("=" * 70)


# Convenience alias for easier imports
Config = ProjectConfiguration