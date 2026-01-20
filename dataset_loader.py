"""
PyTorch Dataset and DataLoader Module for Age Estimation

This module provides:
- Custom Dataset class for loading face images with age labels
- Image preprocessing and augmentation pipelines
- Train/validation split utilities
- Efficient batch loading with DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd


class FaceAgeDataset(Dataset):
    """
    PyTorch Dataset for face images with age labels.

    This dataset handles:
    - Loading images from disk
    - Applying transformations (resize, normalize, augment)
    - Converting age labels to tensors

    Args:
        dataframe (pd.DataFrame): DataFrame with 'path' and 'age' columns
        image_transforms (callable): Torchvision transforms to apply
        is_training (bool): Whether this is training data (affects augmentation)
    """

    def __init__(self, dataframe, image_transforms=None, is_training=True):
        """
        Initialize dataset with dataframe and transforms.
        """
        self.data_frame = dataframe.reset_index(drop=True)
        self.transforms = image_transforms
        self.training_mode = is_training

    def __len__(self):
        """
        Return total number of samples in dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data_frame)

    def __getitem__(self, index):
        """
        Load and return a single sample (image, age_label).

        Args:
            index (int): Index of sample to retrieve

        Returns:
            tuple: (image_tensor, age_label)
                - image_tensor: Preprocessed image as torch.Tensor [C, H, W]
                - age_label: Age as torch.Tensor (long dtype for classification)
        """
        # Get file path and age label from dataframe
        sample_row = self.data_frame.iloc[index]
        image_path = sample_row['path']
        age_value = sample_row['age']

        # Load image and convert to RGB (handles grayscale images)
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if provided
        if self.transforms is not None:
            image = self.transforms(image)

        # Convert age to tensor (long dtype required for CrossEntropyLoss)
        age_tensor = torch.tensor(age_value, dtype=torch.long)

        return image, age_tensor


def create_image_transforms(config, is_training=True):
    """
    Create image transformation pipeline for training or validation.

    Training transforms include augmentation (flips, crops) to prevent overfitting.
    Validation transforms only include necessary preprocessing (resize, normalize).

    Args:
        config: Configuration object with image size and normalization params
        is_training (bool): Whether to apply data augmentation

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline
    """

    if is_training:
        # Training pipeline with augmentation
        transform_pipeline = transforms.Compose([
            # Resize to target dimensions
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),

            # Random horizontal flip for augmentation
            # Faces are roughly symmetric, so flipping is valid
            transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),

            # Optional: Add slight rotation for more robustness
            # transforms.RandomRotation(degrees=5),

            # Optional: Color jitter for lighting variation
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),

            # Convert PIL Image to PyTorch tensor [0, 1]
            transforms.ToTensor(),

            # Normalize using ImageNet statistics
            # This is crucial when using pretrained models
            transforms.Normalize(
                mean=config.NORMALIZATION_MEAN,
                std=config.NORMALIZATION_STD
            ),
        ])
    else:
        # Validation/test pipeline without augmentation
        transform_pipeline = transforms.Compose([
            # Only resize and normalize - no randomness
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZATION_MEAN,
                std=config.NORMALIZATION_STD
            ),
        ])

    return transform_pipeline


def split_dataset(dataframe, validation_ratio=0.2, random_seed=42,
                  stratify_by_age=True):
    """
    Split dataset into training and validation sets.

    Args:
        dataframe (pd.DataFrame): Complete dataset
        validation_ratio (float): Fraction of data for validation (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
        stratify_by_age (bool): Maintain age distribution in both splits

    Returns:
        tuple: (train_df, validation_df)
    """
    print("=" * 70)
    print("SPLITTING DATASET INTO TRAIN/VALIDATION")
    print("=" * 70)

    # Stratification ensures both sets have similar age distributions
    # This is important for balanced training and evaluation
    stratify_column = dataframe['age'] if stratify_by_age else None

    train_dataframe, val_dataframe = train_test_split(
        dataframe,
        test_size=validation_ratio,
        random_state=random_seed,
        stratify=stratify_column
    )

    print(f"✅ Dataset split completed:")
    print(f"   Training samples: {len(train_dataframe)}")
    print(f"   Validation samples: {len(val_dataframe)}")
    print(f"   Split ratio: {100 * (1 - validation_ratio):.0f}% / {100 * validation_ratio:.0f}%")

    if stratify_by_age:
        print(f"   ✓ Age distribution preserved through stratification")

    print("=" * 70)

    return train_dataframe, val_dataframe


def create_data_loaders(train_dataframe, val_dataframe, config):
    """
    Create PyTorch DataLoaders for training and validation.

    DataLoaders handle:
    - Batching samples together
    - Shuffling data (training only)
    - Parallel data loading with multiple workers
    - Automatic tensor stacking

    Args:
        train_dataframe (pd.DataFrame): Training dataset
        val_dataframe (pd.DataFrame): Validation dataset
        config: Configuration object with batch size and worker settings

    Returns:
        tuple: (train_loader, validation_loader)
    """
    print("=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)

    # Create transformation pipelines
    train_transforms = create_image_transforms(config, is_training=True)
    val_transforms = create_image_transforms(config, is_training=False)

    # Create dataset instances
    training_dataset = FaceAgeDataset(
        train_dataframe,
        image_transforms=train_transforms,
        is_training=True
    )

    validation_dataset = FaceAgeDataset(
        val_dataframe,
        image_transforms=val_transforms,
        is_training=False
    )

    # Create DataLoader for training
    # Shuffle=True randomizes order each epoch (prevents overfitting to order)
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Important for training
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False  # Faster GPU transfer
    )

    # Create DataLoader for validation
    # Shuffle=False maintains consistent evaluation order
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No shuffling for validation
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✅ DataLoaders created:")
    print(f"   Training batches: {len(training_loader)}")
    print(f"   Validation batches: {len(validation_loader)}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Worker processes: {config.NUM_WORKERS}")
    print("=" * 70)

    return training_loader, validation_loader


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Test dataset loading functionality.
    """
    from config import Config
    from prepare_utkface import UTKFaceDatasetPreparator

    # Initialize config
    cfg = Config()

    # Prepare dataset
    preparator = UTKFaceDatasetPreparator(
        cfg.DATASET_ARCHIVE,
        cfg.EXTRACTED_DATA_DIR,
        cfg.MIN_VALID_AGE,
        cfg.MAX_VALID_AGE
    )
    dataset_df = preparator.prepare_complete_dataset()

    # Split dataset
    train_df, val_df = split_dataset(
        dataset_df,
        validation_ratio=cfg.VALIDATION_SPLIT,
        random_seed=cfg.RANDOM_SEED
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_df, val_df, cfg)

    # Test loading a batch
    print("\n🧪 Testing batch loading...")
    images, ages = next(iter(train_loader))
    print(f"   Image batch shape: {images.shape}")
    print(f"   Age labels shape: {ages.shape}")
    print(f"   Age range in batch: {ages.min().item()} to {ages.max().item()}")
    print("✅ Dataset loading test successful!")