"""
Age Estimation Package Initialization

This module serves as the package entry point for the age estimation project.
It exposes key components for easy importing and defines package metadata.

Usage:
    from age_estimation import AgePredictor, Config

    predictor = AgePredictor('model.pth')
    age = predictor.predict_single_image('face.jpg')
"""

# Package metadata
__version__ = '1.0.0'
__author__ = 'Age Estimation Project'
__description__ = 'Swin Transformer-based age estimation from facial images'

# Import key components for convenient access
from .config import Config, ProjectConfiguration
from .age_model import SwinAgeEstimator, create_age_estimation_model
from .inference import AgePredictor
from .prepare_utkface import UTKFaceDatasetPreparator
from .dataset_loader import FaceAgeDataset, create_data_loaders, split_dataset

# Define what gets imported with "from age_estimation import *"
__all__ = [
    # Configuration
    'Config',
    'ProjectConfiguration',

    # Models
    'SwinAgeEstimator',
    'create_age_estimation_model',

    # Inference
    'AgePredictor',

    # Data handling
    'UTKFaceDatasetPreparator',
    'FaceAgeDataset',
    'create_data_loaders',
    'split_dataset',
]

# Package-level documentation
def get_info():
    """
    Display package information.

    Returns:
        dict: Package metadata
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'components': __all__
    }