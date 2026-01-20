"""
UTKFace Dataset Extraction and Age Parsing Module

This script handles:
- Extraction of compressed dataset archives
- Parsing age information from filename patterns
- Filtering invalid or corrupted entries
- Creating structured dataframes for training

Expected filename format: ID_BirthDate_PhotoDate.jpg
Example: "12345_1995-03-15_2015-08-20.jpg" → age = 20 years
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


class UTKFaceDatasetPreparator:
    """
    Handles extraction and preprocessing of UTKFace dataset.

    This class manages:
    - Archive extraction to local storage
    - Age calculation from filename timestamps
    - Data validation and filtering
    - Creation of structured metadata
    """

    def __init__(self, archive_path, extraction_directory,
                 min_age=0, max_age=100):
        """
        Initialize dataset preparator.

        Args:
            archive_path (str): Path to compressed dataset file
            extraction_directory (str): Where to extract dataset
            min_age (int): Minimum valid age for filtering
            max_age (int): Maximum valid age for filtering
        """
        self.archive_location = archive_path
        self.target_directory = extraction_directory
        self.min_acceptable_age = min_age
        self.max_acceptable_age = max_age

        # Storage for parsed metadata
        self.image_file_paths = []
        self.calculated_ages = []

    def extract_archive(self, force_reextraction=False):
        """
        Extract dataset from ZIP archive to target directory.

        Args:
            force_reextraction (bool): Re-extract even if already exists
        """
        # Check if extraction already completed
        if os.path.exists(self.target_directory) and not force_reextraction:
            print(f"✅ Dataset already extracted at: {self.target_directory}")
            return

        print("=" * 70)
        print("EXTRACTING DATASET ARCHIVE")
        print("=" * 70)
        print(f"Source: {self.archive_location}")
        print(f"Destination: {self.target_directory}")
        print("This may take several minutes for large datasets...")

        # Create target directory
        os.makedirs(self.target_directory, exist_ok=True)

        # Extract with progress tracking
        try:
            with zipfile.ZipFile(self.archive_location, 'r') as archive:
                # Get total file count for progress bar
                file_count = len(archive.namelist())

                # Extract all files with progress bar
                for file_info in tqdm(archive.namelist(),
                                      desc="Extracting files",
                                      total=file_count):
                    archive.extract(file_info, self.target_directory)

            print(f"✅ Extraction completed successfully!")
            print(f"Files extracted to: {self.target_directory}")

        except zipfile.BadZipFile:
            print(f"❌ Error: Invalid or corrupted ZIP file")
            raise
        except Exception as error:
            print(f"❌ Extraction failed: {error}")
            raise

    def parse_age_from_filename(self, filename):
        """
        Calculate age from filename timestamp pattern.

        Filename format: ID_BirthDate_PhotoDate.extension
        Example: "12345_1995-03-15_2015-08-20.jpg"

        Age calculation: PhotoYear - BirthYear

        Args:
            filename (str): Image filename to parse

        Returns:
            int or None: Calculated age, or None if parsing fails
        """
        try:
            # Split filename into components
            # Expected parts: [ID, BirthDate, PhotoDate.ext]
            components = filename.split('_')

            if len(components) < 3:
                return None

            # Extract birth year from second component
            # Format: YYYY-MM-DD
            birth_year_str = components[1].split('-')[0]
            birth_year = int(birth_year_str)

            # Extract photo year from third component
            # Format: YYYY-MM-DD.jpg (need to remove extension)
            photo_year_str = components[2].split('-')[0].split('.')[0]
            photo_year = int(photo_year_str)

            # Calculate age at time of photo
            calculated_age = photo_year - birth_year

            # Validate age is within acceptable range
            if self.min_acceptable_age <= calculated_age <= self.max_acceptable_age:
                return calculated_age
            else:
                return None

        except (ValueError, IndexError, AttributeError):
            # Parsing failed - return None for invalid filenames
            return None

    def scan_and_parse_dataset(self):
        """
        Recursively scan directory for images and extract age labels.

        This method:
        1. Walks through all subdirectories
        2. Identifies image files (JPG, PNG, JPEG)
        3. Parses age from filename
        4. Stores valid entries

        Returns:
            pd.DataFrame: DataFrame with columns ['path', 'age']
        """
        print("=" * 70)
        print("SCANNING AND PARSING DATASET")
        print("=" * 70)

        # Reset storage
        self.image_file_paths = []
        self.calculated_ages = []

        # Track statistics
        total_files_found = 0
        valid_entries = 0
        invalid_entries = 0

        # Supported image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png')

        # Walk through all directories
        for root_dir, _, filenames in os.walk(self.target_directory):
            for filename in filenames:
                # Check if file is an image
                if filename.lower().endswith(valid_extensions):
                    total_files_found += 1

                    # Parse age from filename
                    age = self.parse_age_from_filename(filename)

                    if age is not None:
                        # Valid entry - store path and age
                        full_path = os.path.join(root_dir, filename)
                        self.image_file_paths.append(full_path)
                        self.calculated_ages.append(age)
                        valid_entries += 1
                    else:
                        invalid_entries += 1

        # Display statistics
        print(f"📊 Scanning Complete:")
        print(f"   Total files found: {total_files_found}")
        print(f"   Valid entries: {valid_entries}")
        print(f"   Invalid/filtered: {invalid_entries}")
        print(f"   Success rate: {100 * valid_entries / max(total_files_found, 1):.1f}%")

        # Create DataFrame
        dataset_dataframe = pd.DataFrame({
            'path': self.image_file_paths,
            'age': self.calculated_ages
        })

        # Display age distribution
        print(f"\n📈 Age Distribution:")
        print(f"   Min age: {dataset_dataframe['age'].min()}")
        print(f"   Max age: {dataset_dataframe['age'].max()}")
        print(f"   Mean age: {dataset_dataframe['age'].mean():.1f}")
        print(f"   Median age: {dataset_dataframe['age'].median():.1f}")

        return dataset_dataframe

    def prepare_complete_dataset(self, force_reextraction=False):
        """
        Execute full dataset preparation pipeline.

        This is the main entry point that:
        1. Extracts archive
        2. Scans for images
        3. Parses ages
        4. Returns structured dataframe

        Args:
            force_reextraction (bool): Force re-extraction of archive

        Returns:
            pd.DataFrame: Prepared dataset with paths and age labels
        """
        # Step 1: Extract archive
        self.extract_archive(force_reextraction=force_reextraction)

        # Step 2: Scan and parse
        dataset_df = self.scan_and_parse_dataset()

        print("\n✅ Dataset preparation completed successfully!")
        print(f"Dataset ready with {len(dataset_df)} samples")
        print("=" * 70)

        return dataset_df


# ============================================================================
# STANDALONE SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Example usage for standalone execution.
    """
    from config import Config

    # Initialize configuration
    config = Config()

    # Create preparator instance
    preparator = UTKFaceDatasetPreparator(
        archive_path=config.DATASET_ARCHIVE,
        extraction_directory=config.EXTRACTED_DATA_DIR,
        min_age=config.MIN_VALID_AGE,
        max_age=config.MAX_VALID_AGE
    )

    # Execute preparation
    dataset = preparator.prepare_complete_dataset()

    # Save to CSV for later use
    csv_output_path = os.path.join(config.EXTRACTED_DATA_DIR, 'dataset_metadata.csv')
    dataset.to_csv(csv_output_path, index=False)
    print(f"💾 Metadata saved to: {csv_output_path}")