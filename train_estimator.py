"""
Training Script for Age Estimation Model

This script orchestrates the complete training pipeline:
- Dataset preparation and loading
- Model initialization
- Training loop with validation
- Checkpoint saving and early stopping
- Performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import os

from config import Config
from prepare_utkface import UTKFaceDatasetPreparator
from dataset_loader import split_dataset, create_data_loaders
from age_model import create_age_estimation_model


class AgeEstimationTrainer:
    """
    Manages the training process for age estimation model.

    This class handles:
    - Training and validation loops
    - Loss computation and backpropagation
    - Performance metrics (MAE calculation)
    - Checkpoint management
    - Early stopping
    """

    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize trainer with model and data loaders.

        Args:
            model: Age estimation neural network
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration object with hyperparameters
        """
        self.network = model
        self.training_data = train_loader
        self.validation_data = val_loader
        self.cfg = config

        # ====================================================================
        # OPTIMIZATION COMPONENTS
        # ====================================================================

        # Loss function: Cross-Entropy for classification
        # Treats age as discrete classes (0-100)
        self.classification_criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Create age range tensor for MAE calculation
        # This is used to convert probabilities to expected ages
        self.age_range_tensor = torch.arange(
            config.NUM_AGE_CLASSES,
            dtype=torch.float32,
            device=config.DEVICE
        )

        # ====================================================================
        # TRACKING VARIABLES
        # ====================================================================

        self.best_validation_mae = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': []
        }

    def compute_mae_from_logits(self, logits, true_ages):
        """
        Calculate Mean Absolute Error from model logits.

        Args:
            logits (torch.Tensor): Raw model outputs [B, num_classes]
            true_ages (torch.Tensor): Ground truth ages [B]

        Returns:
            float: Mean absolute error in years
        """
        # Convert logits to probabilities
        age_probabilities = F.softmax(logits, dim=1)

        # Calculate expected age (weighted sum)
        predicted_ages = (age_probabilities * self.age_range_tensor).sum(dim=1)

        # Compute MAE
        mae = F.l1_loss(predicted_ages, true_ages.float()).item()

        return mae

    def train_single_epoch(self, epoch_number):
        """
        Execute one complete training epoch.

        Args:
            epoch_number (int): Current epoch index

        Returns:
            float: Average training loss for this epoch
        """
        # Set model to training mode (enables dropout, batch norm updates)
        self.network.train()

        total_loss = 0.0
        num_batches = len(self.training_data)

        # Progress bar for this epoch
        progress_bar = tqdm(
            self.training_data,
            desc=f"Epoch {epoch_number + 1}/{self.cfg.NUM_EPOCHS} [Training]",
            leave=False
        )

        for batch_idx, (images, age_labels) in enumerate(progress_bar):
            # Move data to device (GPU/CPU)
            images = images.to(self.cfg.DEVICE)
            age_labels = age_labels.to(self.cfg.DEVICE)

            # ================================================================
            # FORWARD PASS
            # ================================================================

            # Get model predictions
            age_logits = self.network(images)

            # Calculate loss (cross-entropy between predictions and labels)
            loss = self.classification_criterion(age_logits, age_labels)

            # ================================================================
            # BACKWARD PASS AND OPTIMIZATION
            # ================================================================

            # Clear gradients from previous iteration
            self.optimizer.zero_grad()

            # Compute gradients via backpropagation
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # ================================================================
            # TRACKING
            # ================================================================

            total_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate average loss for this epoch
        average_loss = total_loss / num_batches

        return average_loss

    def validate(self):
        """
        Evaluate model on validation set.

        Returns:
            tuple: (average_loss, mean_absolute_error)
        """
        # Set model to evaluation mode (disables dropout, batch norm updates)
        self.network.eval()

        total_loss = 0.0
        total_absolute_error = 0.0
        num_samples = 0

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for images, age_labels in tqdm(
                    self.validation_data,
                    desc="Validating",
                    leave=False
            ):
                # Move data to device
                images = images.to(self.cfg.DEVICE)
                age_labels = age_labels.to(self.cfg.DEVICE)

                # Forward pass
                age_logits = self.network(images)

                # Calculate loss
                loss = self.classification_criterion(age_logits, age_labels)
                total_loss += loss.item()

                # Calculate MAE for this batch
                batch_mae = self.compute_mae_from_logits(age_logits, age_labels)
                total_absolute_error += batch_mae * images.size(0)
                num_samples += images.size(0)

        # Calculate averages
        average_loss = total_loss / len(self.validation_data)
        mean_absolute_error = total_absolute_error / num_samples

        return average_loss, mean_absolute_error

    def save_checkpoint(self, filepath, is_best=False):
        """
        Save model checkpoint to disk.

        Args:
            filepath (str): Path where to save the model
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_validation_mae,
            'config': {
                'backbone': self.cfg.BACKBONE_ARCHITECTURE,
                'num_classes': self.cfg.NUM_AGE_CLASSES,
            }
        }

        torch.save(checkpoint, filepath)

        if is_best:
            print(f"💾 Best model saved to: {filepath}")

    def train(self):
        """
        Execute complete training process.

        This is the main training loop that:
        1. Trains for specified number of epochs
        2. Validates after each epoch
        3. Saves checkpoints when performance improves
        4. Implements early stopping if no improvement
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(self.cfg.NUM_EPOCHS):

            # ================================================================
            # TRAINING PHASE
            # ================================================================

            train_loss = self.train_single_epoch(epoch)

            # ================================================================
            # VALIDATION PHASE
            # ================================================================

            val_loss, val_mae = self.validate()

            # ================================================================
            # LOGGING
            # ================================================================

            # Store metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_mae'].append(val_mae)

            # Print epoch summary
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_mae:.2f} years")

            # ================================================================
            # CHECKPOINT SAVING
            # ================================================================

            # Check if this is the best model so far
            if val_mae < self.best_validation_mae:
                improvement = self.best_validation_mae - val_mae
                self.best_validation_mae = val_mae
                self.epochs_without_improvement = 0

                print(f"  🎯 New best MAE! (improved by {improvement:.2f} years)")

                # Save best model
                if self.cfg.SAVE_BEST_ONLY:
                    self.save_checkpoint(self.cfg.BEST_MODEL_PATH, is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")

            # Periodic checkpoint (regardless of performance)
            if (epoch + 1) % self.cfg.CHECKPOINT_FREQUENCY == 0:
                checkpoint_path = f"{self.cfg.CHECKPOINT_DIR}/checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(checkpoint_path)
                print(f"  💾 Periodic checkpoint saved")

            # ================================================================
            # EARLY STOPPING
            # ================================================================

            if (self.cfg.EARLY_STOPPING_PATIENCE is not None and
                    self.epochs_without_improvement >= self.cfg.EARLY_STOPPING_PATIENCE):
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {self.cfg.EARLY_STOPPING_PATIENCE} consecutive epochs")
                break

        # ====================================================================
        # TRAINING COMPLETE
        # ====================================================================

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Best validation MAE: {self.best_validation_mae:.2f} years")
        print(f"Best model saved at: {self.cfg.BEST_MODEL_PATH}")
        print("=" * 70)

        # Save final model
        self.save_checkpoint(self.cfg.FINAL_MODEL_PATH)
        print(f"Final model saved at: {self.cfg.FINAL_MODEL_PATH}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main training pipeline orchestration.
    """
    print("\n" + "=" * 70)
    print("AGE ESTIMATION MODEL - TRAINING PIPELINE")
    print("=" * 70)

    # ========================================================================
    # STEP 1: INITIALIZE CONFIGURATION
    # ========================================================================

    config = Config()
    config.print_configuration()
    config.create_directories()

    # ========================================================================
    # STEP 2: PREPARE DATASET
    # ========================================================================

    preparator = UTKFaceDatasetPreparator(
        archive_path=config.DATASET_ARCHIVE,
        extraction_directory=config.EXTRACTED_DATA_DIR,
        min_age=config.MIN_VALID_AGE,
        max_age=config.MAX_VALID_AGE
    )

    dataset_df = preparator.prepare_complete_dataset()

    # ========================================================================
    # STEP 3: SPLIT DATASET
    # ========================================================================

    train_df, val_df = split_dataset(
        dataset_df,
        validation_ratio=config.VALIDATION_SPLIT,
        random_seed=config.RANDOM_SEED,
        stratify_by_age=True
    )

    # ========================================================================
    # STEP 4: CREATE DATA LOADERS
    # ========================================================================

    train_loader, val_loader = create_data_loaders(train_df, val_df, config)

    # ========================================================================
    # STEP 5: BUILD MODEL
    # ========================================================================

    model = create_age_estimation_model(config)

    # ========================================================================
    # STEP 6: TRAIN MODEL
    # ========================================================================

    trainer = AgeEstimationTrainer(model, train_loader, val_loader, config)
    trainer.train()

    print("\n✅ Pipeline completed successfully!")


if __name__ == '__main__':
    main()