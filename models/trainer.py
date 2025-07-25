# file : models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from datetime import datetime

from .few_shot_model import SiameseEarNetwork, TripletLoss, ElephantIdentifier

class TripletTrainer:
    """Trainer for few-shot elephant ear identification using triplet loss"""

    def __init__(self, model, device='cuda', learning_rate=1e-4, margin=0.2):
        self.model = model.to(device)
        self.device = device
        self.criterion = TripletLoss(margin=margin)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.7)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for training progress"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_triplet_batch(self, data_loader, batch_size=32):
        """Create triplet batches (anchor, positive, negative)"""
        # Group images by elephant ID
        elephant_images = defaultdict(list)

        for batch_idx, (images, labels, _) in enumerate(data_loader):
            for img, label in zip(images, labels):
                elephant_id = data_loader.dataset.class_names[label.item()]
                elephant_images[elephant_id].append(img)

        # Filter out elephants with less than 2 images
        elephant_images = {k: v for k, v in elephant_images.items() if len(v) >= 2}
        elephant_ids = list(elephant_images.keys())

        if len(elephant_ids) < 2:
            raise ValueError("Need at least 2 elephants with 2+ images each for triplet training")

        anchors, positives, negatives = [], [], []

        for _ in range(batch_size):
            # Select random elephant for anchor
            anchor_elephant = random.choice(elephant_ids)

            # Select anchor and positive from same elephant
            if len(elephant_images[anchor_elephant]) >= 2:
                anchor_img, positive_img = random.sample(elephant_images[anchor_elephant], 2)
            else:
                anchor_img = positive_img = random.choice(elephant_images[anchor_elephant])

            # Select negative from different elephant
            negative_elephants = [e for e in elephant_ids if e != anchor_elephant]
            negative_elephant = random.choice(negative_elephants)
            negative_img = random.choice(elephant_images[negative_elephant])

            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)

        return (torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Create progress bar
        pbar = tqdm(range(len(train_loader) // 4), desc=f'Epoch {epoch}')

        for batch_idx in pbar:
            try:
                # Create triplet batch
                anchors, positives, negatives = self.create_triplet_batch(train_loader)

                # Move to device
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                anchor_embeddings = self.model(anchors)
                positive_embeddings = self.model(positives)
                negative_embeddings = self.model(negatives)

                # Compute triplet loss
                loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            except Exception as e:
                self.logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
                continue

        avg_loss = epoch_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate_epoch(self, val_loader):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx in range(min(20, len(val_loader) // 4)):  # Limit validation batches
                try:
                    anchors, positives, negatives = self.create_triplet_batch(val_loader)

                    anchors = anchors.to(self.device)
                    positives = positives.to(self.device)
                    negatives = negatives.to(self.device)

                    anchor_embeddings = self.model(anchors)
                    positive_embeddings = self.model(positives)
                    negative_embeddings = self.model(negatives)

                    loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                    val_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    continue

        avg_val_loss = val_loss / max(num_batches, 1)
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def evaluate_identification_accuracy(self, val_loader, identifier):
        """Evaluate identification accuracy using prototype matching"""
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for images, labels, _ in val_loader:
                for img, label in zip(images, labels):
                    true_elephant = val_loader.dataset.class_names[label.item()]

                    # Get top prediction
                    predictions = identifier.identify_elephant(img, top_k=1)
                    if predictions:
                        predicted_elephant = predictions[0][0]
                        if predicted_elephant == true_elephant:
                            correct += 1
                    total += 1

                    # Limit evaluation to prevent long wait times
                    if total >= 100:
                        break
                if total >= 100:
                    break

        accuracy = correct / max(total, 1)
        self.val_accuracies.append(accuracy)
        return accuracy

    def train(self, train_loader, val_loader, num_epochs=50, save_dir="checkpoints"):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Evaluate identification accuracy every 5 epochs
            if epoch % 5 == 0:
                identifier = ElephantIdentifier(device=self.device)
                identifier.model = self.model
                identifier.create_prototypes(val_loader)
                val_accuracy = self.evaluate_identification_accuracy(val_loader, identifier)

                self.logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.3f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }

                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }, checkpoint_path)

        self.logger.info("Training completed!")
        return self.model

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot validation accuracy
        if self.val_accuracies:
            epochs_with_acc = [i * 5 for i in range(len(self.val_accuracies))]
            axes[1].plot(epochs_with_acc, self.val_accuracies, label='Val Accuracy', color='green')
            axes[1].set_title('Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)

        # Plot learning rate
        axes[2].plot([self.optimizer.param_groups[0]['lr']] * len(self.train_losses))
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training history plot saved to {save_path}")

    def resume_training(self, checkpoint_path):
        """Resume training from checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            if 'val_accuracies' in checkpoint:
                self.val_accuracies = checkpoint['val_accuracies']

            start_epoch = checkpoint['epoch']
            self.logger.info(f"Resumed training from epoch {start_epoch}")
            return start_epoch
        else:
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0


def create_trainer(embedding_dim=256, learning_rate=1e-4, margin=0.2, device='cuda'):
    """Factory function to create trainer with model"""
    model = SiameseEarNetwork(embedding_dim=embedding_dim)
    trainer = TripletTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        margin=margin
    )
    return trainer
