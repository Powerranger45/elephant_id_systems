"""
Main training script for elephant identification system
Handles both classification and metric learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from models.few_shot_model import SiameseEarNetwork, TripletLoss, ElephantIdentifier
from models.ear_detector import SimpleEarDetector
from utils.data_loader import create_data_loaders
#from utils.metrics import calculate_metrics

class ElephantTrainer:
    """Main trainer class for elephant identification"""

    def __init__(self, data_dir, device='cuda', model_save_path='models/'):
        self.data_dir = data_dir
        self.device = device
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)

        # Initialize model
        self.model = SiameseEarNetwork(embedding_dim=256).to(device)
        self.triplet_loss = TripletLoss(margin=0.2)
        self.ce_loss = nn.CrossEntropyLoss()

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Training history
        self.train_losses = []
        self.val_accuracies = []

    def train_epoch_classification(self, train_loader):
        """Train one epoch using classification loss"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Get embeddings
            embeddings = self.model(images)

            # Add classification head for training
            if not hasattr(self, 'classifier'):
                self.classifier = nn.Linear(256, len(train_loader.dataset.class_names)).to(self.device)
                self.classifier_optimizer = optim.AdamW(self.classifier.parameters(), lr=0.001)

            # Classification loss
            logits = self.classifier(embeddings)
            loss = self.ce_loss(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.classifier_optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%')

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def train_epoch_triplet(self, triplet_loader):
        """Train one epoch using triplet loss"""
        self.model.train()
        total_loss = 0

        for batch_idx, ((anchors, positives, negatives), _) in enumerate(triplet_loader):
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)

            # Get embeddings
            anchor_embeddings = self.model(anchors)
            positive_embeddings = self.model(positives)
            negative_embeddings = self.model(negatives)

            # Triplet loss
            loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Triplet Batch {batch_idx}: Loss={loss.item():.4f}')

        epoch_loss = total_loss / len(triplet_loader)
        return epoch_loss

    def validate(self, val_loader):
        """Validate model using prototype matching"""
        self.model.eval()

        # Create prototypes from validation set
        prototypes = {}
        embeddings_by_class = {}

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.cpu().numpy()

                batch_embeddings = self.model(images).cpu().numpy()

                for embedding, label in zip(batch_embeddings, labels):
                    if label not in embeddings_by_class:
                        embeddings_by_class[label] = []
                    embeddings_by_class[label].append(embedding)

        # Create prototypes (average embeddings)
        for label, embeddings in embeddings_by_class.items():
            prototypes[label] = np.mean(embeddings, axis=0)

        # Validate using nearest prototype
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.cpu().numpy()

                batch_embeddings = self.model(images).cpu().numpy()

                for embedding, true_label in zip(batch_embeddings, labels):
                    # Find nearest prototype
                    distances = {}
                    for proto_label, prototype in prototypes.items():
                        distance = np.linalg.norm(embedding - prototype)
                        distances[proto_label] = distance

                    predicted_label = min(distances, key=distances.get)

                    if predicted_label == true_label:
                        correct += 1
                    total += 1

        accuracy = 100. * correct / total
        return accuracy

    def train(self, epochs=100, use_triplet=True):
        """Main training loop"""
        print("Processing dataset and extracting ear regions...")

        # First, extract ear regions
        ear_detector = SimpleEarDetector()
        processed_dir = os.path.join(os.path.dirname(self.data_dir), 'processed_ears')
        ear_detector.process_dataset(self.data_dir, processed_dir)

        # Create data loaders
        train_loader, val_loader, triplet_loader = create_data_loaders(
            processed_dir, batch_size=16, split_ratio=0.8
        )

        print(f"Training on {len(train_loader.dataset)} images, {len(train_loader.dataset.class_names)} elephants")

        best_accuracy = 0
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)

            # Train with classification loss
            train_loss, train_acc = self.train_epoch_classification(train_loader)

            # Train with triplet loss (if enabled)
            if use_triplet and epoch > 10:  # Start triplet training after 10 epochs
                triplet_loss = self.train_epoch_triplet(triplet_loader)
                print(f"Triplet Loss: {triplet_loss:.4f}")

            # Validation
            val_accuracy = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Save metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Accuracy: {val_accuracy:.2f}%")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0

                # Save model
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'class_names': train_loader.dataset.class_names
                }

                torch.save(checkpoint, os.path.join(self.model_save_path, 'best_model.pth'))
                print(f"New best model saved! Accuracy: {best_accuracy:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")
        self.plot_training_history()

        return best_accuracy

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracies
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='orange')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        plt.show()

def main():
    """Main function to run training"""

    # Configuration
    DATA_DIR = "data/flat_dataset"  # Your dataset path
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 20

    print(f"Using device: {DEVICE}")
    print(f"Training on dataset: {DATA_DIR}")

    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Dataset not found at {DATA_DIR}")
        return

    # Count elephants
    num_elephants = len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Found {num_elephants} elephants in dataset")

    # Initialize trainer
    trainer = ElephantTrainer(DATA_DIR, device=DEVICE)

    # Start training
    try:
        best_accuracy = trainer.train(epochs=EPOCHS, use_triplet=True)
        print(f"Training completed successfully! Best accuracy: {best_accuracy:.2f}%")

    except KeyboardInterrupt:
        print("Training interrupted by user")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
