#file models/few_shot_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import numpy as np
import os

class SiameseEarNetwork(nn.Module):
    """Siamese network for few-shot elephant ear identification"""

    def __init__(self, embedding_dim=256):
        super(SiameseEarNetwork, self).__init__()

        # Backbone - EfficientNet for better performance with small datasets
        self.backbone = efficientnet_b0(pretrained=True)
        backbone_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()

        # Attention mechanism for ear features
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(backbone_features, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        # Feature embedding
        self.embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            L2Norm(dim=1)  # L2 normalization for better similarity computation
        )

    def forward_one(self, x):
        """Forward pass for one image"""
        # Extract features using backbone
        features = self.backbone.features(x)

        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Get embedding
        embedding = self.embedder(attended_features)
        return embedding

    def forward(self, x1, x2=None):
        """Forward pass for siamese network"""
        if x2 is None:
            return self.forward_one(x1)

        # Get embeddings for both images
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)

        return embedding1, embedding2

class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        anchor, positive: same elephant
        anchor, negative: different elephants
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class L2Norm(nn.Module):
    """L2 normalization layer"""
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class ElephantIdentifier:
    """Main class for elephant identification using few-shot learning"""

    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = SiameseEarNetwork().to(device)
        self.elephant_prototypes = {}  # Store prototype embeddings for each elephant

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'prototypes' in checkpoint:
            self.elephant_prototypes = checkpoint['prototypes']

    def save_model(self, model_path):
        """Save model and prototypes"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'prototypes': self.elephant_prototypes
        }, model_path)

    def create_prototypes(self, data_loader):
        """Create prototype embeddings for each elephant class"""
        self.model.eval()
        prototypes = {}

        with torch.no_grad():
            for elephant_id in data_loader.dataset.class_names:
                embeddings = []

                # Get all images for this elephant
                for img, label, _ in data_loader:
                    if data_loader.dataset.class_names[label.item()] == elephant_id:
                        img = img.to(self.device)
                        embedding = self.model(img.unsqueeze(0))
                        embeddings.append(embedding.cpu())

                if embeddings:
                    # Average embeddings to create prototype
                    prototype = torch.stack(embeddings).mean(dim=0)
                    prototypes[elephant_id] = prototype

        self.elephant_prototypes = prototypes
        return prototypes

    def identify_elephant(self, image_tensor, top_k=3):
        """Identify elephant from image using prototype matching"""
        self.model.eval()

        with torch.no_grad():
            # Get embedding for input image
            image_tensor = image_tensor.to(self.device)
            query_embedding = self.model(image_tensor.unsqueeze(0))

            # Compare with all prototypes
            similarities = {}
            for elephant_id, prototype in self.elephant_prototypes.items():
                prototype = prototype.to(self.device)
                similarity = F.cosine_similarity(query_embedding, prototype.unsqueeze(0))
                similarities[elephant_id] = similarity.item()

            # Sort by similarity
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

            return sorted_similarities[:top_k]
