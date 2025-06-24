#file: utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging


class FewShotMetrics:
    """Comprehensive metrics for few-shot elephant identification"""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.predictions = []
        self.ground_truth = []
        self.similarities = []
        self.embeddings = []

    def reset(self):
        """Reset all stored metrics"""
        self.predictions = []
        self.ground_truth = []
        self.similarities = []
        self.embeddings = []

    def update(self, predictions: List[str], ground_truth: List[str],
               similarities: Optional[List[float]] = None):
        """Update metrics with new predictions"""
        self.predictions.extend(predictions)
        self.ground_truth.extend(ground_truth)
        if similarities:
            self.similarities.extend(similarities)

    def compute_basic_metrics(self) -> Dict[str, float]:
        """Compute basic classification metrics"""
        if not self.predictions or not self.ground_truth:
            return {}

        metrics = {
            'accuracy': accuracy_score(self.ground_truth, self.predictions),
            'precision_macro': precision_score(self.ground_truth, self.predictions,
                                             average='macro', zero_division=0),
            'precision_micro': precision_score(self.ground_truth, self.predictions,
                                             average='micro', zero_division=0),
            'recall_macro': recall_score(self.ground_truth, self.predictions,
                                       average='macro', zero_division=0),
            'recall_micro': recall_score(self.ground_truth, self.predictions,
                                       average='micro', zero_division=0),
            'f1_macro': f1_score(self.ground_truth, self.predictions,
                               average='macro', zero_division=0),
            'f1_micro': f1_score(self.ground_truth, self.predictions,
                               average='micro', zero_division=0)
        }

        return metrics

    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics"""
        if not self.predictions or not self.ground_truth:
            return {}

        # Get unique classes from both predictions and ground truth
        all_classes = sorted(set(self.ground_truth + self.predictions))

        precision_scores = precision_score(self.ground_truth, self.predictions,
                                         labels=all_classes, average=None, zero_division=0)
        recall_scores = recall_score(self.ground_truth, self.predictions,
                                   labels=all_classes, average=None, zero_division=0)
        f1_scores = f1_score(self.ground_truth, self.predictions,
                           labels=all_classes, average=None, zero_division=0)

        per_class_metrics = {}
        for i, class_name in enumerate(all_classes):
            per_class_metrics[class_name] = {
                'precision': precision_scores[i] if i < len(precision_scores) else 0.0,
                'recall': recall_scores[i] if i < len(recall_scores) else 0.0,
                'f1': f1_scores[i] if i < len(f1_scores) else 0.0,
                'support': self.ground_truth.count(class_name)
            }

        return per_class_metrics

    def compute_top_k_accuracy(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """Compute top-k accuracy (requires similarity scores)"""
        if not hasattr(self, 'top_k_predictions') or not self.top_k_predictions:
            return {f'top_{k}_accuracy': 0.0 for k in k_values}

        top_k_acc = {}
        for k in k_values:
            correct = 0
            total = len(self.top_k_predictions)

            for i, (true_label, top_k_preds) in enumerate(zip(self.ground_truth, self.top_k_predictions)):
                if len(top_k_preds) >= k and true_label in [pred[0] for pred in top_k_preds[:k]]:
                    correct += 1

            top_k_acc[f'top_{k}_accuracy'] = correct / max(total, 1)

        return top_k_acc

    def update_with_top_k(self, ground_truth: List[str], top_k_predictions: List[List[Tuple[str, float]]]):
        """Update metrics with top-k predictions"""
        self.ground_truth.extend(ground_truth)
        if not hasattr(self, 'top_k_predictions'):
            self.top_k_predictions = []
        self.top_k_predictions.extend(top_k_predictions)

        # Extract top-1 predictions for basic metrics
        top_1_preds = [preds[0][0] if preds else "unknown" for preds in top_k_predictions]
        self.predictions.extend(top_1_preds)

    def plot_confusion_matrix(self, save_path: str = "confusion_matrix.png",
                            figsize: Tuple[int, int] = (12, 10)):
        """Plot confusion matrix heatmap"""
        if not self.predictions or not self.ground_truth:
            return

        # Get all unique classes
        all_classes = sorted(set(self.ground_truth + self.predictions))

        cm = confusion_matrix(self.ground_truth, self.predictions, labels=all_classes)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=all_classes, yticklabels=all_classes)
        plt.title('Confusion Matrix - Elephant Identification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_per_class_metrics(self, save_path: str = "per_class_metrics.png"):
        """Plot per-class precision, recall, and F1 scores"""
        per_class = self.compute_per_class_metrics()
        if not per_class:
            return

        classes = list(per_class.keys())
        precision = [per_class[c]['precision'] for c in classes]
        recall = [per_class[c]['recall'] for c in classes]
        f1 = [per_class[c]['f1'] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.8), 6))

        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Elephant ID')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def compute_similarity_statistics(self) -> Dict[str, float]:
        """Compute statistics about similarity scores"""
        if not self.similarities:
            return {}

        similarities = np.array(self.similarities)

        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'median_similarity': np.median(similarities)
        }

    def generate_classification_report(self) -> str:
        """Generate detailed classification report"""
        if not self.predictions or not self.ground_truth:
            return "No predictions available"

        return classification_report(self.ground_truth, self.predictions, zero_division=0)

    def compute_all_metrics(self) -> Dict:
        """Compute all available metrics"""
        metrics = {
            'basic_metrics': self.compute_basic_metrics(),
            'per_class_metrics': self.compute_per_class_metrics(),
            'similarity_stats': self.compute_similarity_statistics(),
            'classification_report': self.generate_classification_report()
        }

        # Add top-k accuracy if available
        if hasattr(self, 'top_k_predictions'):
            metrics['top_k_accuracy'] = self.compute_top_k_accuracy()

        return metrics


class EmbeddingMetrics:
    """Metrics for evaluating embedding quality"""

    def __init__(self):
        self.embeddings = []
        self.labels = []

    def update(self, embeddings: torch.Tensor, labels: List[str]):
        """Update with new embeddings and labels"""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        self.embeddings.extend(embeddings)
        self.labels.extend(labels)

    def compute_intra_class_distances(self) -> Dict[str, Dict[str, float]]:
        """Compute intra-class distances (within same elephant)"""
        if not self.embeddings:
            return {}

        embeddings = np.array(self.embeddings)
        intra_distances = defaultdict(list)

        # Group embeddings by label
        label_embeddings = defaultdict(list)
        for i, label in enumerate(self.labels):
            label_embeddings[label].append(embeddings[i])

        # Compute pairwise distances within each class
        for label, embs in label_embeddings.items():
            if len(embs) > 1:
                embs = np.array(embs)
                for i in range(len(embs)):
                    for j in range(i + 1, len(embs)):
                        dist = np.linalg.norm(embs[i] - embs[j])
                        intra_distances[label].append(dist)

        # Compute statistics
        intra_stats = {}
        for label, distances in intra_distances.items():
            if distances:
                intra_stats[label] = {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'max': np.max(distances),
                    'count': len(distances)
                }

        return intra_stats

    def compute_inter_class_distances(self) -> Dict[str, float]:
        """Compute inter-class distances (between different elephants)"""
        if not self.embeddings:
            return {}

        embeddings = np.array(self.embeddings)
        inter_distances = []

        # Group embeddings by label
        label_embeddings = defaultdict(list)
        for i, label in enumerate(self.labels):
            label_embeddings[label].append(embeddings[i])

        labels = list(label_embeddings.keys())

        # Compute distances between different classes
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                embs1 = np.array(label_embeddings[label1])
                embs2 = np.array(label_embeddings[label2])

                for emb1 in embs1:
                    for emb2 in embs2:
                        dist = np.linalg.norm(emb1 - emb2)
                        inter_distances.append(dist)

        if inter_distances:
            return {
                'mean': np.mean(inter_distances),
                'std': np.std(inter_distances),
                'min': np.min(inter_distances),
                'max': np.max(inter_distances),
                'count': len(inter_distances)
            }

        return {}

    def compute_silhouette_score(self) -> float:
        """Compute silhouette score for embeddings"""
        try:
            from sklearn.metrics import silhouette_score

            if len(self.embeddings) < 2 or len(set(self.labels)) < 2:
                return 0.0

            embeddings = np.array(self.embeddings)
            return silhouette_score(embeddings, self.labels)

        except ImportError:
            logging.warning("scikit-learn not available for silhouette score")
            return 0.0

    def plot_embedding_distribution(self, save_path: str = "embedding_distribution.png"):
        """Plot embedding distance distributions"""
        intra_stats = self.compute_intra_class_distances()
        inter_stats = self.compute_inter_class_distances()

        if not intra_stats or not inter_stats:
            return

        # Collect all intra-class distances
        all_intra = []
        for stats in intra_stats.values():
            # Generate sample distances based on statistics (approximation)
            mean, std = stats['mean'], stats['std']
            count = min(stats['count'], 100)  # Limit for visualization
            samples = np.random.normal(mean, std, count)
            all_intra.extend(samples)

        # Generate inter-class distances (approximation)
        inter_mean, inter_std = inter_stats['mean'], inter_stats['std']
        inter_count = min(inter_stats['count'], 1000)
        all_inter = np.random.normal(inter_mean, inter_std, inter_count)

        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.hist(all_intra, bins=50, alpha=0.7, label='Intra-class distances', density=True)
        plt.hist(all_inter, bins=50, alpha=0.7, label='Inter-class distances', density=True)
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Density')
        plt.title('Embedding Distance Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ValidationMetrics:
    """Metrics for validation and testing phases"""

    def __init__(self, identifier, val_loader):
        self.identifier = identifier
        self.val_loader = val_loader
        self.metrics = FewShotMetrics([])

    def evaluate_k_shot_performance(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, Dict]:
        """Evaluate performance with different k-shot scenarios"""
        results = {}

        for k in k_values:
            # Create k-shot scenario
            k_shot_data = self._create_k_shot_scenario(k)

            # Evaluate on this scenario
            metrics = self._evaluate_scenario(k_shot_data)
            results[f'{k}_shot'] = metrics

        return results

    def _create_k_shot_scenario(self, k: int) -> Dict:
        """Create k-shot learning scenario"""
        # Group images by elephant
        elephant_images = defaultdict(list)

        for images, labels, _ in self.val_loader:
            for img, label in zip(images, labels):
                elephant_id = self.val_loader.dataset.class_names[label.item()]
                elephant_images[elephant_id].append((img, elephant_id))

        # Select k images per elephant for support set
        support_set = {}
        query_set = []

        for elephant_id, images in elephant_images.items():
            if len(images) >= k + 1:  # Need at least k+1 images
                # Randomly select k images for support
                selected_indices = np.random.choice(len(images), k, replace=False)
                support_images = [images[i] for i in selected_indices]

                # Rest for query
                query_images = [images[i] for i in range(len(images)) if i not in selected_indices]

                support_set[elephant_id] = support_images
                query_set.extend(query_images)

        return {'support': support_set, 'query': query_set}

    def _evaluate_scenario(self, scenario_data: Dict) -> Dict:
        """Evaluate a specific k-shot scenario"""
        # This is a simplified evaluation - in practice, you'd retrain prototypes
        # with only the support set and evaluate on the query set

        predictions = []
        ground_truth = []

        for img, true_label in scenario_data['query']:
            pred_results = self.identifier.identify_elephant(img, top_k=1)
            if pred_results:
                predictions.append(pred_results[0][0])
            else:
                predictions.append("unknown")
            ground_truth.append(true_label)

        # Compute basic metrics
        if predictions and ground_truth:
            accuracy = accuracy_score(ground_truth, predictions)
            return {
                'accuracy': accuracy,
                'num_queries': len(ground_truth),
                'num_support_classes': len(scenario_data['support'])
            }

        return {'accuracy': 0.0, 'num_queries': 0, 'num_support_classes': 0}


def create_comprehensive_report(metrics: FewShotMetrics, embedding_metrics: EmbeddingMetrics,
                              save_path: str = "evaluation_report.txt") -> str:
    """Create a comprehensive evaluation report"""

    all_metrics = metrics.compute_all_metrics()

    report = []
    report.append("=" * 60)
    report.append("ELEPHANT IDENTIFICATION EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Basic metrics
    if 'basic_metrics' in all_metrics:
        report.append("BASIC CLASSIFICATION METRICS:")
        report.append("-" * 30)
        for metric, value in all_metrics['basic_metrics'].items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        report.append("")

    # Top-k accuracy
    if 'top_k_accuracy' in all_metrics:
        report.append("TOP-K ACCURACY:")
        report.append("-" * 15)
        for metric, value in all_metrics['top_k_accuracy'].items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        report.append("")

    # Similarity statistics
    if 'similarity_stats' in all_metrics and all_metrics['similarity_stats']:
        report.append("SIMILARITY STATISTICS:")
        report.append("-" * 22)
        for stat, value in all_metrics['similarity_stats'].items():
            report.append(f"{stat.replace('_', ' ').title()}: {value:.4f}")
        report.append("")

    # Embedding quality metrics
    intra_stats = embedding_metrics.compute_intra_class_distances()
    inter_stats = embedding_metrics.compute_inter_class_distances()

    if intra_stats and inter_stats:
        report.append("EMBEDDING QUALITY METRICS:")
        report.append("-" * 26)

        # Average intra-class distance
        avg_intra = np.mean([stats['mean'] for stats in intra_stats.values()])
        report.append(f"Average Intra-class Distance: {avg_intra:.4f}")
        report.append(f"Average Inter-class Distance: {inter_stats['mean']:.4f}")

        # Separation ratio
        separation_ratio = inter_stats['mean'] / avg_intra if avg_intra > 0 else 0
        report.append(f"Separation Ratio (Inter/Intra): {separation_ratio:.4f}")
        report.append("")

    # Silhouette score
    silhouette = embedding_metrics.compute_silhouette_score()
    if silhouette:
        report.append(f"Silhouette Score: {silhouette:.4f}")
        report.append("")

    # Per-class performance
    if 'per_class_metrics' in all_metrics:
        report.append("PER-CLASS PERFORMANCE:")
        report.append("-" * 22)
        for class_name, class_metrics in all_metrics['per_class_metrics'].items():
            report.append(f"\n{class_name}:")
            for metric, value in class_metrics.items():
                if metric != 'support':
                    report.append(f"  {metric.title()}: {value:.4f}")
                else:
                    report.append(f"  {metric.title()}: {value}")
        report.append("")

    # Classification report
    if 'classification_report' in all_metrics:
        report.append("DETAILED CLASSIFICATION REPORT:")
        report.append("-" * 33)
        report.append(all_metrics['classification_report'])

    report_text = "\n".join(report)

    # Save report
    with open(save_path, 'w') as f:
        f.write(report_text)

    return report_text
