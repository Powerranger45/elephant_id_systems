#file : utils/data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import random
from collections import Counter, defaultdict

class SmartElephantDataset(Dataset):
    """Smart dataset with adaptive augmentation based on class size"""

    def __init__(self, data_dir, transform=None, mode='train', augment_factor=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.images = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}

        self._load_data()
        if mode == 'train' and augment_factor:
            self._apply_smart_augmentation(augment_factor)

    def _load_data(self):
        """Load data and analyze class distribution"""
        class_counts = defaultdict(list)

        # First pass: collect all images
        for elephant_folder in sorted(os.listdir(self.data_dir)):
            elephant_path = os.path.join(self.data_dir, elephant_folder)
            if not os.path.isdir(elephant_path):
                continue

            for img_file in os.listdir(elephant_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(elephant_path, img_file)
                    class_counts[elephant_folder].append(img_path)

        # Create class mapping
        for idx, elephant_id in enumerate(sorted(class_counts.keys())):
            self.class_names.append(elephant_id)
            self.class_to_idx[elephant_id] = idx

        # Add images to dataset
        for elephant_id, img_paths in class_counts.items():
            class_idx = self.class_to_idx[elephant_id]
            for img_path in img_paths:
                self.images.append(img_path)
                self.labels.append(class_idx)

        print(f"Loaded {len(self.images)} images from {len(self.class_names)} elephants")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution statistics"""
        class_counts = Counter(self.labels)
        counts = list(class_counts.values())

        print(f"Class distribution:")
        print(f"  Min images per class: {min(counts)}")
        print(f"  Max images per class: {max(counts)}")
        print(f"  Average images per class: {np.mean(counts):.1f}")
        print(f"  Classes with <3 images: {sum(1 for c in counts if c < 3)}")
        print(f"  Classes with 3-5 images: {sum(1 for c in counts if 3 <= c <= 5)}")
        print(f"  Classes with >5 images: {sum(1 for c in counts if c > 5)}")

    def _apply_smart_augmentation(self, base_factor):
        """Apply different augmentation levels based on class size"""
        class_counts = Counter(self.labels)
        augmented_images = []
        augmented_labels = []

        for class_idx, count in class_counts.items():
            # Determine augmentation factor based on class size
            if count == 1:
                 aug_factor = base_factor * 5
            elif count == 2:
                 aug_factor = base_factor * 4
            elif count <= 3:
                 aug_factor = base_factor * 3
            elif count <= 5:
                 aug_factor = base_factor * 2
            else:
                aug_factor = base_factor  # no extra
# Minimal augmentation for well-represented classes

            # Get images for this class
            class_images = [img for img, label in zip(self.images, self.labels) if label == class_idx]

            # Generate augmented versions
            for _ in range(int(aug_factor)):
                # Randomly select an image from this class
                img_path = random.choice(class_images)
                augmented_images.append(img_path)
                augmented_labels.append(class_idx)

        # Add augmented data to original data
        self.images.extend(augmented_images)
        self.labels.extend(augmented_labels)

        print(f"After augmentation: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = Image.fromarray(image)
                image = self.transform(image)

        return image, label, img_path

class TripletDataset(Dataset):
    """Dataset for triplet loss training"""

    def __init__(self, base_dataset, samples_per_class=10):
        self.base_dataset = base_dataset
        self.samples_per_class = samples_per_class

        # Group images by class
        self.class_to_images = defaultdict(list)
        for i, (_, label, path) in enumerate(base_dataset):
            self.class_to_images[label].append(i)

        self.classes = list(self.class_to_images.keys())

    def __len__(self):
        return len(self.classes) * self.samples_per_class

    def __getitem__(self, idx):
        # Select anchor class
        class_idx = idx // self.samples_per_class
        anchor_class = self.classes[class_idx]

        # Select anchor image
        anchor_idx = random.choice(self.class_to_images[anchor_class])
        anchor_img, anchor_label, _ = self.base_dataset[anchor_idx]

        # Select positive image (same class, different image)
        positive_candidates = [i for i in self.class_to_images[anchor_class] if i != anchor_idx]
        if positive_candidates:
            positive_idx = random.choice(positive_candidates)
        else:
            positive_idx = anchor_idx  # Use same image if only one available
        positive_img, positive_label, _ = self.base_dataset[positive_idx]

        # Select negative image (different class)
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_idx = random.choice(self.class_to_images[negative_class])
        negative_img, negative_label, _ = self.base_dataset[negative_idx]

        return (anchor_img, positive_img, negative_img), (anchor_label, positive_label, negative_label)

def get_transforms(mode='train'):
    """Get transforms for training/validation"""
    if mode == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=20, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=0.3),
                A.GridDistortion(distort_limit=0.1, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.6),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, mask_fill_value=None, p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_data_loaders(data_dir, batch_size=8, split_ratio=0.8):
    """Create train and validation data loaders"""

    # Get transforms
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')

    # Create full dataset first
    full_dataset = SmartElephantDataset(data_dir, transform=val_transform, mode='val')

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size

    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create train dataset with augmentation
    train_dataset = SmartElephantDataset(data_dir, transform=train_transform, mode='train', augment_factor=2)

    # Create triplet dataset for metric learning
    triplet_dataset = TripletDataset(train_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_indices),
                           batch_size=batch_size, shuffle=False, num_workers=2)
    triplet_loader = DataLoader(triplet_dataset, batch_size=batch_size//2, shuffle=True, num_workers=2)

    return train_loader, val_loader, triplet_loader
