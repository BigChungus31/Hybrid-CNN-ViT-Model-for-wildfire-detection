import os
import pandas as pd
import logging
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ViTFeatureExtractor

logger = logging.getLogger(__name__)

class SatelliteDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(SatelliteDataset, self).__init__(root=root, transform=transform)
        self.root = root

    def __getitem__(self, index):
        try:
            return super(SatelliteDataset, self).__getitem__(index)
        except Exception as e:
            logger.error(f"Error loading image at index {index}: {e}")
            if index > 0:
                return self.__getitem__(index - 1)
            else:
                img = Image.new('RGB', (224, 224), color='black')
                if self.transform:
                    img = self.transform(img)
                return img, 0

def explore_dataset(data_dir, split_name):
    """Explore dataset statistics and class balance"""
    class_dirs = os.listdir(data_dir)
    logger.info(f"Found {len(class_dirs)} classes in {split_name}: {class_dirs}")

    class_counts = {}
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[class_name] = count

    if class_counts:
        df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
        logger.info(f"{split_name} class distribution:\n{df}")

        total = sum(class_counts.values())
        for cls, count in class_counts.items():
            percentage = (count / total) * 100
            if percentage < 10:
                logger.warning(f"Class imbalance detected: {cls} has only {percentage:.2f}% of {split_name} data")

    return class_counts

def get_transforms(config):
    """Get data transforms with ViT normalization"""
    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained(config.vit_model_name)
        vit_mean = feature_extractor.image_mean
        vit_std = feature_extractor.image_std
        logger.info(f"Using ViT normalization - Mean: {vit_mean}, Std: {vit_std}")
    except Exception as e:
        logger.warning(f"Failed to load ViT feature extractor: {e}. Using default normalization.")
        vit_mean = config.default_mean
        vit_std = config.default_std

    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(vit_mean, vit_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(vit_mean, vit_std)
    ])

    return train_transform, val_test_transform

def create_data_loaders(config):
    """Create data loaders with class balancing"""
    config.validate_paths()
    
    # Explore datasets
    train_counts = explore_dataset(config.train_dir, "Training")
    val_counts = explore_dataset(config.val_dir, "Validation")
    test_counts = explore_dataset(config.test_dir, "Test")
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    try:
        train_dataset = SatelliteDataset(config.train_dir, transform=train_transform)
        val_dataset = SatelliteDataset(config.val_dir, transform=val_test_transform)
        test_dataset = SatelliteDataset(config.test_dir, transform=val_test_transform)

        # Handle class imbalance with weighted sampling
        if len(set(train_counts.values())) > 1:
            class_weights = [1.0/train_counts[train_dataset.classes[i]] for i in range(len(train_dataset.classes))]
            sample_weights = [class_weights[label] for _, label in train_dataset.samples]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
                                  num_workers=config.num_workers, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                               num_workers=config.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True)

        logger.info(f"Dataset loaded successfully with {len(train_dataset)} training, "
                    f"{len(val_dataset)} validation, and {len(test_dataset)} test samples")
        logger.info(f"Classes: {train_dataset.classes}")

        return train_loader, val_loader, test_loader, train_dataset.classes

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise