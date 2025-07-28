import os
import torch

class Config:
    def __init__(self, data_dir=None):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mixed precision training
        self.use_mixed_precision = torch.cuda.is_available()
        
        # Data paths
        self.data_dir = data_dir or "/content/drive/MyDrive/Dataset"
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'valid')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.output_dir = os.path.join(self.data_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model parameters
        self.num_classes = 2
        self.dropout_rate = 0.5
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 15
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.num_workers = 2
        
        # ViT parameters
        self.vit_model_name = 'google/vit-base-patch16-224-in21k'
        self.image_size = 224
        
        # Default normalization values
        self.default_mean = [0.485, 0.456, 0.406]
        self.default_std = [0.229, 0.224, 0.225]
    
    def validate_paths(self):
        """Check if required directories exist"""
        for directory in [self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")
        return True