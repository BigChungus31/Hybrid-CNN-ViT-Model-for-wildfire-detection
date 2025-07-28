import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel
import logging

logger = logging.getLogger(__name__)

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5, vit_model_name='google/vit-base-patch16-224-in21k'):
        super(HybridCNNViT, self).__init__()

        # CNN backbone - using ResNet50 with unfrozen layers for fine-tuning
        self.cnn = models.resnet50(pretrained=True)
        # Unfreeze only the last two blocks of ResNet for fine-tuning
        for name, param in self.cnn.named_parameters():
            if "layer4" in name or "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Remove classification layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        # Vision Transformer
        self.vit = ViTModel.from_pretrained(vit_model_name)
        # Freeze early layers of ViT for efficiency
        for name, param in self.vit.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split('.')[2])
                if layer_num < 8:  # Freeze first 8 layers
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True

        # Feature dimensions
        self.cnn_feature_dim = 2048
        self.vit_feature_dim = 768

        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + self.vit_feature_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 2),  # 2 for CNN and ViT weights
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + self.vit_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn(x).view(x.size(0), -1)

        # ViT feature extraction
        vit_output = self.vit(pixel_values=x)
        vit_features = vit_output.last_hidden_state[:, 0, :]  # CLS token

        # Concatenate features
        combined_features = torch.cat((cnn_features, vit_features), dim=1)

        # Classification
        output = self.classifier(combined_features)
        return output

def create_model(config):
    """Create and initialize the hybrid model"""
    try:
        model = HybridCNNViT(
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            vit_model_name=config.vit_model_name
        ).to(config.device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {trainable_params:,} trainable parameters")
        
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise