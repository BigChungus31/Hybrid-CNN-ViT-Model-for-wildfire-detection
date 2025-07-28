# Hybrid CNN-ViT Satellite Image Classification

A modular implementation of a hybrid CNN-Vision Transformer model for satellite image classification, combining the strengths of ResNet50 and Vision Transformer architectures.

## Project Structure

```
├── main.py              # Main execution script
├── config.py            # Configuration management
├── data.py              # Data loading and preprocessing
├── model.py             # Hybrid CNN-ViT model definition
├── training.py          # Training logic and utilities
├── evaluation.py        # Model evaluation and metrics
├── visualization.py     # Publication-quality plotting
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Features

- **Hybrid Architecture**: Combines ResNet50 CNN with Vision Transformer for enhanced feature extraction
- **Class Imbalance Handling**: Weighted random sampling for balanced training
- **Mixed Precision Training**: Automatic mixed precision for faster training on supported GPUs
- **Comprehensive Evaluation**: Classification reports, confusion matrices, ROC curves, and PR curves
- **Publication-Quality Plots**: High-resolution plots suitable for academic papers
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd satellite-image-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
Dataset/
├── train/
│   ├── class1/
│   └── class2/
├── valid/
│   ├── class1/
│   └── class2/
├── test/
│   ├── class1/
│   └── class2/
└── output/  # Will be created automatically
```

## Usage

### Full Training and Evaluation
```bash
python main.py
```

### Evaluate Specific Epoch
```bash
python main.py evaluate 10
```

### Custom Data Directory
```bash
python main.py evaluate 10 /path/to/your/dataset
```

## Configuration

Modify `config.py` to adjust:
- Data paths
- Model hyperparameters
- Training parameters
- Device settings

Key parameters:
- `batch_size`: Batch size for training (default: 16)
- `num_epochs`: Number of training epochs (default: 15)
- `learning_rate`: Initial learning rate (default: 2e-5)
- `dropout_rate`: Dropout rate in classifier (default: 0.5)

## Model Architecture

The hybrid model combines:
1. **ResNet50**: Pre-trained CNN backbone with fine-tuned layers 3-4
2. **Vision Transformer**: ViT-base-patch16-224 with fine-tuned layers 8-12
3. **Feature Fusion**: Concatenation of CNN and ViT features
4. **Classification Head**: Multi-layer perceptron with dropout and batch normalization

## Output Files

The model generates:
- **Checkpoints**: `best_model_acc.pth`, `best_model_f1.pth`, `checkpoint_epoch_X.pth`
- **Metrics**: Classification reports as CSV files
- **Visualizations**: Training curves, confusion matrices, ROC/PR curves
- **Publication Plots**: High-resolution figures in `output/paper_plots/`

## Key Features

### Training
- Automatic mixed precision training
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping based on validation metrics
- Comprehensive logging and progress tracking

### Evaluation
- Multi-class classification support
- ROC and Precision-Recall curves for binary classification
- Detailed classification reports
- Confusion matrix visualization

### Visualization
- Publication-quality plots (300 DPI)
- Both PNG and PDF formats
- Comprehensive performance summaries
- Customizable styling and colors

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (automatic fallback to CPU)
- **Memory**: 8GB+ RAM, 4GB+ VRAM for default batch size
- **Storage**: Sufficient space for dataset and model checkpoints

## Logging

All training progress and errors are logged to:
- Console output
- `training.log` file

## Extending the Code

The modular structure makes it easy to:
- Add new model architectures in `model.py`
- Implement custom data augmentations in `data.py`
- Add new evaluation metrics in `evaluation.py`
- Create custom visualizations in `visualization.py`

## Common Issues

### CUDA Out of Memory
- Reduce `batch_size` in `config.py`
- Use gradient accumulation if needed

### Dataset Not Found
- Verify dataset structure matches expected format
- Check paths in `config.py`

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.7+)

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{hybrid-cnn-vit-satellite,
  title={Hybrid CNN-ViT Model for Wildfire Detection},
  author={Abhimanyu Choudhry},
  year={2025},
  url={https://github.com/Hybrid-CNN-ViT-Model-for-wildfire-detection}
}
```