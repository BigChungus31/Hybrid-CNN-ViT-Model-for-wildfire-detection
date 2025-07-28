# Hybrid CNN-ViT Satellite Image Classification

A modular implementation of a hybrid CNN-Vision Transformer model for satellite image classification, combining the strengths of ResNet50 and Vision Transformer architectures.

## Application Context

While this model is trained on wildfire satellite data, its intended application is the detection and mapping of **stubble burning events** using satellite imagery. Due to limited open-source datasets specific to crop residue fires, wildfire data serves as a suitable proxy for model development and experimentation.

This approach can assist in identifying stubble burning hotspots, supporting air quality management and climate policy research.

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
## Dataset

This project uses the [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) from Kaggle.

Due to licensing restrictions, the dataset is not included in this repository.

**To use this project:**
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset).

## About the Dataset

The dataset used in this project was sourced from Canada’s Open Government Portal and is licensed under the Creative Commons 4.0 Attribution (CC BY) license by Quebec. It comprises 42,750 satellite images (350×350 px), divided into:

22,710 wildfire images (53.1%)

20,040 no-wildfire images (46.9%)

Images were collected using Mapbox API based on GPS coordinates of wildfire events (>0.01 acres burned) from the Canadian Wildfire Database. This ensures a rich, varied dataset for deep learning, with images captured across different seasons and environmental conditions.

The data was split as follows:

70% Training (29,925 images)

15% Validation (6,413 images)

15% Testing (6,412 images)

This structure supports robust training, hyperparameter tuning, and evaluation for wildfire risk prediction using satellite imagery.

## Sample Images from the Dataset

![No Wildfire image](https://github.com/user-attachments/assets/62ceedd2-8335-4466-91e2-433cf90b80af)  
*No Wildfire Image*

![Wildfire image](https://github.com/user-attachments/assets/85322fc2-7d8a-4522-a261-3db26488c76a)  
*Wildfire Image*

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

## Output

Confusion Matrices

<img width="1000" height="800" alt="test_confusion_matrix" src="https://github.com/user-attachments/assets/719e2e6d-d43e-4bea-ad93-9afdfa0edea2" />  

<img width="1000" height="800" alt="validation_confusion_matrix" src="https://github.com/user-attachments/assets/d90ac589-4006-427a-ba23-5e0da03cb441" /> 

Test Metrics

| Class        | Precision | Recall | F1-Score   | Support |
| ------------ | --------- | ------ | ---------- | ------- |
| nowildfire   | 0.9894    | 0.9957 | 0.9926     | 2820    |
| wildfire     | 0.9965    | 0.9914 | 0.9939     | 3480    |
| **Accuracy** |           |        | **0.9933** | 6300    |
| Macro Avg    | 0.9930    | 0.9936 | 0.9933     | 6300    |
| Weighted Avg | 0.9934    | 0.9933 | 0.9933     | 6300    |

Validation Metrics

| Class        | Precision | Recall | F1-Score   | Support |
| ------------ | --------- | ------ | ---------- | ------- |
| nowildfire   | 0.9887    | 0.9943 | 0.9915     | 2820    |
| wildfire     | 0.9954    | 0.9908 | 0.9931     | 3480    |
| **Accuracy** |           |        | **0.9924** | 6300    |
| Macro Avg    | 0.9920    | 0.9926 | 0.9923     | 6300    |
| Weighted Avg | 0.9924    | 0.9924 | 0.9924     | 6300    |




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

## License
This project is licensed under the [Apache 2.0 License](./LICENSE).

See the LICENSE file for more details.

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
