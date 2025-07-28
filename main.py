#!/usr/bin/env python3
"""
Hybrid CNN-ViT Satellite Image Classification
Main execution script for training and evaluation
"""

import os
import logging
import torch

from config import Config
from data import create_data_loaders
from model import create_model
from training import train_model
from evaluation import evaluate_model, load_and_evaluate_checkpoint
from visualization import PublicationPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = Config()
        logger.info(f"Using device: {config.device}")
        
        # Create data loaders
        logger.info("Loading datasets...")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(config)
        config.num_classes = len(class_names)
        
        # Create model
        logger.info("Initializing model...")
        model = create_model(config)
        
        # Train model
        logger.info("Starting training...")
        best_val_acc, best_f1 = train_model(model, train_loader, val_loader, config)
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%, Best F1 score: {best_f1:.4f}")
        
        # Load best model and evaluate
        logger.info("Loading best model for evaluation...")
        best_model_path = os.path.join(config.output_dir, "best_model_f1.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with F1 score {checkpoint['val_f1']:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        criterion = torch.nn.CrossEntropyLoss()
        test_acc, test_preds, test_labels, test_probs = evaluate_model(
            model, test_loader, criterion, class_names, config.output_dir, "Test"
        )
        
        # Generate publication plots
        logger.info("Generating publication-quality plots...")
        plotter = PublicationPlotter(config.output_dir)
        plotter.generate_all_plots(class_names)
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def evaluate_specific_epoch(epoch_num, data_dir=None):
    """Evaluate a specific epoch checkpoint"""
    try:
        config = Config(data_dir)
        
        # Load data
        _, val_loader, test_loader, class_names = create_data_loaders(config)
        config.num_classes = len(class_names)
        
        # Create model
        model = create_model(config)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Load and evaluate specific checkpoint
        checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch_num}.pth")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {epoch_num}")
            
            # Evaluate on validation and test sets
            val_acc, val_preds, val_labels, val_probs = evaluate_model(
                model, val_loader, criterion, class_names, config.output_dir, "Validation"
            )
            test_acc, test_preds, test_labels, test_probs = evaluate_model(
                model, test_loader, criterion, class_names, config.output_dir, "Test"
            )
            
            logger.info(f"Validation accuracy: {val_acc:.2f}%")
            logger.info(f"Test accuracy: {test_acc:.2f}%")
            
            return val_acc, test_acc
        else:
            logger.error(f"Checkpoint for epoch {epoch_num} not found at {checkpoint_path}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error evaluating epoch {epoch_num}: {e}")
        return None, None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "evaluate" and len(sys.argv) > 2:
            # Evaluate specific epoch: python main.py evaluate 10
            epoch = int(sys.argv[2])
            data_dir = sys.argv[3] if len(sys.argv) > 3 else None
            evaluate_specific_epoch(epoch, data_dir)
        else:
            print("Usage:")
            print("  python main.py                    # Full training and evaluation")
            print("  python main.py evaluate <epoch>   # Evaluate specific epoch")
    else:
        # Full training and evaluation
        exit_code = main()
        sys.exit(exit_code)