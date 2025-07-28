import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)

def setup_training(model, config):
    """Setup training components"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    return criterion, optimizer, scheduler, scaler

def train_model(model, train_loader, val_loader, config):
    """Train the model with improved monitoring and checkpointing"""
    criterion, optimizer, scheduler, scaler = setup_training(model, config)
    
    best_val_acc = 0.0
    best_f1 = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'lr': []
    }

    logger.info(f"Starting training for {config.num_epochs} epochs")

    try:
        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            train_loss, correct, total = 0.0, 0, 0

            with tqdm(train_loader, unit="batch") as t:
                t.set_description(f"Epoch {epoch+1}/{config.num_epochs}")

                for images, labels in t:
                    images, labels = images.to(config.device), labels.to(config.device)
                    optimizer.zero_grad()

                    # Forward pass with mixed precision if available
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    # Statistics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    t.set_postfix(loss=train_loss/len(t), acc=100.*correct/total)

            # Calculate epoch statistics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation phase
            val_loss, val_acc, val_f1 = validate_model(model, val_loader, criterion, config.device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            # Log results
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, F1: {val_f1:.4f}")
            logger.info(f"  Learning Rate: {current_lr}")

            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            # Save best models
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, scheduler, epoch + 1, best_val_acc, val_f1, 
                              config.output_dir, "best_model_acc.pth")
                logger.info(f"  Saved new best model with accuracy: {best_val_acc:.2f}%")

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc, best_f1,
                              config.output_dir, "best_model_f1.pth")
                logger.info(f"  Saved new best model with F1 score: {best_f1:.4f}")

            # Save epoch checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc, val_f1,
                          config.output_dir, f"checkpoint_epoch_{epoch+1}.pth")

        # Save training history and plot curves
        save_training_history(history, config.output_dir)
        plot_training_curves(history, config.output_dir)

        return best_val_acc, best_f1

    except Exception as e:
        logger.error(f"Training error: {e}")
        save_checkpoint(model, optimizer, scheduler, 0, 0, 0,
                      config.output_dir, "emergency_checkpoint.pth")
        logger.info("Saved emergency checkpoint due to error")
        raise

def validate_model(model, val_loader, criterion, device):
    """Validate the model and return metrics"""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_loss, val_acc, val_f1

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_f1, output_dir, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': val_acc,
        'val_f1': val_f1
    }, os.path.join(output_dir, filename))

def save_training_history(history, output_dir):
    """Save training history to CSV"""
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()