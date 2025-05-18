import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch.nn.functional as F

from model_new import EfficientLaFNetDual
from preprocessing_new import get_dataloaders

# Settings
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 400
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 16

class HybridLoss(nn.Module):
    """Combined structure-aware loss for localization"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def edge_loss(self, pred, target):
        # Compute vertical and horizontal gradients
        pred_vgrad = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_hgrad = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        target_vgrad = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_hgrad = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Compute L1 loss for both gradients
        v_loss = torch.mean(torch.abs(pred_vgrad - target_vgrad))
        h_loss = torch.mean(torch.abs(pred_hgrad - target_hgrad))
        
        return (v_loss + h_loss) / 2.0
    
    def forward(self, pred, target):
        # Resize prediction to match target
        pred_resized = F.interpolate(pred, size=target.shape[2:], mode='bilinear')
        return self.bce(pred_resized, target) + 0.5*self.edge_loss(torch.sigmoid(pred_resized), target)

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Classification logger
        self.class_csv_path = os.path.join(log_dir, 'classification_log.csv')
        self.class_csv_file = open(self.class_csv_path, 'w', newline='', encoding='utf-8')
        self.class_writer = csv.writer(self.class_csv_file)
        self.class_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train Precision', 'Train Recall', 
                              'Train F1', 'Val Loss', 'Val Acc', 'Val Precision', 'Val Recall', 
                              'Val F1'])
        
        # Localization logger
        self.loc_csv_path = os.path.join(log_dir, 'localization_log.csv')
        self.loc_csv_file = open(self.loc_csv_path, 'w', newline='', encoding='utf-8')
        self.loc_writer = csv.writer(self.loc_csv_file)
        self.loc_writer.writerow(['Epoch', 'Train Loss', 'Train F1', 'Train Acc', 'Train IoU', 
                              'Train Dice', 'Val Loss', 'Val F1', 'Val Acc', 'Val IoU', 
                              'Val Dice'])
        
    def log_classification(self, epoch, train_loss, train_acc, train_prec, train_rec, train_f1,
                         val_loss, val_acc, val_prec, val_rec, val_f1):
        self.class_writer.writerow([epoch, train_loss, train_acc, train_prec, train_rec, train_f1,
                              val_loss, val_acc, val_prec, val_rec, val_f1])
        self.class_csv_file.flush()
    
    def log_localization(self, epoch, train_loss, train_f1, train_acc, train_iou, train_dice,
                        val_loss, val_f1, val_acc, val_iou, val_dice):
        self.loc_writer.writerow([epoch, train_loss, train_f1, train_acc, train_iou, train_dice,
                                val_loss, val_f1, val_acc, val_iou, val_dice])
        self.loc_csv_file.flush()
        
    def close(self):
        self.class_csv_file.close()
        self.loc_csv_file.close()

def calculate_classification_metrics(y_true, y_pred, num_classes):
    """Calculate precision, recall, and F1 for classification"""
    if not len(y_true) or not len(y_pred):
        return 0, 0, 0
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return precision, recall, f1

def calculate_localization_metrics(preds, masks):
    """Calculate metrics for localization (segmentation)"""
    preds = preds.float()
    masks = masks.float()
    
    # Use batched approach to avoid memory overflow
    batch_size = 4  # Process 4 images at a time
    num_batches = (preds.size(0) + batch_size - 1) // batch_size  # Ceiling division
    
    # Initialize metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    # Process in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, preds.size(0))
        
        batch_preds = preds[start_idx:end_idx]
        batch_masks = masks[start_idx:end_idx]
        
        # Calculate basic metrics for this batch
        tp = (batch_preds * batch_masks).sum().item()
        fp = (batch_preds * (1 - batch_masks)).sum().item()
        fn = ((1 - batch_preds) * batch_masks).sum().item()
        tn = ((1 - batch_preds) * (1 - batch_masks)).sum().item()
        
        # Accumulate metrics
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
    
    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)  # Intersection over Union
    dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)  # Dice coefficient
    
    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'dice': dice
    }

def train_one_epoch(model, dataloaders, criteria, optimizer, device, num_classes, epoch=0):
    """
    Train model for one epoch on both classification and localization tasks
    
    Args:
        model: The model to train
        dataloaders: Dictionary containing classification and localization dataloaders
        criteria: Dictionary containing loss functions for both tasks
        optimizer: Optimizer for training
        device: Device to train on
        num_classes: Number of classes for classification
        epoch: Current epoch number
    """
    model.train()
    classification_loss = 0
    localization_loss = 0
    
    # For classification metrics
    all_class_targets = []
    all_class_predictions = []
    
    # For localization metrics
    # Initialize accumulated metrics
    loc_metrics_accum = {
        'loss': 0,
        'f1': 0,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'iou': 0,
        'dice': 0
    }
    loc_batch_count = 0
    
    # Check if both tasks are available
    has_classification = 'classification' in dataloaders
    has_localization = 'localization' in dataloaders
    
    # Number of batches per epoch
    if has_classification and has_localization:
        classification_len = len(dataloaders['classification']['train'])
        localization_len = len(dataloaders['localization']['train'])
        total_batches = max(classification_len, localization_len)
    elif has_classification:
        total_batches = len(dataloaders['classification']['train'])
    else:
        total_batches = len(dataloaders['localization']['train'])
    
    # Get iterators for dataloaders
    if has_classification:
        classification_iter = iter(dataloaders['classification']['train'])
    if has_localization:
        localization_iter = iter(dataloaders['localization']['train'])
    
    # Progress bar
    progress = tqdm(range(total_batches), desc=f'Epoch {epoch+1}')
    
    optimizer.zero_grad()
    
    for batch_idx in progress:
        batch_loss = 0
        
        # Train classification if available
        if has_classification:
            try:
                images, targets = next(classification_iter)
            except StopIteration:
                classification_iter = iter(dataloaders['classification']['train'])
                images, targets = next(classification_iter)
                
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass for classification
            outputs = model(images, mode='classification')
            class_loss = criteria['classification'](outputs, targets)
            
            # Track metrics
            _, predicted = outputs.max(1)
            all_class_targets.extend(targets.cpu().numpy())
            all_class_predictions.extend(predicted.cpu().numpy())
            
            # Update total loss
            classification_loss += class_loss.item()
            batch_loss += class_loss
        
        # Train localization if available
        if has_localization:
            try:
                images, masks = next(localization_iter)
            except StopIteration:
                localization_iter = iter(dataloaders['localization']['train'])
                images, masks = next(localization_iter)
                
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass for localization
            outputs = model(images, mode='localization')
            loc_loss = criteria['localization'](outputs, masks)
            
            # Track metrics - process immediately instead of storing
            outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
            preds = (torch.sigmoid(outputs_resized) > 0.5).float()
            
            # Calculate metrics for this batch
            batch_metrics = calculate_localization_metrics(preds.detach().cpu(), masks.detach().cpu())
            
            # Accumulate metrics
            for k in loc_metrics_accum.keys():
                if k == 'loss':
                    loc_metrics_accum[k] += loc_loss.item()
                else:
                    loc_metrics_accum[k] += batch_metrics[k]
            
            loc_batch_count += 1
            
            # Update total loss
            localization_loss += loc_loss.item()
            batch_loss += loc_loss
        
        # Backward pass
        batch_loss = batch_loss / GRADIENT_ACCUMULATION_STEPS
        batch_loss.backward()
        
        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Compute metrics
    results = {}
    
    if has_classification:
        # Classification metrics
        class_accuracy = accuracy_score(all_class_targets, all_class_predictions)
        class_precision, class_recall, class_f1 = calculate_classification_metrics(
            all_class_targets, all_class_predictions, num_classes
        )
        
        results['classification'] = {
            'loss': classification_loss / len(dataloaders['classification']['train']),
            'accuracy': class_accuracy * 100,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        }
    
    if has_localization and loc_batch_count > 0:
        # Average localization metrics
        loc_metrics = {}
        for k in loc_metrics_accum.keys():
            if k == 'loss':
                loc_metrics[k] = loc_metrics_accum[k] / len(dataloaders['localization']['train'])
            else:
                loc_metrics[k] = loc_metrics_accum[k] / loc_batch_count
        
        results['localization'] = loc_metrics
    
    return results

def validate(model, dataloaders, criteria, device, num_classes):
    """
    Validate model on both classification and localization tasks
    
    Args:
        model: The model to validate
        dataloaders: Dictionary containing classification and localization dataloaders
        criteria: Dictionary containing loss functions for both tasks
        device: Device to validate on
        num_classes: Number of classes for classification
    """
    model.eval()
    results = {}
    
    # Check if both tasks are available
    has_classification = 'classification' in dataloaders
    has_localization = 'localization' in dataloaders
    
    # For classification
    if has_classification:
        class_loss = 0
        all_class_targets = []
        all_class_predictions = []
        
        with torch.no_grad():
            for images, targets in dataloaders['classification']['val']:
                images, targets = images.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(images, mode='classification')
                class_loss += criteria['classification'](outputs, targets).item()
                
                # Track metrics
                _, predicted = outputs.max(1)
                all_class_targets.extend(targets.cpu().numpy())
                all_class_predictions.extend(predicted.cpu().numpy())
        
        # Compute metrics
        class_accuracy = accuracy_score(all_class_targets, all_class_predictions)
        class_precision, class_recall, class_f1 = calculate_classification_metrics(
            all_class_targets, all_class_predictions, num_classes
        )
        
        results['classification'] = {
            'loss': class_loss / len(dataloaders['classification']['val']),
            'accuracy': class_accuracy * 100,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'confusion_matrix': confusion_matrix(all_class_targets, all_class_predictions)
        }
    
    # For localization
    if has_localization:
        loc_loss = 0
        # Initialize accumulated metrics 
        loc_metrics_accum = {
            'f1': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'iou': 0,
            'dice': 0
        }
        loc_batch_count = 0
        
        with torch.no_grad():
            for images, masks in dataloaders['localization']['val']:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(images, mode='localization')
                loc_loss += criteria['localization'](outputs, masks).item()
                
                # Track and calculate metrics immediately
                outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
                preds = (torch.sigmoid(outputs_resized) > 0.5).float()
                
                # Calculate metrics for this batch
                batch_metrics = calculate_localization_metrics(preds.cpu(), masks.cpu())
                
                # Accumulate metrics
                for k in loc_metrics_accum.keys():
                    loc_metrics_accum[k] += batch_metrics[k]
                
                loc_batch_count += 1
        
        # Calculate average metrics
        if loc_batch_count > 0:
            loc_metrics = {}
            for k in loc_metrics_accum.keys():
                loc_metrics[k] = loc_metrics_accum[k] / loc_batch_count
            
            loc_metrics['loss'] = loc_loss / len(dataloaders['localization']['val'])
            results['localization'] = loc_metrics
    
    return results

def print_metrics_table(epoch, epochs, train_metrics, val_metrics, task):
    """Print formatted metrics table for a specific task"""
    print("\n" + "="*80)
    print(f"EPOCH {epoch+1}/{epochs} {task.upper()} METRICS")
    print("="*80)
    print(f"{'Metric':<15} {'Training':<15} {'Validation':<15}")
    print("-"*80)
    
    metrics_to_print = train_metrics.keys()
    
    for metric in metrics_to_print:
        if metric != 'confusion_matrix':  # Skip confusion matrix for table
            if isinstance(train_metrics[metric], float) and isinstance(val_metrics[metric], float):
                if metric == 'accuracy':
                    print(f"{metric.capitalize():<15} {train_metrics[metric]:<14.2f}% {val_metrics[metric]:<14.2f}%")
                else:
                    print(f"{metric.capitalize():<15} {train_metrics[metric]:<15.4f} {val_metrics[metric]:<15.4f}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Train dual-task forgery detection model')
    parser.add_argument('--dataset', default='./dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--create_splits', action='store_true', help='Force recreate train/val/test splits')
    parser.add_argument('--use_albumentations', action='store_true', help='Use albumentations for augmentations')
    parser.add_argument('--no_save', action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--save_lite', action='store_true', help='Save lightweight checkpoints (no optimizer state)')
    parser.add_argument('--output_dir', type=str, default='./runs', help='Directory to save outputs')
    parser.add_argument('--compress_ckpt', action='store_true', help='Use compression for checkpoints')
    parser.add_argument('--prune_weights', action='store_true', help='Prune small weights to reduce checkpoint size')
    parser.add_argument('--pruning_threshold', type=float, default=1e-4, help='Threshold for weight pruning')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient dataloader and enable garbage collection')
    parser.add_argument('--best_only', action='store_true', default=True, help='Save only best checkpoints (no epoch-based checkpoints)')
    parser.add_argument('--save_final', action='store_true', help='Save final model at end of training')
    parser.add_argument('--combined_model', action='store_true', default=True, help='Save a combined model for both tasks')
    args = parser.parse_args()
    
    # Memory optimizations
    if args.memory_efficient:
        import gc
        # Force garbage collection
        gc.collect()
        # Set PyTorch memory allocator to release memory faster
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create output directory: {e}")
        # Try an alternative directory if home directory is full
        alt_output_dir = os.environ.get('TMPDIR', '/tmp')
        if alt_output_dir:
            print(f"Trying alternative output directory: {alt_output_dir}")
            output_dir = os.path.join(alt_output_dir, f"training_{timestamp}")
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create logger
    try:
        logger = Logger(output_dir)
    except OSError as e:
        print(f"Warning: Could not create logger: {e}")
        print("Training will continue without logging to CSV files")
        logger = None
    
    # Load the datasets
    print("\nCreating data loaders...")
    dataloaders, class_to_idx = get_dataloaders(
        args.dataset, 
        batch_size=args.batch_size, 
        create_splits=args.create_splits,
        use_albumentations=args.use_albumentations,
        num_workers=4
    )
    
    # Check available tasks
    has_classification = 'classification' in dataloaders
    has_localization = 'localization' in dataloaders
    
    if not has_classification and not has_localization:
        print("Error: No valid data found for either classification or localization")
        return
    
    # Save class mapping
    try:
        class_mapping_path = os.path.join(output_dir, 'class_mapping.txt')
        with open(class_mapping_path, 'w') as f:
            for class_name, idx in class_to_idx.items():
                f.write(f"{class_name}: {idx}\n")
    except OSError as e:
        print(f"Warning: Could not save class mapping: {e}")
    
    num_classes = len(class_to_idx)
    print(f"\nTraining with {num_classes} classes: {list(class_to_idx.keys())}")
    
    # Create model
    model = EfficientLaFNetDual(num_classes=num_classes).to(DEVICE)
    
    # Initialize loss functions
    criteria = {}
    if has_classification:
        criteria['classification'] = nn.CrossEntropyLoss()
    if has_localization:
        criteria['localization'] = HybridLoss()
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Initialize early stopping variables
    best_class_f1 = 0
    best_loc_f1 = 0
    class_epochs_without_improvement = 0
    loc_epochs_without_improvement = 0
    
    print("\nStarting training...")
    print(f"Training on: {', '.join(dataloaders.keys())}")
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, dataloaders, criteria, optimizer, DEVICE, num_classes, epoch
        )
        
        # Validate
        val_metrics = validate(model, dataloaders, criteria, DEVICE, num_classes)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        if has_classification:
            print_metrics_table(
                epoch, args.epochs,
                train_metrics['classification'],
                val_metrics['classification'],
                'classification'
            )
            
            # Log classification metrics
            if logger:
                logger.log_classification(
                    epoch,
                    train_metrics['classification']['loss'],
                    train_metrics['classification']['accuracy'],
                    train_metrics['classification']['precision'],
                    train_metrics['classification']['recall'],
                    train_metrics['classification']['f1'],
                    val_metrics['classification']['loss'],
                    val_metrics['classification']['accuracy'],
                    val_metrics['classification']['precision'],
                    val_metrics['classification']['recall'],
                    val_metrics['classification']['f1']
                )
        
        if has_localization:
            print_metrics_table(
                epoch, args.epochs,
                train_metrics['localization'],
                val_metrics['localization'],
                'localization'
            )
            
            # Log localization metrics
            if logger:
                logger.log_localization(
                    epoch,
                    train_metrics['localization']['loss'],
                    train_metrics['localization']['f1'],
                    train_metrics['localization']['accuracy'],
                    train_metrics['localization']['iou'],
                    train_metrics['localization']['dice'],
                    val_metrics['localization']['loss'],
                    val_metrics['localization']['f1'],
                    val_metrics['localization']['accuracy'],
                    val_metrics['localization']['iou'],
                    val_metrics['localization']['dice']
                )
        
        # Skip saving checkpoints if no_save is True
        if args.no_save:
            print("Skipping checkpoint saving as --no_save is set")
            continue
        
        # Force save a combined model on the first epoch
        if epoch == 0 and args.combined_model:
            try:
                combined_path = os.path.join(checkpoint_dir, 'combined_model.pth')
                print(f"Saving initial combined model to {combined_path}")
                save_checkpoint(
                    combined_path, 
                    epoch, 
                    model, 
                    optimizer if not args.save_lite else None,
                    scheduler if not args.save_lite else None, 
                    None,
                    class_to_idx, 
                    {
                        'classification': val_metrics.get('classification', {}),
                        'localization': val_metrics.get('localization', {}),
                        'best_class_f1': best_class_f1,
                        'best_loc_f1': best_loc_f1
                    }, 
                    args.compress_ckpt, 
                    args.prune_weights, 
                    args.pruning_threshold
                )
            except OSError as e:
                print(f"Warning: Failed to save initial combined model: {e}")
        
        # Check for improvement on classification task
        if has_classification:
            current_class_f1 = val_metrics['classification']['f1']
            if current_class_f1 > best_class_f1:
                print(f'Classification F1 improved from {best_class_f1:.4f} to {current_class_f1:.4f}')
                best_class_f1 = current_class_f1
                class_epochs_without_improvement = 0
                
                # Save best classification model
                try:
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_classification_model.pth')
                    save_checkpoint(checkpoint_path, epoch, model, optimizer if not args.save_lite else None, 
                                   scheduler if not args.save_lite else None, best_class_f1, 
                                   class_to_idx, val_metrics['classification'], args.compress_ckpt, args.prune_weights, args.pruning_threshold)
                    
                    # Also save a combined model if requested
                    if args.combined_model:
                        combined_path = os.path.join(checkpoint_dir, 'combined_model.pth')
                        save_checkpoint(
                            combined_path, 
                            epoch, 
                            model, 
                            optimizer if not args.save_lite else None,
                            scheduler if not args.save_lite else None, 
                            None,
                            class_to_idx, 
                            {
                                'classification': val_metrics.get('classification', {}),
                                'localization': val_metrics.get('localization', {}),
                                'best_class_f1': best_class_f1,
                                'best_loc_f1': best_loc_f1
                            }, 
                            args.compress_ckpt, 
                            args.prune_weights, 
                            args.pruning_threshold
                        )
                except OSError as e:
                    print(f"Warning: Failed to save classification checkpoint: {e}")
            else:
                class_epochs_without_improvement += 1
                print(f'No improvement in classification F1 for {class_epochs_without_improvement} epochs')
        
        # Check for improvement on localization task
        if has_localization:
            current_loc_f1 = val_metrics['localization']['f1']
            if current_loc_f1 > best_loc_f1:
                print(f'Localization F1 improved from {best_loc_f1:.4f} to {current_loc_f1:.4f}')
                best_loc_f1 = current_loc_f1
                loc_epochs_without_improvement = 0
                
                # Save best localization model
                try:
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_localization_model.pth')
                    save_checkpoint(checkpoint_path, epoch, model, optimizer if not args.save_lite else None,
                                   scheduler if not args.save_lite else None, best_loc_f1, 
                                   None, val_metrics['localization'], args.compress_ckpt, args.prune_weights, args.pruning_threshold)
                    
                    # Also save a combined model if requested
                    if args.combined_model:
                        combined_path = os.path.join(checkpoint_dir, 'combined_model.pth')
                        save_checkpoint(
                            combined_path, 
                            epoch, 
                            model, 
                            optimizer if not args.save_lite else None,
                            scheduler if not args.save_lite else None, 
                            None,
                            class_to_idx, 
                            {
                                'classification': val_metrics.get('classification', {}),
                                'localization': val_metrics.get('localization', {}),
                                'best_class_f1': best_class_f1,
                                'best_loc_f1': best_loc_f1
                            }, 
                            args.compress_ckpt, 
                            args.prune_weights, 
                            args.pruning_threshold
                        )
                except OSError as e:
                    print(f"Warning: Failed to save localization checkpoint: {e}")
            else:
                loc_epochs_without_improvement += 1
                print(f'No improvement in localization F1 for {loc_epochs_without_improvement} epochs')
        
        # Memory optimizations
        if args.memory_efficient and epoch % 5 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Early stopping check
        if has_classification and has_localization:
            if (class_epochs_without_improvement >= EARLY_STOPPING_PATIENCE and 
                loc_epochs_without_improvement >= EARLY_STOPPING_PATIENCE):
                print(f'\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement on both tasks')
                break
        elif has_classification and class_epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement on classification')
            break
        elif has_localization and loc_epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement on localization')
            break
    
    # Print final results
    print("\nTraining completed!")
    if has_classification:
        print(f"Best classification F1: {best_class_f1:.4f}")
    if has_localization:
        print(f"Best localization F1: {best_loc_f1:.4f}")
    
    # Save final model if requested
    if args.save_final and not args.no_save:
        try:
            final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
            print(f"Saving final model to {final_model_path}")
            save_checkpoint(
                final_model_path, 
                args.epochs - 1, 
                model, 
                optimizer if not args.save_lite else None,
                scheduler if not args.save_lite else None, 
                None,
                class_to_idx, 
                {
                    'classification': val_metrics.get('classification', {}),
                    'localization': val_metrics.get('localization', {})
                }, 
                args.compress_ckpt, 
                args.prune_weights, 
                args.pruning_threshold
            )
        except OSError as e:
            print(f"Warning: Failed to save final model: {e}")
    
    # Close logger
    if logger:
        logger.close()

def save_checkpoint(path, epoch, model, optimizer=None, scheduler=None, best_f1=None, 
                   class_to_idx=None, metrics=None, compress=False, prune_weights=False, 
                   threshold=1e-4):
    """Save checkpoint with optional compression to save disk space"""
    # Option 1: Quantize model weights to half precision (float16) to reduce size by ~50%
    if prune_weights:
        # Prune small weights to further reduce size
        model_state_dict = {}
        for k, v in model.state_dict().items():
            # For weight tensors (not biases, batch norm, etc.), prune small values
            if ('weight' in k and 'norm' not in k and len(v.shape) > 1 and v.shape[0] > 1):
                # Create a mask for values above threshold
                mask = torch.abs(v) > threshold
                # Create a sparse tensor representation
                indices = mask.nonzero(as_tuple=True)
                values = v[indices]
                shape = v.shape
                model_state_dict[k] = {
                    'indices': [idx.tolist() for idx in indices],
                    'values': values.half().tolist(),
                    'shape': list(shape)
                }
            else:
                # For other parameters, just convert to half
                model_state_dict[k] = v.half()
    else:
        # Simply convert to half precision without pruning
        model_state_dict = {k: v.half() for k, v in model.state_dict().items()}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'pruned': prune_weights,
    }
    
    # Only add these if not None to save space
    if optimizer is not None:
        # Option 2: We don't need optimizer state for inference, so we can simplify it
        # Just store the hyperparameters, not the full state
        opt_state = {'param_groups': optimizer.state_dict()['param_groups']}
        checkpoint['optimizer_state_dict'] = opt_state
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if best_f1 is not None:
        checkpoint['best_f1'] = best_f1
    if class_to_idx is not None:
        checkpoint['class_to_idx'] = class_to_idx
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Use pickle protocol 4 which is more efficient
    if compress:
        import gzip
        with gzip.open(path + '.gz', 'wb', compresslevel=9) as f:
            torch.save(checkpoint, f, pickle_protocol=4)
        saved_path = path + '.gz'
    else:
        torch.save(checkpoint, path, pickle_protocol=4)
        saved_path = path
    
    # Check and print file size
    try:
        size_mb = os.path.getsize(saved_path) / (1024 * 1024)
        print(f"Saved checkpoint to {saved_path} (Size: {size_mb:.2f} MB)")
    except OSError:
        print(f"Saved checkpoint to {saved_path}")
    
    return saved_path

def load_checkpoint(path, model, device):
    """Load a checkpoint that was saved with half precision weights"""
    if path.endswith('.gz'):
        import gzip
        with gzip.open(path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)
    else:
        checkpoint = torch.load(path, map_location=device)
    
    # Check if weights were pruned
    if checkpoint.get('pruned', False):
        # Convert pruned sparse representation back to dense tensors
        model_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if isinstance(v, dict) and 'indices' in v:
                # This is a pruned tensor, reconstruct it
                tensor = torch.zeros(v['shape'], dtype=torch.float16, device=device)
                indices = tuple(torch.tensor(idx) for idx in v['indices'])
                values = torch.tensor(v['values'], dtype=torch.float16, device=device)
                tensor[indices] = values
                model_state_dict[k] = tensor.float()  # Convert back to float32
            else:
                model_state_dict[k] = v.float() if hasattr(v, 'float') else v
    else:
        # Convert half precision weights back to full precision when loading
        model_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            model_state_dict[k] = v.float() if hasattr(v, 'float') else v
    
    model.load_state_dict(model_state_dict)
    
    return checkpoint

def manage_checkpoints(checkpoint_dir, keep_n=3):
    """Keep only the N best checkpoints to save disk space"""
    if keep_n <= 0:  # Keep all checkpoints
        return
    
    # Classification checkpoints
    class_path = os.path.join(checkpoint_dir, 'best_classification_model.pth')
    if os.path.exists(class_path):
        # Keep this one since it's the best
        pass
        
    # Localization checkpoints
    loc_path = os.path.join(checkpoint_dir, 'best_localization_model.pth')
    if os.path.exists(loc_path):
        # Keep this one since it's the best
        pass
    
    # Find all epoch checkpoints
    epoch_checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('model_epoch_') and (filename.endswith('.pth') or filename.endswith('.pth.gz')):
            full_path = os.path.join(checkpoint_dir, filename)
            try:
                epoch = int(filename.replace('model_epoch_', '').replace('.pth', '').replace('.gz', ''))
                epoch_checkpoints.append((epoch, full_path))
            except ValueError:
                continue
    
    # Sort by epoch (higher is more recent)
    epoch_checkpoints.sort(reverse=True)
    
    # Keep only keep_n most recent checkpoints
    if len(epoch_checkpoints) > keep_n:
        for _, path in epoch_checkpoints[keep_n:]:
            try:
                os.remove(path)
                print(f"Removed old checkpoint: {path}")
            except OSError as e:
                print(f"Warning: Could not remove checkpoint {path}: {e}")

if __name__ == '__main__':
    main() 