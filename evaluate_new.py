import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, 
    classification_report, accuracy_score, f1_score
)
import os
import argparse
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from model_new import EfficientLaFNetDual
from preprocessing_new import get_dataloaders

class ForgeryEvaluator:
    def __init__(self, model_path, num_classes=4, img_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EfficientLaFNetDual(num_classes=num_classes).to(self.device)
        
        # Load model with checkpoint format
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']}")
            
            # Print metrics if available
            if 'classification_metrics' in checkpoint and checkpoint['classification_metrics']:
                print("Classification metrics:")
                for k, v in checkpoint['classification_metrics'].items():
                    if k != 'confusion_matrix':
                        print(f"  {k}: {v:.4f}")
                        
            if 'localization_metrics' in checkpoint and checkpoint['localization_metrics']:
                print("Localization metrics:")
                for k, v in checkpoint['localization_metrics'].items():
                    print(f"  {k}: {v:.4f}")
        else:
            # Fallback for old format
            self.model.load_state_dict(checkpoint)
            print("Loaded model from old format checkpoint")
            
        self.model.eval()
        self.img_size = img_size
        
        # Read class mapping if available
        self.class_to_idx = checkpoint.get('class_to_idx', None)
        if self.class_to_idx:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def evaluate_classification(self, test_loader):
        """Evaluate classification performance on test set"""
        all_preds = []
        all_probs = []
        all_targets = []
        test_loss = 0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Evaluating classification'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs, mode='classification')
                
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Get predicted class and probabilities
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # If probabilities are available, calculate ROC AUC
        auroc = None
        if len(all_probs) > 0 and all_probs.shape[1] > 1:
            # One-hot encode targets for ROC AUC
            try:
                auroc = roc_auc_score(
                    np.eye(all_probs.shape[1])[all_targets],
                    all_probs,
                    multi_class='ovr',
                    average='macro'
                )
            except ValueError:
                print("Warning: Could not calculate AUROC")
                auroc = None
        
        class_names = [self.idx_to_class[i] if self.idx_to_class else f"Class {i}" 
                      for i in range(len(np.unique(all_targets)))]
        
        class_report = classification_report(all_targets, all_preds, 
                                             target_names=class_names, 
                                             digits=4)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'f1': f1,
            'auroc': auroc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': all_preds,
            'probabilities': all_probs,
            'targets': all_targets
        }
    
    def evaluate_localization(self, test_loader):
        """Evaluate localization performance on test set"""
        all_preds = []
        all_masks = []
        total_loss = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        
        from train_new import HybridLoss
        criterion = HybridLoss()
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc='Evaluating localization'):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images, mode='localization')
                
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Resize predictions to match mask size
                outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear')
                preds = (torch.sigmoid(outputs_resized) > 0.5).float()
                
                # Accumulate confusion matrix elements
                tp += (preds * masks).sum()
                fp += (preds * (1 - masks)).sum()
                fn += ((1 - preds) * masks).sum()
                tn += ((1 - preds) * (1 - masks)).sum()
                
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        # Calculate probabilistic metrics if possible
        try:
            preds_concat = torch.cat([torch.sigmoid(p) for p in all_preds]).view(-1).numpy()
            masks_concat = torch.cat(all_masks).view(-1).numpy()
            auroc = roc_auc_score(masks_concat, preds_concat)
            ap = average_precision_score(masks_concat, preds_concat)
        except Exception as e:
            print(f"Warning: Could not calculate probabilistic metrics: {str(e)}")
            auroc = None
            ap = None
        
        return {
            'f1': f1.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'iou': iou.item(),
            'dice': dice.item(),
            'auroc': auroc,
            'ap': ap,
            'loss': total_loss / len(test_loader)
        }
        
    def visualize_results(self, image_path, mask_path=None, output_path=None):
        """Visualize model predictions on a single image"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            # Get classification results
            class_output = self.model(img_tensor, mode='classification')
            class_probs = F.softmax(class_output, dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            
            # Get localization results if applicable
            if mask_path:
                loc_output = self.model(img_tensor, mode='localization')
                loc_output_resized = F.interpolate(loc_output, size=(self.img_size, self.img_size), mode='bilinear')
                mask_pred = torch.sigmoid(loc_output_resized)[0, 0].cpu().numpy()
                
                # Load ground truth mask if available
                mask_gt = None
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask_transform = transforms.Compose([
                        transforms.Resize((self.img_size, self.img_size), 
                                        interpolation=transforms.InterpolationMode.NEAREST),
                        transforms.ToTensor()
                    ])
                    mask_gt = mask_transform(mask)[0].cpu().numpy()
        
        # Generate visualizations
        if mask_path and mask_gt is not None:
            # For localization
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax[0].imshow(img)
            class_name = self.idx_to_class[predicted_class] if self.idx_to_class else f"Class {predicted_class}"
            ax[0].set_title(f'Image\nPredicted: {class_name}\nProb: {class_probs[0, predicted_class]:.2f}')
            ax[0].axis('off')
            
            # Ground truth mask
            ax[1].imshow(mask_gt, cmap='gray')
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')
            
            # Predicted mask
            ax[2].imshow(mask_pred, cmap='jet')
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')
        else:
            # For classification only
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax[0].imshow(img)
            class_name = self.idx_to_class[predicted_class] if self.idx_to_class else f"Class {predicted_class}"
            ax[0].set_title(f'Image\nPredicted: {class_name}')
            ax[0].axis('off')
            
            # Class probabilities
            all_classes = [self.idx_to_class[i] if self.idx_to_class else f"Class {i}" 
                          for i in range(len(class_probs[0]))]
            
            ax[1].bar(all_classes, class_probs[0].cpu().numpy())
            ax[1].set_title('Class Probabilities')
            ax[1].set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
        
    def visualize_classification_results(self, test_loader, output_dir, num_samples=5):
        """Visualize classification results on test set"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_images = []
        all_targets = []
        all_preds = []
        all_probs = []
        
        # Get predictions for test set
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                outputs = self.model(images, mode='classification')
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_images.append(images.cpu())
                all_targets.append(targets)
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                
                if len(all_images) * images.size(0) >= num_samples:
                    break
        
        # Generate visualization samples
        all_images = torch.cat(all_images)
        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        
        # Get correct and incorrect predictions
        correct_mask = all_targets == all_preds
        correct_indices = torch.nonzero(correct_mask).squeeze()
        incorrect_indices = torch.nonzero(~correct_mask).squeeze()
        
        # Get half correct, half incorrect samples
        half_samples = num_samples // 2
        correct_samples = min(half_samples, len(correct_indices))
        incorrect_samples = min(num_samples - correct_samples, len(incorrect_indices))
        
        if len(correct_indices) > 0:
            correct_indices = correct_indices[torch.randperm(len(correct_indices))[:correct_samples]]
        
        if len(incorrect_indices) > 0:
            incorrect_indices = incorrect_indices[torch.randperm(len(incorrect_indices))[:incorrect_samples]]
        
        # Combine indices
        indices = torch.cat([correct_indices, incorrect_indices])[:num_samples]
        
        # Generate visualizations
        for i, idx in enumerate(indices):
            img = all_images[idx]
            target = all_targets[idx].item()
            pred = all_preds[idx].item()
            prob = all_probs[idx, pred].item()
            
            # Reverse normalization to display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Generate class names
            target_name = self.idx_to_class[target] if self.idx_to_class else f"Class {target}"
            pred_name = self.idx_to_class[pred] if self.idx_to_class else f"Class {pred}"
            
            # Plot
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax[0].imshow(img_display)
            ax[0].set_title(f'Ground Truth: {target_name}\nPredicted: {pred_name}\nProb: {prob:.2f}')
            ax[0].axis('off')
            
            # Class probabilities
            all_classes = [self.idx_to_class[i] if self.idx_to_class else f"Class {i}" 
                          for i in range(len(all_probs[idx]))]
            
            ax[1].bar(all_classes, all_probs[idx].numpy())
            ax[1].set_title('Class Probabilities')
            ax[1].set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save visualization
            correct_str = "correct" if target == pred else "incorrect"
            plt.savefig(os.path.join(output_dir, f'classification_{i}_{correct_str}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def visualize_localization_results(self, test_loader, output_dir, num_samples=5):
        """Visualize localization results on test set"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sample data from the test loader
        images = []
        masks = []
        
        with torch.no_grad():
            for img_batch, mask_batch in test_loader:
                for i in range(len(img_batch)):
                    images.append(img_batch[i])
                    masks.append(mask_batch[i])
                    if len(images) >= num_samples:
                        break
                if len(images) >= num_samples:
                    break
        
        # Generate visualizations
        for i, (img, mask) in enumerate(zip(images[:num_samples], masks[:num_samples])):
            img = img.to(self.device).unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            # Get model prediction
            loc_output = self.model(img, mode='localization')
            loc_output_resized = F.interpolate(loc_output, size=mask.shape[2:], mode='bilinear')
            
            # Get binary prediction and probability map
            mask_pred_prob = torch.sigmoid(loc_output_resized)[0, 0].cpu().numpy()
            mask_pred_binary = (mask_pred_prob > 0.5).astype(np.float32)
            mask_gt = mask[0, 0].cpu().numpy()
            
            # Also get classification if available
            class_name = None
            try:
                class_output = self.model(img, mode='classification')
                class_probs = F.softmax(class_output, dim=1)
                predicted_class = torch.argmax(class_probs, dim=1).item()
                class_name = self.idx_to_class[predicted_class] if self.idx_to_class else f"Class {predicted_class}"
            except Exception:
                pass
            
            # Reverse normalization for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img[0].cpu() * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Calculate metrics for this sample
            tp = float(np.sum((mask_pred_binary == 1) & (mask_gt == 1)))
            fp = float(np.sum((mask_pred_binary == 1) & (mask_gt == 0)))
            fn = float(np.sum((mask_pred_binary == 0) & (mask_gt == 1)))
            tn = float(np.sum((mask_pred_binary == 0) & (mask_gt == 0)))
            
            iou = tp / (tp + fp + fn + 1e-8)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            
            # Generate visualization
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            ax[0].imshow(img_display)
            title = f'Input Image'
            if class_name:
                title += f'\nClass: {class_name}'
            ax[0].set_title(title)
            ax[0].axis('off')
            
            # Ground truth mask
            ax[1].imshow(mask_gt, cmap='gray')
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')
            
            # Predicted probability map
            ax[2].imshow(mask_pred_prob, cmap='jet')
            ax[2].set_title('Prediction (Probability)')
            ax[2].axis('off')
            
            # Binary mask prediction
            ax[3].imshow(mask_pred_binary, cmap='gray')
            ax[3].set_title(f'Binary Prediction\nIoU: {iou:.4f}, Dice: {dice:.4f}')
            ax[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'localization_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

def evaluate_model(model_path, dataset_dir, output_dir=None, create_splits=False, use_albumentations=True):
    """
    Evaluate the model on test data for both classification and localization tasks
    
    Args:
        model_path: Path to the trained model
        dataset_dir: Path to the dataset directory
        output_dir: Directory to save visualization results (optional)
        create_splits: Whether to create new train/val/test splits
        use_albumentations: Whether to use albumentations for transforms
    """
    # Load model checkpoint to get class mapping
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        num_classes = len(class_to_idx)
    else:
        # Try to infer from model architecture
        print("Warning: Could not find class mapping in checkpoint, assuming 4 classes")
        num_classes = 4
    
    # Create evaluator
    evaluator = ForgeryEvaluator(model_path, num_classes=num_classes)
    
    # Load dataloaders
    print("\nLoading dataloaders...")
    dataloaders, _ = get_dataloaders(
        dataset_dir, 
        create_splits=create_splits,
        use_albumentations=use_albumentations,
        num_workers=4
    )
    
    results = {}
    
    # Check if we have classification data
    if 'classification' in dataloaders:
        print("\nEvaluating classification performance...")
        classification_metrics = evaluator.evaluate_classification(dataloaders['classification']['test'])
        results['classification'] = classification_metrics
        
        # Print classification results
        print("\nClassification Results:")
        print(f"Accuracy: {classification_metrics['accuracy']:.2f}%")
        print(f"F1 Score: {classification_metrics['f1']:.4f}")
        if classification_metrics['auroc']:
            print(f"AUROC: {classification_metrics['auroc']:.4f}")
        print("\nClassification Report:")
        print(classification_metrics['classification_report'])
        
        # Create confusion matrix visualization
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(10, 8))
            cm = classification_metrics['confusion_matrix']
            
            # Get class names if available
            if hasattr(evaluator, 'idx_to_class') and evaluator.idx_to_class:
                class_names = [evaluator.idx_to_class[i] for i in range(len(cm))]
            else:
                class_names = [f"Class {i}" for i in range(len(cm))]
            
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            # Add labels and ticks
            plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
            plt.yticks(np.arange(len(class_names)), class_names)
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(output_dir, 'classification_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            
            # Save classification report
            with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
                f.write(f"Accuracy: {classification_metrics['accuracy']:.2f}%\n")
                f.write(f"F1 Score: {classification_metrics['f1']:.4f}\n")
                if classification_metrics['auroc']:
                    f.write(f"AUROC: {classification_metrics['auroc']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(classification_metrics['classification_report'])
            
            # Generate visualization samples
            print("\nGenerating classification visualizations...")
            visualizations_dir = os.path.join(output_dir, 'classification_examples')
            os.makedirs(visualizations_dir, exist_ok=True)
            evaluator.visualize_classification_results(
                dataloaders['classification']['test'], 
                visualizations_dir
            )
    
    # Check if we have localization data
    if 'localization' in dataloaders:
        print("\nEvaluating localization performance...")
        localization_metrics = evaluator.evaluate_localization(dataloaders['localization']['test'])
        results['localization'] = localization_metrics
        
        # Print localization results
        print("\nLocalization Results:")
        print(f"F1 Score: {localization_metrics['f1']:.4f}")
        print(f"Accuracy: {localization_metrics['accuracy']:.4f}")
        print(f"IoU: {localization_metrics['iou']:.4f}")
        print(f"Dice: {localization_metrics['dice']:.4f}")
        if localization_metrics['auroc']:
            print(f"AUROC: {localization_metrics['auroc']:.4f}")
        if localization_metrics['ap']:
            print(f"Average Precision: {localization_metrics['ap']:.4f}")
        
        # Save localization results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'localization_metrics.txt'), 'w') as f:
                f.write(f"F1 Score: {localization_metrics['f1']:.4f}\n")
                f.write(f"Accuracy: {localization_metrics['accuracy']:.4f}\n")
                f.write(f"IoU: {localization_metrics['iou']:.4f}\n")
                f.write(f"Dice: {localization_metrics['dice']:.4f}\n")
                if localization_metrics['auroc']:
                    f.write(f"AUROC: {localization_metrics['auroc']:.4f}\n")
                if localization_metrics['ap']:
                    f.write(f"Average Precision: {localization_metrics['ap']:.4f}\n")
            
            # Generate visualization samples
            print("\nGenerating localization visualizations...")
            visualizations_dir = os.path.join(output_dir, 'localization_examples')
            os.makedirs(visualizations_dir, exist_ok=True)
            evaluator.visualize_localization_results(
                dataloaders['localization']['test'], 
                visualizations_dir
            )
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate dual-task forgery detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--create_splits', action='store_true', help='Force recreate train/val/test splits')
    parser.add_argument('--use_albumentations', action='store_true', help='Use albumentations for transforms')
    args = parser.parse_args()
    
    evaluate_model(
        args.model, 
        args.dataset, 
        args.output,
        create_splits=args.create_splits,
        use_albumentations=args.use_albumentations
    ) 