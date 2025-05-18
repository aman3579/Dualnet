import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Transpose, 
    RGBShift, RandomResizedCrop, Normalize
)
from albumentations.pytorch import ToTensorV2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
IMG_SIZE = 512  # Using 512 to match localization required size
BATCH_SIZE = 16
FORGERY_TYPES = ['real', 'splicing', 'copy_move', 'aigenerated']

def find_corresponding_mask(image_path):
    """
    Find the corresponding mask file for an image with various naming conventions
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Path to the mask file if found, None otherwise
    """
    # Get base name and directory
    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    
    # Convert images directory to masks directory
    masks_dir = os.path.join(os.path.dirname(image_dir), 'masks')
    
    if not os.path.exists(masks_dir):
        return None
    
    # Try different possible mask patterns
    possible_mask_patterns = [
        f"{base_name}_gt.png",
        f"{base_name}_gt.jpg",
        f"{base_name}_gt.tif",
        f"{base_name}_mask.png",
        f"{base_name}_mask.jpg",
        f"{base_name}_mask.tif",
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"{base_name}.tif"
    ]
    
    for mask_pattern in possible_mask_patterns:
        mask_path = os.path.join(masks_dir, mask_pattern)
        if os.path.exists(mask_path):
            return mask_path
    
    return None

def is_valid_image(image_path):
    """Check if an image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            # Verify the image can be loaded by accessing some attribute
            img.size
            return True
    except Exception as e:
        print(f"Corrupted image detected at {image_path}: {str(e)}")
        return False

def create_dataset_splits(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create CSV files for train/val/test splits for both classification and localization tasks
    
    Args:
        base_dir: Root directory containing multiple datasets
        train_ratio, val_ratio, test_ratio: Split ratios
    """
    base_path = os.path.join(base_dir)
    
    # Create output directory for CSV files
    splits_dir = os.path.join(base_path, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # Data for classification (just images)
    classification_data = []
    # Data for localization (image-mask pairs)
    localization_data = []
    
    # Process each class
    class_to_idx = {class_name: idx for idx, class_name in enumerate(FORGERY_TYPES)}
    
    print("Processing dataset for classification and localization...")
    
    for class_name in FORGERY_TYPES:
        class_dir = os.path.join(base_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_name} not found")
            continue
        
        print(f"\nProcessing class: {class_name}")
        
        # Get all subdirectories (dataset sources) within this class
        subdirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(class_dir, subdir)
            
            # Check if this is a standard subdir with images folder
            images_dir = os.path.join(subdir_path, "images")
            masks_dir = os.path.join(subdir_path, "masks")
            
            # Case 1: Subdir has both images and masks folders (for localization)
            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                print(f"  - Processing {class_name}/{subdir} for localization")
                
                valid_pairs = 0
                for filename in os.listdir(images_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                        image_path = os.path.join(images_dir, filename)
                        
                        # Find corresponding mask
                        mask_path = find_corresponding_mask(image_path)
                        
                        if mask_path and is_valid_image(image_path) and is_valid_image(mask_path):
                            # Add to localization data
                            localization_data.append({
                                'image_path': image_path,
                                'mask_path': mask_path,
                                'class_name': class_name,
                                'class_idx': class_to_idx[class_name],
                                'dataset': subdir
                            })
                            
                            # Also add to classification data
                            classification_data.append({
                                'image_path': image_path,
                                'class_name': class_name,
                                'class_idx': class_to_idx[class_name],
                                'dataset': subdir
                            })
                            
                            valid_pairs += 1
                
                print(f"    Found {valid_pairs} valid image-mask pairs")
                
            # Case 2: Subdir has only images (no masks) or direct images (for classification only)
            else:
                # Try two cases: either direct images or nested in 'images' folder
                img_dir = images_dir if os.path.exists(images_dir) else subdir_path
                print(f"  - Processing {class_name}/{subdir} for classification only")
                
                valid_images = 0
                for root, _, files in os.walk(img_dir):
                    for filename in files:
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                            image_path = os.path.join(root, filename)
                            
                            if is_valid_image(image_path):
                                # Add to classification data only
                                classification_data.append({
                                    'image_path': image_path,
                                    'class_name': class_name,
                                    'class_idx': class_to_idx[class_name],
                                    'dataset': subdir
                                })
                                valid_images += 1
                
                print(f"    Found {valid_images} valid images")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"  Classification data: {len(classification_data)} images")
    print(f"  Localization data: {len(localization_data)} image-mask pairs")
    
    # Split classification data
    df_classification = pd.DataFrame(classification_data)
    if len(df_classification) > 0:
        try:
            # Stratify by both class and dataset if possible
            df_classification['stratify_col'] = df_classification['class_name'] + '_' + df_classification['dataset']
            train_df, temp_df = train_test_split(
                df_classification, 
                test_size=(val_ratio + test_ratio), 
                random_state=42, 
                stratify=df_classification['stratify_col']
            )
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=(1-val_ratio_adjusted), 
                random_state=42, 
                stratify=temp_df['stratify_col']
            )
            
            # Remove helper column
            train_df = train_df.drop('stratify_col', axis=1)
            val_df = val_df.drop('stratify_col', axis=1)
            test_df = test_df.drop('stratify_col', axis=1)
            
        except ValueError as e:
            # If stratification fails, fall back to stratifying just by class
            print(f"Warning: Stratified split by dataset failed ({str(e)}). Falling back to class stratification.")
            train_df, temp_df = train_test_split(
                df_classification, 
                test_size=(val_ratio + test_ratio), 
                random_state=42, 
                stratify=df_classification['class_name']
            )
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=(1-val_ratio_adjusted), 
                random_state=42, 
                stratify=temp_df['class_name']
            )
        
        # Save classification splits
        train_df.to_csv(os.path.join(splits_dir, 'classification_train.csv'), index=False)
        val_df.to_csv(os.path.join(splits_dir, 'classification_val.csv'), index=False)
        test_df.to_csv(os.path.join(splits_dir, 'classification_test.csv'), index=False)
        print(f"\nClassification splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Split localization data
    df_localization = pd.DataFrame(localization_data)
    if len(df_localization) > 0:
        try:
            # Stratify by both class and dataset if possible
            df_localization['stratify_col'] = df_localization['class_name'] + '_' + df_localization['dataset']
            train_loc_df, temp_loc_df = train_test_split(
                df_localization, 
                test_size=(val_ratio + test_ratio), 
                random_state=42, 
                stratify=df_localization['stratify_col']
            )
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_loc_df, test_loc_df = train_test_split(
                temp_loc_df, 
                test_size=(1-val_ratio_adjusted), 
                random_state=42, 
                stratify=temp_loc_df['stratify_col']
            )
            
            # Remove helper column
            train_loc_df = train_loc_df.drop('stratify_col', axis=1)
            val_loc_df = val_loc_df.drop('stratify_col', axis=1)
            test_loc_df = test_loc_df.drop('stratify_col', axis=1)
            
        except ValueError as e:
            # If stratification fails, fall back to stratifying just by class
            print(f"Warning: Stratified split by dataset failed ({str(e)}). Falling back to class stratification.")
            train_loc_df, temp_loc_df = train_test_split(
                df_localization, 
                test_size=(val_ratio + test_ratio), 
                random_state=42, 
                stratify=df_localization['class_name']
            )
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_loc_df, test_loc_df = train_test_split(
                temp_loc_df, 
                test_size=(1-val_ratio_adjusted), 
                random_state=42, 
                stratify=temp_loc_df['class_name']
            )
        
        # Save localization splits
        train_loc_df.to_csv(os.path.join(splits_dir, 'localization_train.csv'), index=False)
        val_loc_df.to_csv(os.path.join(splits_dir, 'localization_val.csv'), index=False)
        test_loc_df.to_csv(os.path.join(splits_dir, 'localization_test.csv'), index=False)
        print(f"Localization splits: {len(train_loc_df)} train, {len(val_loc_df)} val, {len(test_loc_df)} test")
    
    return class_to_idx

class ClassificationDataset(Dataset):
    """Dataset for the classification task"""
    def __init__(self, csv_file, img_size=IMG_SIZE, transform=None, use_albumentations=True):
        self.data_frame = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform
        self.use_albumentations = use_albumentations
        
        # Default albumentations transforms if none provided
        if self.use_albumentations and self.transform is None:
            self.transform = Compose([
                RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                HorizontalFlip(),
                VerticalFlip(),
                RandomRotate90(),
                RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                Transpose(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # Default torchvision transforms if not using albumentations
        if not self.use_albumentations and self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data_frame)
    
    def _add_compression(self, img):
        """Add JPEG compression artifacts"""
        quality = np.random.randint(30, 90)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return Image.open(io.BytesIO(enc)).convert('RGB')
    
    def __getitem__(self, idx):
        try:
            img_path = self.data_frame.iloc[idx]['image_path']
            class_idx = self.data_frame.iloc[idx]['class_idx']
            
            # Check if file exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.use_albumentations:
                # Convert PIL Image to numpy array for albumentations
                img_np = np.array(img)
                # Apply albumentations transforms
                transformed = self.transform(image=img_np)
                img_tensor = transformed['image']  # This is now a tensor from ToTensorV2
            else:
                # Use traditional torchvision transforms
                img_tensor = self.transform(img)
            
            return img_tensor, class_idx
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a valid but empty sample
            img_tensor = torch.zeros(3, self.img_size, self.img_size)
            return img_tensor, 0  # Default to 'real' class (0)

class LocalizationDataset(Dataset):
    """Dataset for the localization task"""
    def __init__(self, csv_file, img_size=IMG_SIZE, transform=None, use_albumentations=True):
        self.data_frame = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform
        self.use_albumentations = use_albumentations
        
        # Default transforms for training (with augmentations)
        if self.use_albumentations and self.transform is None:
            self.transform = Compose([
                RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                HorizontalFlip(),
                VerticalFlip(),
                RandomRotate90(),
                RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # Default torchvision transforms if not using albumentations
        if not self.use_albumentations and self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        # Mask transform for torchvision
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        try:
            img_path = self.data_frame.iloc[idx]['image_path']
            mask_path = self.data_frame.iloc[idx]['mask_path']
            
            # Check if files exist
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                raise FileNotFoundError(f"Image or mask file not found: {img_path} or {mask_path}")
            
            # Load image and mask
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Apply transforms
            if self.use_albumentations:
                # Convert PIL Images to numpy arrays for albumentations
                img_np = np.array(img)
                mask_np = np.array(mask)
                
                # Apply same spatial augmentations to both image and mask
                transformed = self.transform(image=img_np, mask=mask_np)
                img_tensor = transformed['image']  # This is now a tensor from ToTensorV2
                
                # Convert mask to binary and then to tensor
                mask_tensor = torch.from_numpy((transformed['mask'] > 127).astype(np.float32)).unsqueeze(0)
            else:
                # Use traditional torchvision transforms
                img_tensor = self.transform(img)
                mask_tensor = (self.mask_transform(mask) > 0.5).float()
            
            return img_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a valid but empty sample
            img_tensor = torch.zeros(3, self.img_size, self.img_size)
            mask_tensor = torch.zeros(1, self.img_size, self.img_size)
            return img_tensor, mask_tensor

def create_val_transforms(img_size=IMG_SIZE, use_albumentations=True):
    """Create validation transforms with less augmentation"""
    if use_albumentations:
        return Compose([
            RandomResizedCrop(height=img_size, width=img_size, scale=(0.9, 1.0)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_dataloaders(dataset_dir, batch_size=BATCH_SIZE, num_workers=4, 
                    create_splits=False, use_albumentations=True, pin_memory=True):
    """
    Create dataloaders for both classification and localization tasks
    
    Args:
        dataset_dir: Path to the dataset directory
        batch_size: Batch size for training
        num_workers: Number of workers for dataloader
        create_splits: Whether to create new train/val/test splits
        use_albumentations: Whether to use albumentations for transforms
        
    Returns:
        Dictionary containing dataloaders for both tasks
    """
    splits_dir = os.path.join(dataset_dir, 'splits')
    
    # Create CSV files if they don't exist or if explicitly requested
    if create_splits or not os.path.exists(os.path.join(splits_dir, 'classification_train.csv')):
        print("Creating dataset splits...")
        class_to_idx = create_dataset_splits(dataset_dir)
    else:
        print("Using existing dataset splits. Use create_splits=True to regenerate.")
        # Read the class mapping from existing files
        class_df = pd.read_csv(os.path.join(splits_dir, 'classification_train.csv'))
        class_to_idx = dict(zip(class_df['class_name'].unique(), class_df['class_idx'].unique()))
    
    # Create transforms for validation/test
    val_transforms = create_val_transforms(use_albumentations=use_albumentations)
    
    # Classification datasets
    classification_train = ClassificationDataset(
        os.path.join(splits_dir, 'classification_train.csv'), 
        use_albumentations=use_albumentations
    )
    classification_val = ClassificationDataset(
        os.path.join(splits_dir, 'classification_val.csv'), 
        transform=val_transforms,
        use_albumentations=use_albumentations
    )
    classification_test = ClassificationDataset(
        os.path.join(splits_dir, 'classification_test.csv'), 
        transform=val_transforms,
        use_albumentations=use_albumentations
    )
    
    # Check if localization files exist
    has_localization = os.path.exists(os.path.join(splits_dir, 'localization_train.csv'))
    
    dataloaders = {
        'classification': {
            'train': DataLoader(
                classification_train, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=pin_memory
            ),
            'val': DataLoader(
                classification_val, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=pin_memory
            ),
            'test': DataLoader(
                classification_test, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=pin_memory
            )
        }
    }
    
    if has_localization:
        # Localization datasets
        localization_train = LocalizationDataset(
            os.path.join(splits_dir, 'localization_train.csv'), 
            use_albumentations=use_albumentations
        )
        localization_val = LocalizationDataset(
            os.path.join(splits_dir, 'localization_val.csv'), 
            transform=val_transforms,
            use_albumentations=use_albumentations
        )
        localization_test = LocalizationDataset(
            os.path.join(splits_dir, 'localization_test.csv'), 
            transform=val_transforms,
            use_albumentations=use_albumentations
        )
        
        dataloaders['localization'] = {
            'train': DataLoader(
                localization_train, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=pin_memory
            ),
            'val': DataLoader(
                localization_val, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=pin_memory
            ),
            'test': DataLoader(
                localization_test, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=pin_memory
            )
        }
    
    # Print dataset sizes
    print(f"\nClassification dataset sizes:")
    print(f"  Train: {len(classification_train)}")
    print(f"  Val: {len(classification_val)}")
    print(f"  Test: {len(classification_test)}")
    
    if has_localization:
        print(f"\nLocalization dataset sizes:")
        print(f"  Train: {len(localization_train)}")
        print(f"  Val: {len(localization_val)}")
        print(f"  Test: {len(localization_test)}")
    
    return dataloaders, class_to_idx

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for combined classification and localization')
    parser.add_argument('--dataset', default='./dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--create_splits', action='store_true', help='Force recreate train/val/test splits')
    args = parser.parse_args()
    
    # Create dataset splits
    dataloaders, class_mapping = get_dataloaders(args.dataset, create_splits=args.create_splits)
    
    print("\nClass mapping:", class_mapping)
    print("\nDataset preparation completed successfully.") 