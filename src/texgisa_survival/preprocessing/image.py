# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Image data preprocessing for survival analysis.

This module processes medical images and extracts deep learning features
suitable for survival analysis models.
"""

import os
import io
import zipfile
import tempfile
from typing import Optional, List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Image processing imports with graceful fallback
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    warnings.warn("PIL not available. Install with: pip install pillow")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    import torchvision
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.transforms import v2 as T
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False
    if _HAS_TORCH:
        warnings.warn("torchvision not available. Install with: pip install torchvision")


class ImageDataset(Dataset):
    """PyTorch dataset for image loading."""
    
    def __init__(self, 
                 image_paths: List[Path],
                 labels: Optional[pd.DataFrame] = None,
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels if available
        if self.labels is not None:
            label = self.labels.iloc[idx].to_dict()
        else:
            label = {'file': str(img_path.name)}
        
        return image, label, str(img_path)


class ImageProcessor:
    """
    Process medical images for survival analysis.
    
    This processor handles:
    - Various image formats (JPEG, PNG, DICOM with optional support)
    - Deep feature extraction using pretrained models
    - Batch processing for efficiency
    - Custom feature extractors
    
    Parameters
    ----------
    model_name : str
        Pretrained model to use: 'resnet50', 'resnet101', 'vit', 'densenet'
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps')
    batch_size : int
        Batch size for processing
    num_workers : int
        Number of data loading workers
    image_size : tuple
        Target size for images (height, width)
    normalize : bool
        Apply ImageNet normalization
    verbose : bool
        Print processing information
        
    Examples
    --------
    >>> processor = ImageProcessor(model_name='resnet50')
    >>> 
    >>> # Process from directory with labels
    >>> features = processor.process_directory('images/', 'labels.csv')
    >>> 
    >>> # Process from ZIP file
    >>> features = processor.process_zip('images.zip', 'labels.csv')
    >>> 
    >>> # Process single image
    >>> features = processor.process_image('scan.jpg')
    """
    
    SUPPORTED_MODELS = ['resnet50', 'resnet101', 'densenet121', 'efficientnet_b0']
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self,
                 model_name: str = 'resnet50',
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 verbose: bool = True):
        
        # Check dependencies
        if not _HAS_PIL:
            raise ImportError("PIL is required for image processing. Install with: pip install pillow")
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for image processing. Install with: pip install torch")
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for image processing. Install with: pip install torchvision")
        
        self.model_name = model_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.normalize = normalize
        self.verbose = verbose
        
        # Set device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        # Load model and transforms
        self.model, self.transform, self.feature_dim = self._load_model_and_transform()
        
        if self.verbose:
            print(f"Initialized {model_name} on {self.device}")
            print(f"Feature dimension: {self.feature_dim}")
    
    def process_zip(self,
                   zip_path: Union[str, Path, io.BytesIO],
                   labels: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Process images from ZIP file.
        
        Parameters
        ----------
        zip_path : str, Path, or BytesIO
            Path to ZIP file containing images
        labels : str, Path, or DataFrame
            Labels with columns: image, duration, event
            
        Returns
        -------
        features_df : pd.DataFrame
            DataFrame with extracted features and survival labels
        """
        with tempfile.TemporaryDirectory(prefix="images_") as tmp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            
            return self.process_directory(tmp_dir, labels)
    
    def process_directory(self,
                         directory: Union[str, Path],
                         labels: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Process all images in a directory.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing images
        labels : str, Path, or DataFrame
            Labels with columns: image, duration, event
            
        Returns
        -------
        features_df : pd.DataFrame
            DataFrame with extracted features and survival labels
        """
        directory = Path(directory)
        
        # Load labels
        if isinstance(labels, pd.DataFrame):
            labels_df = labels.copy()
        else:
            labels_df = pd.read_csv(labels)
        
        # Validate labels
        if 'image' not in labels_df.columns:
            if 'file' in labels_df.columns:
                labels_df['image'] = labels_df['file']
            else:
                raise ValueError("Labels must have 'image' or 'file' column")
        
        required_cols = {'duration', 'event'}
        if not required_cols.issubset(labels_df.columns):
            missing = required_cols - set(labels_df.columns)
            raise ValueError(f"Labels missing required columns: {missing}")
        
        # Find all images
        image_files = self._scan_images(directory)
        
        if self.verbose:
            print(f"Found {len(image_files)} images")
        
        # Match images with labels
        matched_images = []
        matched_labels = []
        
        for img_path in image_files:
            img_name = img_path.stem  # filename without extension
            
            # Try to find matching label
            matches = labels_df[labels_df['image'].str.contains(img_name, na=False)]
            
            if len(matches) == 0:
                # Try with full filename
                matches = labels_df[labels_df['image'].str.contains(img_path.name, na=False)]
            
            if len(matches) > 0:
                matched_images.append(img_path)
                matched_labels.append(matches.iloc[0])
            elif self.verbose:
                print(f"Warning: No label found for {img_path.name}")
        
        if not matched_images:
            raise ValueError("No images matched with labels")
        
        # Create dataset
        matched_labels_df = pd.DataFrame(matched_labels).reset_index(drop=True)
        dataset = ImageDataset(matched_images, matched_labels_df, self.transform)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device != 'cpu')
        )
        
        # Extract features
        if self.verbose:
            print(f"Extracting features from {len(matched_images)} images...")
        
        all_features = []
        all_labels = []
        all_paths = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, batch_labels, paths) in enumerate(dataloader):
                # Move to device
                images = images.to(self.device)
                
                # Extract features
                features = self.model(images)
                
                # Move to CPU and convert to numpy
                features = features.cpu().numpy()
                
                all_features.append(features)
                all_labels.extend(batch_labels)
                all_paths.extend(paths)
                
                if self.verbose and batch_idx % 10 == 0:
                    print(f"  Processed {min((batch_idx + 1) * self.batch_size, len(matched_images))}/{len(matched_images)}")
        
        # Combine features
        features_array = np.vstack(all_features)
        
        # Create DataFrame
        feature_cols = [f'img_feat_{i:04d}' for i in range(features_array.shape[1])]
        features_df = pd.DataFrame(features_array, columns=feature_cols)
        
        # Add labels
        labels_df = pd.DataFrame(all_labels).reset_index(drop=True)
        features_df = pd.concat([labels_df, features_df], axis=1)
        
        # Reorder columns: labels first
        label_cols = ['duration', 'event', 'image']
        extra_cols = [col for col in labels_df.columns if col not in label_cols]
        feature_cols = [col for col in features_df.columns if col.startswith('img_feat_')]
        
        col_order = label_cols + extra_cols + feature_cols
        col_order = [col for col in col_order if col in features_df.columns]
        features_df = features_df[col_order]
        
        if self.verbose:
            print(f"Extracted {len(feature_cols)} features from {len(features_df)} images")
        
        return features_df
    
    def process_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Process a single image and extract features.
        
        Parameters
        ----------
        image_path : str or Path
            Path to image file
            
        Returns
        -------
        features : np.ndarray
            Extracted feature vector
        """
        image_path = Path(image_path)
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Extract features
        self.model.eval()
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features.cpu().numpy().squeeze()
    
    def _scan_images(self, directory: Path) -> List[Path]:
        """Scan directory for image files."""
        image_files = []
        
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))
        
        return sorted(list(set(image_files)))
    
    def _load_model_and_transform(self) -> Tuple[torch.nn.Module, Any, int]:
        """Load pretrained model and preprocessing transforms."""
        
        if self.model_name == 'resnet50':
            # Load ResNet50
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = resnet50(weights=weights)
            
            # Remove final classification layer
            feature_dim = model.fc.in_features
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            # Get transforms
            transform = weights.transforms()
            
        elif self.model_name == 'resnet101':
            from torchvision.models import resnet101, ResNet101_Weights
            weights = ResNet101_Weights.IMAGENET1K_V2
            model = resnet101(weights=weights)
            feature_dim = model.fc.in_features
            model = torch.nn.Sequential(*list(model.children())[:-1])
            transform = weights.transforms()
            
        elif self.model_name == 'densenet121':
            from torchvision.models import densenet121, DenseNet121_Weights
            weights = DenseNet121_Weights.IMAGENET1K_V1
            model = densenet121(weights=weights)
            feature_dim = model.classifier.in_features
            model.classifier = torch.nn.Identity()
            transform = weights.transforms()
            
        elif self.model_name == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = torch.nn.Identity()
            transform = weights.transforms()
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Add global pooling if needed
        class GlobalPooling(torch.nn.Module):
            def forward(self, x):
                if len(x.shape) == 4:  # [B, C, H, W]
                    return x.mean(dim=[2, 3])
                return x
        
        model = torch.nn.Sequential(
            model,
            GlobalPooling()
        )
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        # Override transform if custom size specified
        if self.image_size != (224, 224):
            transform = T.Compose([
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]) if self.normalize else T.Lambda(lambda x: x)
            ])
        
        return model, transform, feature_dim
    
    def set_custom_model(self, model: torch.nn.Module, transform=None, feature_dim: int = None):
        """
        Set a custom feature extraction model.
        
        Parameters
        ----------
        model : torch.nn.Module
            Custom PyTorch model for feature extraction
        transform : callable, optional
            Custom preprocessing transform
        feature_dim : int, optional
            Output feature dimension
        """
        self.model = model.to(self.device)
        self.model.eval()
        
        if transform:
            self.transform = transform
        
        if feature_dim:
            self.feature_dim = feature_dim
        else:
            # Try to infer feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, *self.image_size).to(self.device)
                output = self.model(dummy_input)
                self.feature_dim = output.shape[1]
        
        if self.verbose:
            print(f"Set custom model with feature dimension: {self.feature_dim}")