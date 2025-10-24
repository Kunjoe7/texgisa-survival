# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Multimodal support for TEXGISA - ported from SAWEB.

This module provides:
- Image encoder (ResNet/ViT)
- Sensor encoder (CNN/Transformer)
- Multimodal fusion model
- Asset-backed dataset for streaming raw data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Union
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


# ---------------------- Image Encoder ----------------------

class ImageEncoder(nn.Module):
    """
    Image encoder using pretrained ResNet or Vision Transformer.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone.startswith("resnet"):
            import torchvision.models as models
            if backbone == "resnet18":
                self.encoder = models.resnet18(pretrained=pretrained)
                self.out_dim = 512
            elif backbone == "resnet50":
                self.encoder = models.resnet50(pretrained=pretrained)
                self.out_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")

            # Remove the final classification layer
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        elif backbone.startswith("vit"):
            try:
                import timm
                if backbone == "vit_base" or backbone == "vit_b_16":
                    self.encoder = timm.create_model('vit_base_patch16_224',
                                                    pretrained=pretrained,
                                                    num_classes=0)  # No classification head
                    self.out_dim = 768
                else:
                    raise ValueError(f"Unsupported ViT variant: {backbone}")
            except ImportError:
                raise ImportError("timm library required for ViT models. Install with: pip install timm")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Image features [B, out_dim]
        """
        features = self.encoder(x)
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        return features


# ---------------------- Sensor Encoder ----------------------

class SensorEncoder(nn.Module):
    """
    1D sensor/time-series encoder using CNN or Transformer.
    """

    def __init__(self,
                 input_channels: int = 1,
                 backbone: str = "cnn",
                 hidden_dim: int = 128):
        super().__init__()

        if backbone == "cnn":
            # 1D CNN for time series
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.out_dim = hidden_dim

        elif backbone == "transformer":
            # Simple transformer for sequences
            self.positional_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
            self.input_projection = nn.Linear(input_channels, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            self.out_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported sensor backbone: {backbone}")

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sensor data [B, C, L] for CNN or [B, L, C] for Transformer

        Returns
        -------
        torch.Tensor
            Sensor features [B, out_dim]
        """
        if self.backbone == "cnn":
            # Expects [B, C, L]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            features = self.encoder(x)
            features = features.squeeze(-1)

        elif self.backbone == "transformer":
            # Expects [B, L, C]
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # Add feature dimension
            elif x.dim() == 3 and x.size(1) == 1:
                # [B, 1, L] -> [B, L, 1]
                x = x.transpose(1, 2)

            B, L, C = x.shape
            x = self.input_projection(x)

            # Add positional encoding
            if L <= self.positional_encoding.size(1):
                x = x + self.positional_encoding[:, :L, :]

            features = self.encoder(x)
            features = features.transpose(1, 2)  # [B, hidden, L]
            features = self.pooling(features)

        return features


# ---------------------- Multimodal Fusion Model ----------------------

class MultiModalFusionModel(nn.Module):
    """
    Gating-based fusion model for multimodal survival analysis.
    """

    def __init__(self,
                 modality_configs: Dict[str, Dict[str, Any]],
                 num_bins: int,
                 hidden: int = 256,
                 depth: int = 2,
                 dropout: float = 0.2):
        """
        Parameters
        ----------
        modality_configs : dict
            Configuration for each modality:
            {
                'tabular': {'input_dim': 100},
                'image': {'backbone': 'resnet18', 'pretrained': True},
                'sensor': {'input_channels': 3, 'backbone': 'cnn'}
            }
        num_bins : int
            Number of time bins for survival
        hidden : int
            Hidden dimension
        depth : int
            Network depth
        dropout : float
            Dropout rate
        """
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.modality_order = []

        for name, config in modality_configs.items():
            self.modality_order.append(name)

            if name == "tabular":
                input_dim = config['input_dim']
                encoder = self._build_tabular_encoder(input_dim, hidden, depth, dropout)
                embed_dim = hidden

            elif name == "image":
                backbone = config.get('backbone', 'resnet18')
                pretrained = config.get('pretrained', True)
                encoder = ImageEncoder(backbone, pretrained)
                embed_dim = encoder.out_dim

            elif name == "sensor":
                input_channels = config.get('input_channels', 1)
                backbone = config.get('backbone', 'cnn')
                encoder = SensorEncoder(input_channels, backbone)
                embed_dim = encoder.out_dim
            else:
                raise ValueError(f"Unknown modality: {name}")

            self.encoders[name] = encoder

            # Project to common dimension
            if embed_dim != hidden:
                self.projectors[name] = nn.Sequential(
                    nn.Linear(embed_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                )
            else:
                self.projectors[name] = nn.Identity()

            # Gating for modality fusion
            self.gates[name] = nn.Linear(hidden, 1)

        # Final hazard prediction
        self.hazard_layer = nn.Linear(hidden, num_bins)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_tabular_encoder(self, input_dim: int, hidden: int,
                              depth: int, dropout: float) -> nn.Sequential:
        """Build MLP encoder for tabular data."""
        layers = []
        d = input_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self,
                x: Union[Dict[str, torch.Tensor], torch.Tensor],
                modality_mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : dict or tensor
            If dict: {modality_name: tensor} for each modality
            If tensor: concatenated features (for backward compatibility)
        modality_mask : torch.Tensor, optional
            Binary mask [B, num_modalities] indicating available modalities
        return_embeddings : bool
            Whether to return embeddings along with hazards

        Returns
        -------
        hazards : torch.Tensor
            Predicted hazards [B, num_bins]
        embeddings : torch.Tensor, optional
            Fused embeddings [B, hidden] if return_embeddings=True
        """
        if isinstance(x, dict):
            # Process each modality
            features = []
            gate_logits = []

            for idx, name in enumerate(self.modality_order):
                if name not in x:
                    raise ValueError(f"Missing modality: {name}")

                # Encode modality
                encoded = self.encoders[name](x[name])
                projected = self.projectors[name](encoded)

                # Apply modality mask if provided
                if modality_mask is not None:
                    mask = modality_mask[:, idx].unsqueeze(1)
                    projected = projected * mask

                features.append(projected)
                gate_logits.append(self.gates[name](projected))

            # Gating-based fusion
            logits = torch.cat(gate_logits, dim=1)  # [B, num_modalities]

            if modality_mask is not None:
                # Mask out unavailable modalities
                logits = logits.masked_fill(modality_mask == 0, -1e4)

            attn = torch.softmax(logits, dim=1)

            # Weighted fusion
            B = x[self.modality_order[0]].size(0)
            device = x[self.modality_order[0]].device
            fused = torch.zeros(B, features[0].size(1), device=device)

            for idx, feat in enumerate(features):
                fused = fused + feat * attn[:, idx].unsqueeze(1)

        else:
            # Backward compatibility: assume tabular only
            if 'tabular' not in self.encoders:
                raise ValueError("Tabular encoder required for tensor input")

            encoded = self.encoders['tabular'](x)
            fused = self.projectors['tabular'](encoded)

        # Predict hazards
        hazards = torch.sigmoid(self.hazard_layer(fused))

        if return_embeddings:
            return hazards, fused
        return hazards


# ---------------------- Asset-backed Dataset ----------------------

class SensorSpec:
    """Specification for sensor data."""

    def __init__(self, channels: int = 1, target_len: int = 1000):
        self.channels = channels
        self.target_len = target_len


class AssetBackedMultiModalDataset(Dataset):
    """
    Dataset that streams raw assets (images, sensors) from disk.
    Keeps tabular data in memory for efficiency.
    """

    def __init__(self,
                 ids: List[str],
                 tabular: np.ndarray,
                 labels: np.ndarray,
                 masks: np.ndarray,
                 modality_mask: np.ndarray,
                 tabular_names: List[str],
                 image_paths: Optional[List[Optional[str]]] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 sensor_paths: Optional[List[Optional[str]]] = None,
                 sensor_spec: Optional[SensorSpec] = None):
        """
        Parameters
        ----------
        ids : list
            Sample IDs
        tabular : np.ndarray
            Tabular features [N, D]
        labels : np.ndarray
            Supervision labels [N, T]
        masks : np.ndarray
            Risk set masks [N, T]
        modality_mask : np.ndarray
            Modality availability [N, num_modalities]
        tabular_names : list
            Feature names for tabular data
        image_paths : list, optional
            Paths to image files
        image_size : tuple
            Target image size (H, W)
        sensor_paths : list, optional
            Paths to sensor data files
        sensor_spec : SensorSpec, optional
            Sensor data specification
        """
        self.ids = ids
        self.tabular = torch.from_numpy(tabular.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.masks = torch.from_numpy(masks.astype(np.float32))
        self.modality_mask = torch.from_numpy(modality_mask.astype(np.float32))
        self.tabular_names = tabular_names

        self.image_paths = image_paths
        self.image_size = image_size
        self.sensor_paths = sensor_paths
        self.sensor_spec = sensor_spec or SensorSpec()

        # Lazy import for image loading
        self._pil_image = None
        self._transforms = None

    def __len__(self):
        return len(self.ids)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image."""
        if self._pil_image is None:
            from PIL import Image
            self._pil_image = Image

        if self._transforms is None:
            from torchvision import transforms
            self._transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        try:
            img = self._pil_image.open(path).convert('RGB')
            return self._transforms(img)
        except Exception:
            # Return zeros if loading fails
            return torch.zeros(3, *self.image_size)

    def _load_sensor(self, path: str) -> torch.Tensor:
        """Load and preprocess sensor data."""
        try:
            # Try to load as numpy file
            if path.endswith('.npy'):
                data = np.load(path)
            elif path.endswith('.csv'):
                import pandas as pd
                data = pd.read_csv(path).values
            else:
                # Try numpy loadtxt
                data = np.loadtxt(path)

            # Ensure correct shape [channels, length]
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T

            # Resample to target length
            if data.shape[1] != self.sensor_spec.target_len:
                data = self._resample_sequence(data, self.sensor_spec.target_len)

            return torch.from_numpy(data.astype(np.float32))

        except Exception:
            # Return zeros if loading fails
            return torch.zeros(self.sensor_spec.channels, self.sensor_spec.target_len)

    def _resample_sequence(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        """Resample sequence to target length."""
        current_len = seq.shape[1]
        if current_len == target_len:
            return seq

        # Simple linear interpolation
        indices = np.linspace(0, current_len - 1, target_len)
        resampled = np.zeros((seq.shape[0], target_len))

        for c in range(seq.shape[0]):
            resampled[c] = np.interp(indices, np.arange(current_len), seq[c])

        return resampled

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = {
            'id': self.ids[idx],
            'modalities': {'tabular': self.tabular[idx]},
            'y': self.labels[idx],
            'm': self.masks[idx],
            'modality_mask': self.modality_mask[idx]
        }

        # Load image if available
        if self.image_paths and self.image_paths[idx]:
            sample['modalities']['image'] = self._load_image(self.image_paths[idx])

        # Load sensor if available
        if self.sensor_paths and self.sensor_paths[idx]:
            sample['modalities']['sensor'] = self._load_sensor(self.sensor_paths[idx])

        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        ids = [s['id'] for s in batch]

        # Stack modalities
        modality_names = batch[0]['modalities'].keys()
        modalities = {}
        for name in modality_names:
            if name in batch[0]['modalities']:
                modalities[name] = torch.stack([s['modalities'][name] for s in batch])

        return {
            'ids': ids,
            'modalities': modalities,
            'y': torch.stack([s['y'] for s in batch]),
            'm': torch.stack([s['m'] for s in batch]),
            'modality_mask': torch.stack([s['modality_mask'] for s in batch])
        }