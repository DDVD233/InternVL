import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from functools import partial
from apex.normalization import FusedLayerNorm
import math
import torch.utils.checkpoint as checkpoint
from .eva import VisionTransformer

logger = logging.getLogger(__name__)


class EVA02Classifier(nn.Module):
    def __init__(self, vision_output_size: int, checkpoint_path: str = None):
        super().__init__()

        # Initialize model attributes
        self.vision_only = True
        self.num_image_token = 1

        # Model configuration for EVA-02-Large
        model_config = {
            'img_size': 448,
            'patch_size': 14,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4 * 2 / 3,
            'qkv_bias': True,
            'norm_layer': partial(FusedLayerNorm, eps=1e-6),
            'subln': True,
            'xattn': True,
            'naiveswiglu': True,
            'rope': True,
            'pt_hw_seq_len': 16,
            'intp_freq': True,
        }

        # Initialize vision transformer
        self.vision_model = VisionTransformer(
            num_classes=0,  # Remove classification head
            **model_config
        )

        # Add classification head
        self.fc = nn.Linear(model_config['embed_dim'], vision_output_size)

        # Initialize the classification head
        std = 0.02
        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.zero_()

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.dtype = getattr(torch, 'bfloat16')  # Default to bfloat16
        logger.info(f'Initialized EVA02Classifier with output size: {vision_output_size}')

    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'module' in checkpoint:
                checkpoint = checkpoint['module']

            # Remove any classification head weights from checkpoint
            # checkpoint = {k: v for k, v in checkpoint.items()
            #               if not k.startswith('head.')}

            missing, unexpected = self.vision_model.load_state_dict(checkpoint, strict=False)

            if missing:
                logger.warning(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def forward_vision(
            self,
            pixel_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            classify: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the vision model.

        Args:
            pixel_values: Input images (batch_size, num_frames, channels, height, width)
            attention_mask: Attention mask for frame aggregation
            classify: Whether to apply classification head

        Returns:
            torch.Tensor: Features or classification logits
        """
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * n, c, h, w)

        # Get features from vision transformer
        features = self.vision_model.forward_features(pixel_values, return_patch_tokens=False)

        # Reshape back to include frame dimension
        features = features.view(b, n, -1)

        if attention_mask is not None:
            # Apply temporal attention mask for frame aggregation
            attention_mask = attention_mask.to(features.dtype)
            mask = attention_mask.unsqueeze(-1)
            features = features * mask
            features_sum = features.sum(dim=1)
            mask_sum = mask.sum(dim=1).clamp(min=1.0)
            features = features_sum / mask_sum
        else:
            # Simple average pooling over frames
            features = features.mean(dim=1)

        if classify:
            return self.classify(features)
        return features

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Apply classification head to features"""
        return self.fc(features)

    def to(self, *args, **kwargs):
        """Handle moving model to device and changing dtype"""
        self = super().to(*args, **kwargs)
        self.vision_model = self.vision_model.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        # Update dtype if it's in the arguments
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        return self