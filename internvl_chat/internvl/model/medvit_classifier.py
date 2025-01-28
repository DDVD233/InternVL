import logging
from typing import Optional
import torch
import torch.nn as nn
from functools import partial
from .med_vit import MedViT

logger = logging.getLogger(__name__)


class MedViTClassifier(nn.Module):
    def __init__(self, vision_output_size: int):
        super().__init__()

        # Initialize the MedViT model - using small variant as example
        self.vision_model = MedViT(
            stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2,
            num_classes=0  # Remove the classification head
        )

        # Get the feature dimension from the model (1024 for MedViT as per the original implementation)
        hidden_size = self.vision_model.output_channel

        # Add a new classification head
        self.fc = nn.Linear(hidden_size, vision_output_size)

        # Initialize the classification head weights
        std = 0.02
        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.zero_()

        # Required attributes to match your interface
        self.vision_only = True
        self.num_image_token = 1  # Uses global average pooling

        self.dtype = self.fc.weight.dtype
        logger.info(f'Initialized MedViTClassifier with output size: {vision_output_size}')

    def forward_vision(
            self,
            pixel_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            classify: bool = True,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the vision model.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_frames, channels, height, width)
            attention_mask (torch.Tensor, optional): Attention mask for frame aggregation
            classify (bool): Whether to apply classification head

        Returns:
            torch.Tensor: Either features or classification logits
        """
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * n, c, h, w)

        # Forward through stem and features
        x = self.vision_model.stem(pixel_values)
        for layer in self.vision_model.features:
            x = layer(x)

        # Apply normalization
        x = self.vision_model.norm(x)

        # Global average pooling
        features = self.vision_model.avgpool(x)
        features = torch.flatten(features, 1)  # Shape: (b*n, hidden_size)

        # Reshape back to include frame dimension
        features = features.view(b, n, -1)  # Shape: (b, n, hidden_size)

        if attention_mask is not None:
            # Apply temporal attention mask
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
        """
        Apply classification head to features.

        Args:
            features (torch.Tensor): Input features

        Returns:
            torch.Tensor: Classification logits
        """
        return self.fc(features)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.vision_model = self.vision_model.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        return self