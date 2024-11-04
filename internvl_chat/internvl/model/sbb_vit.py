import logging
from typing import Optional

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ViTSBBClassifier(nn.Module):
    def __init__(self, vision_output_size: int):
        super().__init__()

        # Initialize the ViT model without the classification head
        self.vision_model = timm.create_model(
            'vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k',  # Fixed model name
            pretrained=True,
            num_classes=0  # Remove the classification head
        )

        # Get the feature dimension from the model
        hidden_size = self.vision_model.num_features

        # Add a new classification head
        self.fc = nn.Linear(hidden_size, vision_output_size)

        # Initialize the classification head weights
        std = 0.02
        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.zero_()

        # Required attributes to match your interface
        self.vision_only = True
        self.num_image_token = 1  # ViT uses global average pooling

        # Get model specific transforms
        self.data_config = timm.data.resolve_model_data_config(self.vision_model)
        self.dtype = self.fc.weight.dtype
        logger.info(f'Initialized ViTSBBClassifier with output size: {vision_output_size}')

    def forward_vision(
            self,
            pixel_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            classify: bool = True
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

        # Get features before pooling
        features = self.vision_model.forward_features(pixel_values)  # Shape: (b*n, num_patches, hidden_size)

        # Apply global average pooling
        features = self.vision_model.forward_head(features, pre_logits=True)  # Shape: (b*n, hidden_size)

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