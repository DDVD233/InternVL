import logging
from typing import Optional
import torch
import torch.nn as nn
from open_clip import create_model_and_transforms
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Constants for model paths
PMC_VIT_MODEL_PATH = "models/open_clip_model.safetensors"
PMC_VIT_CONFIG_PATH = "models/open_clip_config.json"
REPO_ID = "ryanyip7777/pmc_vit_l_14"


class PMCViTClassifier(nn.Module):
    def __init__(self, vision_output_size: int):
        super().__init__()

        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)

        # Download files if they don't exist
        if not Path(PMC_VIT_MODEL_PATH).exists():
            logger.info(f"Downloading model weights from {REPO_ID}...")
            hf_hub_download(
                REPO_ID,
                "open_clip_model.safetensors",
                local_dir="models",
                local_dir_use_symlinks=False
            )

        if not Path(PMC_VIT_CONFIG_PATH).exists():
            logger.info(f"Downloading model config from {REPO_ID}...")
            hf_hub_download(
                REPO_ID,
                "open_clip_config.json",
                local_dir="models",
                local_dir_use_symlinks=False
            )

        # Load config
        with open(PMC_VIT_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        # Initialize the PMC ViT model
        model, _, _ = create_model_and_transforms(
            'ViT-L-14',
            pretrained=PMC_VIT_MODEL_PATH
        )
        self.vision_model = model.visual  # Get just the visual part

        # Get the feature dimension from the model
        hidden_size = self.vision_model.output_dim

        # Add a new classification head
        self.fc = nn.Linear(hidden_size, vision_output_size)

        # Initialize the classification head weights
        std = 0.02  # Standard initialization for transformers
        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.zero_()

        # Required attributes to match your interface
        self.vision_only = True
        self.num_image_token = 1  # Uses CLS token

        self.dtype = self.fc.weight.dtype
        logger.info(f'Initialized PMCViTClassifier with output size: {vision_output_size}')

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

        # Forward through ViT
        features = self.vision_model(pixel_values)  # OpenCLIP ViT returns pooled features directly

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