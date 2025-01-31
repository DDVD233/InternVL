import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class RadDinoClassifier(PreTrainedModel):
    config_class = AutoConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['ViTLayer']

    def __init__(self, config, vision_output_size=-1):
        super().__init__(config)

        self.vision_model = AutoModel.from_config(config)
        self.vision_model = self.vision_model.to(torch.bfloat16)
        self.img_size = 518  # As specified in the model card for RAD-DINO
        hidden_size = config.hidden_size  # ViT models use hidden_size directly

        # Add classification head
        self.fc = nn.Linear(hidden_size, vision_output_size)

        # Initialize weights
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                m.bias.data.zero_()

        # Required attributes to match interface
        self.vision_only = True
        self.num_image_token = 1  # RAD-DINO uses CLS token

        logger.info(f'Initialized RadDinoClassifier with output size: {vision_output_size}')

    def forward_vision(self, pixel_values, attention_mask=None, classify=True, **kwargs):
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

        # Forward through RAD-DINO
        outputs = self.vision_model(pixel_values)
        features = outputs.pooler_output  # Shape: (b*n, hidden_size)

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

    def classify(self, features):
        """
        Apply classification head to features.

        Args:
            features (torch.Tensor): Input features

        Returns:
            torch.Tensor: Classification logits
        """
        return self.fc(features)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path="microsoft/rad-dino", vision_output_size=-1, *model_args, **kwargs):
        """
        Load pretrained RAD-DINO model and add classification head.

        Args:
            pretrained_model_name_or_path (str): HuggingFace model name or path, defaults to microsoft/rad-dino
            vision_output_size (int): Number of output classes
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            RadDinoClassifier: Initialized model
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, vision_output_size=vision_output_size, *model_args, **kwargs)

        # Load pretrained weights
        pretrained_model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        state_dict = pretrained_model.state_dict()

        # Load only the vision model weights
        missing_keys, unexpected_keys = model.vision_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")

        return model
