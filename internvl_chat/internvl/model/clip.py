import logging

import torch
import torch.nn as nn
from open_clip import create_model_and_transforms

logger = logging.getLogger(__name__)


class OpenCLIPClassifier(nn.Module):
    def __init__(self, model_name='ViT-g-14', pretrained='laion2b_s34b_b88k', vision_output_size=-1,
                 dtype=torch.float32):
        super().__init__()

        # Load the CLIP model with specified dtype
        self.vision_model, _, _ = create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision='bf16' if dtype == torch.bfloat16 else 'fp32',
            device='cuda',
            jit=False
        )

        hidden_size = self.vision_model.visual.output_dim
        self.img_size = 224  # Default for ViT-B/16

        # Initialize classifier in the same dtype
        self.fc = nn.Linear(hidden_size, vision_output_size, dtype=dtype)

        # Initialize the classifier weights
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()

        # Convert entire model to specified dtype
        self.to(dtype)

        # Required attributes for compatibility with training script
        self.vision_only = True
        self.num_image_token = 1
        self.dtype = dtype

        logger.info(f'Initialized OpenCLIPClassifier with:')
        logger.info(f'- Output size: {vision_output_size}')
        logger.info(f'- Image size: {self.img_size}x{self.img_size}')
        logger.info(f'- Hidden size: {hidden_size}')
        logger.info(f'- Dtype: {dtype}')

    def forward_vision(self, pixel_values, attention_mask=None, classify=True):
        """
        Forward pass through the vision model.
        """
        b, n, c, h, w = pixel_values.shape
        if h != self.img_size or w != self.img_size:
            logger.warning(f'Input size ({h}x{w}) does not match model size ({self.img_size}x{self.img_size})')

        pixel_values = pixel_values.view(b * n, c, h, w)

        # Forward through CLIP's vision encoder
        with torch.cuda.amp.autocast(dtype=self.dtype):
            features = self.vision_model.encode_image(pixel_values)
            features = features.view(b, n, -1)  # Reshape to include frame dimension

            if attention_mask is not None:
                attention_mask = attention_mask.to(features.dtype)
                mask = attention_mask.unsqueeze(-1)
                features = features * mask
                features_sum = features.sum(dim=1)
                mask_sum = mask.sum(dim=1).clamp(min=1.0)
                features = features_sum / mask_sum
            else:
                features = features.mean(dim=1)

            if classify:
                return self.classify(features)
        return features

    def classify(self, features):
        with torch.cuda.amp.autocast(dtype=self.dtype):
            return self.fc(features)

    @classmethod
    def from_pretrained(cls, model_path, vision_output_size=-1, dtype=torch.bfloat16, *args, **kwargs):
        if 'ViT-g-14' in model_path:
            model_name = 'ViT-g-14'
            pretrained = 'laion2b_s34b_b88k'
        else:
            raise ValueError(f"Unsupported model path: {model_path}")

        return cls(model_name=model_name,
                   pretrained=pretrained,
                   vision_output_size=vision_output_size,
                   dtype=dtype)