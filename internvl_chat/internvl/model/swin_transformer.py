import logging
import torch
import torch.nn as nn
from typing import Optional

from torch import Tensor
from torch.nn import functional as F
from transformers import Swinv2Model, Swinv2Config

logger = logging.getLogger(__name__)


class TemporalAttentionPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        """
        Initialize temporal attention pooling layer.

        Args:
            embed_dim (int): Embedding dimension of features
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable temporal query token
        self.temporal_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights with truncated normal distribution
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

        # Initialize temporal query
        nn.init.normal_(self.temporal_query, std=0.02)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention pooling.

        Args:
            x: Input features of shape (batch_size, num_frames, embed_dim)
            attention_mask: Optional mask of shape (batch_size, num_frames)

        Returns:
            Pooled features of shape (batch_size, embed_dim)
        """
        batch_size, num_frames, embed_dim = x.shape

        # Expand temporal query for batch size
        query = self.temporal_query.expand(batch_size, -1, -1)

        # Apply layer norm to input
        x = self.norm(x)

        # Project queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting
            attention_mask: Tensor = attention_mask.view(batch_size, 1, 1, num_frames)
            attention_mask = attention_mask.expand(-1, self.num_heads, 1, -1)

            # Apply mask with large negative value for masked positions
            attn_weights = attn_weights.masked_fill(
                attention_mask == 0,
                float('-inf')
            )

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, 1, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output.squeeze(1)


class SwinV2Classifier(nn.Module):
    def __init__(
            self,
            vision_output_size: int,
            model_name: str = "microsoft/swinv2-large-patch4-window12-192-22k",
            pretrained: bool = True,
            num_attention_heads: int = 8
    ):
        """
        Initialize Swin Transformer V2 classifier.

        Args:
            vision_output_size (int): Number of output classes
            model_name (str): Name of pretrained model to load
            pretrained (bool): Whether to load pretrained weights
            num_attention_heads (int): Number of heads in temporal attention pooling
        """
        super().__init__()

        self.vision_only = True
        self.num_image_token = 1

        # Initialize Swin V2 model
        if pretrained:
            self.vision_model = Swinv2Model.from_pretrained(model_name)
            logger.info(f"Loaded pretrained weights from {model_name}")
        else:
            config = Swinv2Config.from_pretrained(model_name)
            self.vision_model = Swinv2Model(config)
            logger.info("Initialized model with random weights")

        # Get embedding dimension from model config
        self.embed_dim = self.vision_model.config.hidden_size

        # Add temporal attention pooling
        self.temporal_pool = TemporalAttentionPool(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads
        )

        # Add classification head
        self.fc = nn.Linear(self.embed_dim, vision_output_size)

        # Initialize the classification head
        std = 0.02
        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.zero_()

        # Set default dtype
        self.dtype = getattr(torch, 'bfloat16')
        logger.info(f'Initialized SwinV2Classifier with output size: {vision_output_size}')

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
        # Reshape input for multi-frame processing
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * n, c, h, w)

        # Forward pass through Swin V2
        outputs = self.vision_model(pixel_values, return_dict=True)
        features = outputs.pooler_output

        # Reshape features back to include frame dimension
        features = features.view(b, n, -1)

        # Apply temporal attention pooling
        features = self.temporal_pool(features, attention_mask)

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
        self.temporal_pool = self.temporal_pool.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)
        # Update dtype if it's in the arguments
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        return self

    def train(self, mode: bool = True):
        """Set train/eval mode"""
        super().train(mode)
        self.vision_model.train(mode)
        self.temporal_pool.train(mode)
        return self

    def eval(self):
        """Set eval mode"""
        super().eval()
        self.vision_model.eval()
        self.temporal_pool.eval()
        return self
