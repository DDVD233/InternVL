import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from timm.models.layers import DropPath
from internvl.model.internvl_chat.configuration_internvl_chat import InternVisionConfig
from internvl.model.internvl_chat.modeling_internvl_chat import AttentionPoolingBlock, InternVLChatModel
from internvl.model.internvl_chat.modeling_intern_vit import InternVisionEmbeddings, InternAttention
from functools import partial
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s")
logger = logging.getLogger(__name__)


class ExpertLayer(nn.Module):
    """Expert MLP layer for transformer blocks"""

    def __init__(self, hidden_size: int, intermediate_size: int, activation_fn=F.gelu):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MoELayer(nn.Module):
    """Mixture of Experts layer with capacity-based routing"""

    def __init__(self, config, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Top-k experts to route to
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertLayer(self.hidden_size, self.intermediate_size)
            for _ in range(num_experts)
        ])

        # Router network
        self.router = nn.Linear(self.hidden_size, num_experts)

        # Load balancing loss coefficient
        self.balance_loss_coef = 0.01

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute routing probabilities
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Reshape input for expert computation
        hidden_states = hidden_states.view(-1, hidden_size)

        # Compute expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            # Find which positions need this expert
            expert_mask = (top_k_indices == i).any(dim=-1).view(-1)
            if expert_mask.any():
                expert_inputs = hidden_states[expert_mask]
                expert_outputs[expert_mask] += expert(expert_inputs)

        # Reshape output back to original size
        output = expert_outputs.view(batch_size, seq_len, hidden_size)

        # Compute load balancing loss
        importance = routing_weights.mean(dim=[0, 1])
        load_balancing_loss = self.balance_loss_coef * (self.num_experts * importance * torch.log(importance)).sum()

        return output, load_balancing_loss


class MoEVisionEncoderLayer(nn.Module):
    """Vision transformer encoder layer with MoE FFN"""

    def __init__(self, config, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_experts = getattr(config, "num_experts", 4)

        # Regular attention mechanism
        self.attention = InternAttention(config)
        # Replace standard MLP with MoE
        self.moe = MoELayer(config, num_experts=self.num_experts)

        # Layer norms and other components
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention block
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.drop_path1(hidden_states * self.ls1)
        hidden_states = residual + hidden_states

        # MoE block
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states, load_loss = self.moe(hidden_states)
        hidden_states = self.drop_path2(hidden_states * self.ls2)
        hidden_states = residual + hidden_states

        return hidden_states, load_loss


class MoEVisionEncoder(nn.Module):
    """Vision transformer encoder with MoE layers"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]

        # Create encoder layers with MoE
        self.layers = nn.ModuleList([
            MoEVisionEncoderLayer(config, dpr[i])
            for i in range(config.num_hidden_layers)
        ])

        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds: torch.Tensor,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_load_losses = []
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states, load_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                )
            else:
                hidden_states, load_loss = encoder_layer(hidden_states)

            all_load_losses.append(load_loss)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        total_load_loss = sum(all_load_losses)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, total_load_loss] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=None,
        ), total_load_loss


class MoEVisionModel(PreTrainedModel):
    config_class = InternVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config, vision_output_size=-1):
        super().__init__(config)
        self.config = config

        # Keep the original embeddings
        self.embeddings = InternVisionEmbeddings(config)
        # Replace the encoder with MoE version
        self.encoder = MoEVisionEncoder(config)

        # Add classification components
        # First the attention pooling
        self.clip_projector = AttentionPoolingBlock(
            dim=config.hidden_size,
            num_heads=8,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-5),
            out_dim=512
        )

        # Then the final classifier
        self.fc = nn.Linear(512, vision_output_size)

        # Initialize the classifier
        torch.nn.init.normal_(self.fc.weight, std=0.02)
        torch.nn.init.constant_(self.fc.bias, 0)

    def forward_vision(self, pixel_values, attention_mask=None, classify=True):
        """Forward pass with classification option"""
        batch_size, num_frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_frames, channels, height, width)

        # Get embeddings
        features = self.embeddings(pixel_values)

        # Pass through encoder
        encoder_outputs, load_loss = self.encoder(
            inputs_embeds=features,
            output_hidden_states=False,
            return_dict=True,
        )
        features = encoder_outputs.last_hidden_state[:, 1:, :]  # Remove CLS token

        # Reshape features
        b_n, np, c = features.shape
        features = features.view(batch_size, num_frames, np, c)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(features.dtype)
            mask = attention_mask.unsqueeze(-1).unsqueeze(-1)
            features = features * mask

        # Pool features
        pooled_features = []
        for i in range(batch_size):
            if attention_mask is not None:
                valid_length = int(attention_mask[i].sum().item())
                frame_features = features[i, :valid_length]
            else:
                frame_features = features[i]
            pooled = self.clip_projector(frame_features)
            pooled = pooled.mean(0)
            pooled_features.append(pooled)

        features = torch.stack(pooled_features)

        if classify:
            features = self.fc(features)

        return features, load_loss

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f"Wrong pixel_values size: {pixel_values.shape}")

        encoder_outputs, load_loss = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output, load_loss) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=None,
        ), load_loss

    def load_from_internvl(self, pretrained_model_name_or_path: str, noise_scale: float = 0.01):
        """
        Load weights from a pretrained InternVL model, adapting them for MoE architecture.

        Args:
            pretrained_model_name_or_path (str): HuggingFace model name or path
            noise_scale (float): Scale of noise to add to duplicated expert weights
        """
        from transformers import AutoModel

        # Load the original model
        original_model = InternVLChatModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=self.dtype
        )
        original_model = original_model.vision_model

        # Load embeddings and other non-MoE components
        embed_dict = {
            k: v for k, v in original_model.state_dict().items()
            if 'embeddings' in k or 'clip_projector' in k or 'fc' in k
        }
        missing_keys, unexpected_keys = self.load_state_dict(embed_dict, strict=False)

        # Load attention layers (they remain unchanged)
        attn_dict = {
            k: v for k, v in original_model.state_dict().items()
            if 'attention' in k or 'norm1' in k or 'ls1' in k
        }
        missing_keys, unexpected_keys = self.load_state_dict(attn_dict, strict=False)

        # Handle MoE layers
        for layer_idx in range(len(self.encoder.layers)):
            # Get original MLP weights
            original_mlp_dict = {
                k.replace(f'encoder.layers.{layer_idx}.mlp.', ''): v
                for k, v in original_model.state_dict().items()
                if f'encoder.layers.{layer_idx}.mlp.' in k
            }

            # Initialize each expert with original weights plus noise
            for expert_idx in range(self.encoder.layers[layer_idx].moe.num_experts):
                expert = self.encoder.layers[layer_idx].moe.experts[expert_idx]

                # Add random noise to weights
                noisy_state_dict = {}
                for k, v in original_mlp_dict.items():
                    if 'weight' in k:
                        noise = torch.randn_like(v) * noise_scale * v.std()
                        noisy_state_dict[k] = v + noise
                    else:  # bias terms
                        noisy_state_dict[k] = v.clone()

                # Load noisy weights into expert
                expert.load_state_dict(noisy_state_dict)

            # Initialize router with random weights
            nn.init.normal_(
                self.encoder.layers[layer_idx].moe.router.weight,
                std=0.02
            )
            if self.encoder.layers[layer_idx].moe.router.bias is not None:
                nn.init.zeros_(self.encoder.layers[layer_idx].moe.router.bias)

            # Copy layer norms and other components
            norm_dict = {
                k: v for k, v in original_model.state_dict().items()
                if f'encoder.layers.{layer_idx}.norm2' in k or f'encoder.layers.{layer_idx}.ls2' in k
            }
            self.encoder.layers[layer_idx].load_state_dict(norm_dict, strict=False)

        logger.info("Successfully loaded pretrained weights with MoE adaptation")

        # Verify all components are loaded
        loaded_keys = set(embed_dict.keys()) | set(attn_dict.keys())
        original_keys = set(original_model.state_dict().keys())
        converted_keys = {k for k in original_keys if 'mlp' not in k}

        if loaded_keys != converted_keys:
            missing = converted_keys - loaded_keys
            extra = loaded_keys - converted_keys
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if extra:
                logger.warning(f"Unexpected keys: {extra}")

        return self
