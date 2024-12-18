# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import copy
import os
import warnings
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import get_peft_model, LoraConfig
from timm.models.layers import DropPath
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import AutoModel, GenerationConfig, LlamaForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None,
                 vision_only=False, vision_output_size=1, all_separate=False, modalities=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        self.all_separate = all_separate
        self.modalities = modalities

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.vision_only = vision_only
        self.clip_projector = AttentionPoolingBlock(
            dim=vit_hidden_size,
            num_heads=8,  # Can be configured in config
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-5),
            out_dim=512
        )
        text_hidden_size = config.llm_config.hidden_size
        self.contrastive_query_embeds = nn.Parameter(
            torch.zeros(1, 96, text_hidden_size)  # [1, num_queries, hidden_size]
        )
        torch.nn.init.normal_(self.contrastive_query_embeds, std=0.02)

        self.fc = nn.Linear(512, vision_output_size)

        # init_weights
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                m.bias.data.zero_()
        if vision_only:
            self.language_model = None
            self.mlp1 = None

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def init_encoder_array(self):
        """Initialize separate encoders for each modality."""
        if not self.all_separate or not self.modalities:
            logger.warning("Separate encoders are not enabled or modalities are not provided.")
            return

        logger.info(f"Initializing separate encoders for modalities: {self.modalities}")

        # Create array of encoders
        self.encoder_array = nn.ModuleList([
            copy.deepcopy(self.vision_model) for _ in range(len(self.modalities))
        ])

        # Create mapping from modality to encoder index
        self.modality_to_encoder = {
            modality: idx for idx, modality in enumerate(self.modalities)
        }

        # Load weights from original model to all encoders
        original_state_dict = self.vision_model.state_dict()
        for encoder in self.encoder_array:
            encoder.load_state_dict(original_state_dict)

        logger.info(f"Initialized {len(self.encoder_array)} separate encoders")
        logger.info(f"Modality to encoder mapping: {self.modality_to_encoder}")

        # Remove the original vision model to save memory
        self.vision_model = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # First load the model normally
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # After loading pre-trained weights, initialize our new components
        model._init_attention_pool()
        nn.init.trunc_normal_(model.fc.weight, std=0.02)
        nn.init.constant_(model.fc.bias, 0)

        return model

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def _init_attention_pool(self):
        """Initialize the attention pooling parameters."""
        # Initialize query tokens
        torch.nn.init.normal_(self.contrastive_query_embeds, std=0.02)
        for name, param in self.clip_projector.named_parameters():
            if 'q' in name:
                nn.init.trunc_normal_(param, std=0.02)
            elif 'k' in name or 'v' in name:
                nn.init.trunc_normal_(param, std=0.02)
            elif 'proj' in name:
                nn.init.trunc_normal_(param, std=0.02)
                if 'bias' in name:
                    nn.init.constant_(param, 0)
            elif 'norm' in name:
                if 'weight' in name:
                    nn.init.constant_(param, 1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward_vision(self, pixel_values, attention_mask=None, classify=True, modalities=None):
        """Forward pass with support for multiple encoders."""
        assert self.vision_only, "This method is only available in vision_only mode."

        if not self.all_separate or not modalities:
            # Use original forward_vision logic if not using separate encoders
            return self._forward_vision_single(
                self.vision_model,
                pixel_values,
                attention_mask,
                classify
            )

        # Group samples by modality
        b, n, c, h, w = pixel_values.shape
        batch_indices = torch.arange(b, device=pixel_values.device)
        features_list = []

        # Process each modality separately
        for modality in self.modalities:
            # Find samples belonging to this modality
            modality_mask = torch.tensor(
                [m == modality for m in modalities],
                device=pixel_values.device
            )
            if not modality_mask.any():
                continue

            # Get encoder for this modality
            encoder_idx = self.modality_to_encoder[modality]
            encoder = self.encoder_array[encoder_idx]

            # Select samples for this modality
            modality_indices = batch_indices[modality_mask]
            modality_pixels = pixel_values[modality_mask]
            modality_attention = attention_mask[modality_mask] if attention_mask is not None else None

            # Process samples with corresponding encoder
            modality_features = self._forward_vision_single(
                encoder,
                modality_pixels,
                modality_attention,
                classify=False  # We'll classify after combining all features
            )

            # Store features in the original batch order
            features = torch.zeros(
                b,
                modality_features.size(1),
                device=modality_features.device,
                dtype=modality_features.dtype
            )
            features[modality_mask] = modality_features
            features_list.append(features)

        # Combine features from all modalities
        if len(features_list) > 1:
            # Sum up features where we have them
            combined_features = torch.stack(features_list).sum(0)
        else:
            combined_features = features_list[0]

        if classify:
            return self.classify(combined_features)
        else:
            return combined_features

    def _forward_vision_single(self, vision_model, pixel_values, attention_mask=None, classify=True):
        """Helper method for processing a single modality."""
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * n, c, h, w)
        pixel_values = pixel_values.to(vision_model.dtype)

        features = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True
        ).last_hidden_state[:, 1:, :]

        b_n, np, c = features.shape
        features = features.view(b, n, np, c)

        if attention_mask is not None:
            attention_mask = attention_mask.to(features.dtype)
            mask = attention_mask.unsqueeze(-1).unsqueeze(-1)
            features = features * mask

        pooled_features = []
        for i in range(b):
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
            return self.classify(features)
        else:
            return features

    def forward_contrastive(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass that combines:
        1. Contrastive learning between image and text embeddings
        2. Generative caption training

        Args:
            pixel_values: Image input tensor [batch_size, channels, height, width]
            input_ids: Caption token ids [batch_size, seq_len]
            attention_mask: Attention mask for captions [batch_size, seq_len]

        Returns:
            Tuple of (image_embeddings, text_embeddings, generative_loss)
        """
        # Extract vision features and transform to language model dimension
        vit_embeds = self.extract_feature(pixel_values)
        image_embeds = self.mlp1(vit_embeds)

        batch_size = pixel_values.shape[0]
        num_image_tokens = vit_embeds.shape[1]

        # Get caption embeddings
        caption_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Expand query embeddings
        query_embeds = self.contrastive_query_embeds.expand(batch_size, -1, -1)

        # For contrastive learning - combine all embeddings
        contrastive_embeds = torch.cat([image_embeds, query_embeds, caption_embeds], dim=1)

        # Create attention masks for contrastive path
        image_attention = torch.ones(
            (batch_size, num_image_tokens),
            dtype=torch.long,
            device=image_embeds.device
        )
        query_attention = torch.ones(
            (batch_size, query_embeds.shape[1]),
            dtype=torch.long,
            device=image_embeds.device
        )
        contrastive_attention_mask = torch.cat([image_attention, query_attention, attention_mask], dim=1)


        ignore_tokens = torch.full(
            (batch_size, num_image_tokens + query_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=input_ids.device
        )
        gen_labels = torch.cat([ignore_tokens, input_ids], dim=1)

        # Forward pass for contrastive learning
        outputs = self.language_model(
            inputs_embeds=contrastive_embeds,
            attention_mask=contrastive_attention_mask,
            output_hidden_states=True,
            labels=gen_labels,
            return_dict=True,
        )

        # Get embeddings for contrastive loss
        hidden_states = outputs.hidden_states[-1]
        image_embeddings = hidden_states[:, :num_image_tokens].mean(dim=1)
        text_embeddings = hidden_states[:, num_image_tokens:].mean(dim=1)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        generative_loss = outputs.loss

        return image_embeddings, text_embeddings, generative_loss

    def forward_generate(
            self,
            pixel_values: torch.FloatTensor,
            tokenizer,
            max_new_tokens: int = 128,
            min_new_tokens: int = 1,
            temperature: float = 0.7,
            top_p: float = 0.9,
            num_beams: int = 1,
            do_sample: bool = True,
    ) -> List[str]:
        """
        Generate captions given images using only image embeddings and query tokens.

        Args:
            pixel_values: Image input tensor [batch_size, channels, height, width]
            tokenizer: Tokenizer for decoding
            max_new_tokens: Maximum number of new tokens to generate
            min_new_tokens: Minimum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling or greedy decoding
        """
        # Extract vision features and transform to language model dimension
        vit_embeds = self.extract_feature(pixel_values)
        image_embeds = self.mlp1(vit_embeds)

        batch_size = pixel_values.shape[0]

        # Expand query embeddings
        query_embeds = self.contrastive_query_embeds.expand(batch_size, -1, -1)

        # Concatenate image embeddings with query embeddings
        inputs_embeds = torch.cat([image_embeds, query_embeds], dim=1)

        # Create attention mask
        attention_mask = torch.ones(
            (batch_size, inputs_embeds.shape[1]),
            dtype=torch.long,
            device=inputs_embeds.device
        )

        # Configure generation parameters
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Generate caption tokens
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
        )

        # Decode generated tokens to text
        generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        return generated_texts

    def classify(self, features):
        return self.fc(features)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # if not self.vision_only:
        #     vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
