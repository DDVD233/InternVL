import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
import torch.distributed as dist
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from internvl.model.clip import OpenCLIPClassifier
from internvl.model.convnext import ConvNextV2Classifier
from internvl.model.eva_classifier import EVA02Classifier
from internvl.model.sbb_vit import ViTSBBClassifier
from transformers import CLIPTextModel, CLIPTokenizer

from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel, AttentionPoolingBlock
from internvl.train.dataset import build_transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from functools import partial
from internvl.train.soap import SOAP
from internvl.train.sf_soap import SFPaLMForeachSOAP

from internvl.train.pretrain_utils import get_2d_sincos_pos_embed, patchify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return torch.stack(output, 0)

    @staticmethod
    def backward(ctx, grads):
        input, = ctx.saved_tensors
        dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class MAEDecoder(nn.Module):
    def __init__(
            self,
            encoder_embed_dim: int,
            decoder_embed_dim: int,
            image_size: int = 448,
            patch_size: int = 14,
            decoder_depth: int = 8,
            decoder_num_heads: int = 16,
            mlp_ratio: float = 4.0,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding will be added during forward pass
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ) for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.decoder_pos_embed.shape[1] ** 0.5),  # square root of number of patches
            cls_token=False
        )
        self.decoder_pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x


class MAEContrastiveWrapper(nn.Module):
    def __init__(
            self,
            base_model: torch.nn.Module,
            mask_ratio: float = 0.75,
            decoder_embed_dim: int = 512,
            temperature: float = 0.07,
            contrastive_weight: float = 0.5,
            clip_text_model: str = "openai/clip-vit-large-patch14",
            label_smoothing: float = 0.1,
            device: str = "cuda"
    ):
        super().__init__()
        self.encoder = base_model.vision_model
        self.base_model = base_model.to(device)
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.label_smoothing = label_smoothing
        self.device = device

        # Initialize log temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        # Try to get text model from base model, otherwise initialize CLIP
        if hasattr(base_model, 'text_model'):
            logger.info("Using text encoder from base model")
            self.text_encoder = base_model.text_model
            self.tokenizer = base_model.text_tokenizer
        else:
            logger.info(f"Base model has no text encoder, initializing CLIP text encoder from {clip_text_model}")
            self.text_encoder = CLIPTextModel.from_pretrained(clip_text_model)
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_text_model)

        # MAE decoder
        encoder_embed_dim = self.encoder.config.hidden_size
        self.decoder = MAEDecoder(
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )

        # Determine output dimension by running a sample forward pass
        with torch.no_grad():
            # Create a blank image tensor with correct dimensions
            dummy_image = torch.zeros(1, 1, 3,
                                      self.encoder.config.image_size,
                                      self.encoder.config.image_size)
            # Move to same device and dtype as encoder
            dummy_image = dummy_image.to(
                device=device,
                dtype=next(self.encoder.parameters()).dtype
            )
            # Get output size from image encoder
            out_dim = self.base_model.forward_vision(dummy_image, classify=False).shape[-1]
            logger.info(f"Detected vision encoder output dimension: {out_dim}")

        # Text projection with matched output dimension
        self.text_projection = AttentionPoolingBlock(
            dim=self.text_encoder.config.hidden_size,
            num_heads=8,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-5),
            out_dim=out_dim  # Use detected dimension
        )

        # Initialize weights
        self.apply(self._init_weights)
        self._init_attention_pool()

    def encode_text(self, input_ids, attention_mask):
        """Text encoding with proper handling of both base model and CLIP text encoders"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Get all hidden states instead of just [CLS]
        hidden_states = text_outputs.last_hidden_state

        # Apply attention pooling
        text_embeds = self.text_projection(hidden_states)

        return F.normalize(text_embeds, dim=-1)

    def encode_image(self, pixel_values):
        """Image encoding with output normalization
        Args:
            pixel_values: [b, n, c, h, w] image tensor
            """
        if len(pixel_values.shape) == 4:  # this is of shape b, c, h, w
            pixel_values = pixel_values.unsqueeze(1)  # add n dim
        # Get vision encoder output (CLS token)
        image_embeds = self.base_model.forward_vision(
            pixel_values,
            classify=False
        )
        return F.normalize(image_embeds, dim=-1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_attention_pool(self):
        """Initialize the attention pooling parameters."""
        for name, param in self.text_projection.named_parameters():
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, ids_restore

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Forward pass with support for negative samples
        """
        # MAE forward pass
        x = self.encoder(pixel_values, output_hidden_states=True).last_hidden_state[:, 1:, :]
        x, self.ids_restore = self.random_masking(x, self.mask_ratio)
        mae_pred = self.decoder(x, self.ids_restore)

        # Contrastive learning forward pass
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        return mae_pred, image_embeds, text_embeds

    def get_contrastive_loss(self, image_embeds, text_embeds):
        """
        Compute contrastive loss using cross entropy with index-based targets.
        Args:
            image_embeds: [batch_size, embed_dim]
            text_embeds: [batch_size, embed_dim]
        """
        # Get batch size and device
        batch_size = image_embeds.size(0)
        device = image_embeds.device

        # Get current rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Gather embeddings from all devices if using distributed training
        if dist.is_initialized():
            image_embeds_all = torch.cat(GatherLayer.apply(image_embeds), dim=0)
            text_embeds_all = torch.cat(GatherLayer.apply(text_embeds), dim=0)
        else:
            image_embeds_all = image_embeds
            text_embeds_all = text_embeds

        # Compute similarity matrices with temperature scaling
        logit_scale = self.logit_scale.exp()
        sim_i2t = logit_scale * (image_embeds @ text_embeds_all.t())
        sim_t2i = logit_scale * (text_embeds @ image_embeds_all.t())

        # Create targets - indices along the diagonal
        targets = torch.linspace(rank * batch_size,
                                 rank * batch_size + batch_size - 1,
                                 batch_size,
                                 dtype=torch.long,
                                 device=device)

        # Compute symmetric cross entropy loss with label smoothing
        loss_i2t = F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing)

        return (loss_i2t + loss_t2i) / 2

    def get_loss(self, mae_pred, pixel_values, image_embeds, text_embeds):
        # MAE reconstruction loss
        target = patchify(pixel_values)
        mae_loss = F.mse_loss(mae_pred, target)

        # Simplified contrastive loss
        contrastive_loss = self.get_contrastive_loss(image_embeds, text_embeds)

        # Combined loss
        total_loss = (1 - self.contrastive_weight) * mae_loss + self.contrastive_weight * contrastive_loss

        return total_loss, mae_loss, contrastive_loss


def calculate_multilabel_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate sensitivity (recall) and specificity for multilabel classification.

    Parameters:
    y_true: numpy array of shape (n_samples, n_classes) with true binary labels
    y_pred: numpy array of shape (n_samples, n_classes) with predicted probabilities
    threshold: float, classification threshold (default 0.5)

    Returns:
    sensitivity, specificity (macro-averaged across classes)
    """
    n_classes = y_true.shape[1]

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Initialize metrics storage
    sensitivities = []
    specificities = []

    # Calculate metrics for each class
    for i in range(n_classes):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_binary[:, i]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # Calculate macro-averaged metrics
    sensitivity = np.mean(sensitivities)
    specificity = np.mean(specificities)

    return sensitivity, specificity


def evaluate_zero_shot(
        model,
        dataset,
        device,
        split: str,
        batch_size: int,
        step: int,
        epoch: int,
        num_samples: int = -1,
        dataloader: Optional[DataLoader] = None
) -> Dict:
    """
    Zero-shot evaluation using contrastive text-image matching.
    """
    model.eval()
    dataset_outputs = defaultdict(list)
    dataset_labels = defaultdict(list)
    dataset_gt_similarities = defaultdict(list)

    # First, encode all unique captions
    unique_captions = dataset.unique_captions
    caption_to_idx = {caption: idx for idx, caption in enumerate(unique_captions)}

    all_text_features = []
    logger.info("Encoding all unique captions...")
    for i in range(0, len(unique_captions), batch_size):
        batch_captions = unique_captions[i:i + batch_size]
        tokenized = model.tokenizer(
            batch_captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_features = model.encode_text(
                tokenized.input_ids,
                tokenized.attention_mask
            )
        all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features, dim=0)

    # Create dataloader if not provided
    if dataloader is None:
        if len(dataset) > num_samples > 0:
            subset_indices = random.sample(range(len(dataset)), num_samples)
            subset = Subset(dataset, subset_indices)
        else:
            subset = dataset

        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True
        )

    # Evaluate images
    with torch.no_grad():
        num_batches = len(dataloader)
        if num_samples > 0:
            num_batches = math.ceil(num_samples / batch_size)

        bar = tqdm.tqdm(dataloader, desc=f'Evaluating {split}', total=num_batches)

        batch: dict
        for index, batch in enumerate(bar):
            pixel_values = batch['pixel_values'].to(device)
            datasets = batch['dataset']
            captions: List[str] = batch.get('caption', [])
            if len(captions) == 0:
                logger.warning("No captions found in batch, skipping...")
                continue
            labels = torch.zeros(len(pixel_values), len(unique_captions), device=device)
            for i, cap in enumerate(captions):
                labels[i, caption_to_idx[cap]] = 1

            # Get image embeddings
            image_features = model.encode_image(pixel_values)

            # Calculate similarity scores with all captions
            similarity = image_features @ all_text_features.T

            # Convert to probabilities
            probs = F.softmax(similarity / model.temperature, dim=-1)

            # Calculate ground truth similarities if captions are available
            gt_indices = torch.tensor([caption_to_idx[cap] for cap in captions], device=device)
            gt_text_features = all_text_features[gt_indices]
            gt_similarities = F.cosine_similarity(image_features, gt_text_features)

            # Store outputs, labels, and ground truth similarities by dataset
            for prob, label, gt_sim, ds in zip(probs, labels, gt_similarities, datasets):
                dataset_outputs[ds].append(prob.float().cpu().numpy())
                dataset_labels[ds].append(label.cpu().numpy())
                dataset_gt_similarities[ds].append(gt_sim.cpu().item())

            if num_samples > 0 and index >= num_batches:
                break

    # Calculate metrics by dataset and modality
    overall_stats = {
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'gt_similarity': []
    }

    by_modality_stats = defaultdict(lambda: defaultdict(list))

    for ds in dataset_outputs.keys():
        ds_outputs = np.stack(dataset_outputs[ds])
        ds_labels = np.stack(dataset_labels[ds])
        ds_gt_similarities = np.array(dataset_gt_similarities[ds])

        # Calculate metrics only for classes that appear in the labels
        valid_classes = np.where((ds_labels.sum(axis=0) > 0) & (ds_labels.sum(axis=0) < len(ds_labels)))[0]

        if len(valid_classes) == 0:
            logger.warning(f"No valid classes found for dataset {ds}")
            continue

        ds_outputs = ds_outputs[:, valid_classes]
        ds_labels = ds_labels[:, valid_classes]

        # Calculate AUC
        try:
            auc = roc_auc_score(ds_labels, ds_outputs, average='weighted')
        except ValueError:
            auc = None

        # Calculate sensitivity and specificity using the correct function
        sensitivity, specificity = calculate_multilabel_metrics(ds_labels, ds_outputs)

        # Calculate mean ground truth similarity
        mean_gt_similarity = np.mean(ds_gt_similarities)

        # Log dataset-specific metrics
        stats = {
            f'{ds}/{split}/zero_shot_auc': auc,
            f'{ds}/{split}/zero_shot_sensitivity': sensitivity,
            f'{ds}/{split}/zero_shot_specificity': specificity,
            f'{ds}/{split}/zero_shot_gt_similarity': mean_gt_similarity,
            'step': step
        }
        wandb.log(stats)
        logger.info(f"Dataset {ds}: {stats}")

        # Accumulate overall stats
        if auc is not None:
            overall_stats['auc'].append(auc)
        overall_stats['sensitivity'].append(sensitivity)
        overall_stats['specificity'].append(specificity)
        overall_stats['gt_similarity'].append(mean_gt_similarity)

        # Accumulate modality stats
        modality = ds.split('_')[0]
        by_modality_stats[modality]['auc'].append(auc)
        by_modality_stats[modality]['sensitivity'].append(sensitivity)
        by_modality_stats[modality]['specificity'].append(specificity)
        by_modality_stats[modality]['gt_similarity'].append(mean_gt_similarity)

    # Log overall statistics
    try:
        overall_auc = np.mean(overall_stats['auc']) if overall_stats['auc'] else None
        overall_sensitivity = np.mean(overall_stats['sensitivity'])
        overall_specificity = np.mean(overall_stats['specificity'])
        overall_gt_similarity = np.mean(overall_stats['gt_similarity'])

        overall_metrics = {
            f'{split}/zero_shot_overall_auc': overall_auc,
            f'{split}/zero_shot_overall_sensitivity': overall_sensitivity,
            f'{split}/zero_shot_overall_specificity': overall_specificity,
            f'{split}/zero_shot_overall_gt_similarity': overall_gt_similarity,
            'step': step,
            'epoch': epoch
        }

        # Add modality-specific metrics
        for modality, stats in by_modality_stats.items():
            modality_auc = np.mean(stats['auc']) if stats['auc'] else None
            modality_sensitivity = np.mean(stats['sensitivity'])
            modality_specificity = np.mean(stats['specificity'])
            modality_gt_similarity = np.mean(stats['gt_similarity'])

            overall_metrics.update({
                f'{split}/{modality}_zero_shot_auc': modality_auc,
                f'{split}/{modality}_zero_shot_sensitivity': modality_sensitivity,
                f'{split}/{modality}_zero_shot_specificity': modality_specificity,
                f'{split}/{modality}_zero_shot_gt_similarity': modality_gt_similarity
            })

        wandb.log(overall_metrics)
        logger.info(f"Overall zero-shot metrics: {overall_metrics}")

    except ValueError:
        logger.error('Error calculating overall statistics')
        return None

    model.train()
    return overall_metrics


class PretrainingDataset(Dataset):
    def __init__(
            self,
            meta_path: str,
            tokenizer: Any,
            image_size: int = 448,
            max_length: int = 77,
            frames_per_video: int = 4,  # Number of frames to extract from each video
            is_train: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.frames_per_video = frames_per_video

        # Load metadata
        self.image_groups = []  # List of lists of image paths or video paths
        self.captions = []
        self.datasets = []  # Track which dataset each entry belongs to
        self.is_video = []  # Track which entries are videos

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        for ds_name, ds_meta in meta.items():
            root = ds_meta['root']
            with open(ds_meta['annotation'], 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'images' in data:
                        this_images = [os.path.join(root, image) for image in data['images']]
                        self.image_groups.append(this_images)
                        self.is_video.append(False)
                    elif 'videos' in data and len(data['videos']) > 0:
                        video_path = os.path.join(root, data['videos'][0])
                        self.image_groups.append([video_path])  # Store as single-item list for consistency
                        self.is_video.append(True)
                    else:
                        continue

                    target = data['conversations'][1]['value'].lower()
                    modality = ds_name.split('_')[0]
                    modality = modality.replace('chest', 'chest x-ray').replace('mammo', 'mammography').replace('derm',
                                                                                                                'dermoscopy')
                    modality = modality.replace('mri', 'MRI').replace('ct', 'CT')
                    caption = modality + ', ' + target
                    caption = caption.replace('_', ' ')
                    self.captions.append(caption)

                    self.datasets.append(ds_name)

        self.unique_captions = list(set(self.captions))

        logger.info(f'Loaded {len(self.image_groups)} image/video groups for pretraining')
        self.transform = build_transform(
            is_train=is_train,
            input_size=image_size
        )

    def extract_video_frames(self, video_path: str) -> List[torch.Tensor]:
        """Extract frames from video and convert to tensors."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)

            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f'Empty video file: {video_path}')
                return [torch.zeros((3, self.image_size, self.image_size))]

            # Calculate frame indices to extract
            indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image and apply transforms
                    frame = Image.fromarray(frame)
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)
                else:
                    logger.error(f'Failed to read frame {idx} from video: {video_path}')
                    frames.append(torch.zeros((3, self.image_size, self.image_size)))

            cap.release()
            return frames

        except Exception as e:
            logger.error(f'Error processing video {video_path}: {e}')
            return [torch.zeros((3, self.image_size, self.image_size))] * self.frames_per_video

    def get_grid_dimensions(self, n_images: int) -> Tuple[int, int]:
        """
        Calculate the optimal grid dimensions for n images.
        Returns (rows, cols) that form the most square-like arrangement.
        """
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        return rows, cols

    def create_image_grid(self, images: List[torch.Tensor], rows: int, cols: int) -> torch.Tensor:
        """
        Create a grid of images with specified dimensions.
        Pads empty spaces with black if necessary.
        """
        n_images = len(images)
        c, h, w = images[0].shape

        # Create empty grid
        grid = torch.zeros((c, h * rows, w * cols))

        for idx, img in enumerate(images):
            i = idx // cols  # row
            j = idx % cols  # column
            grid[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = img

        return grid

    def load_and_transform_image(self, image_path: str) -> torch.Tensor:
        """Load and transform a single image, handling errors."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f'Error loading image {image_path}: {e}')
            return torch.zeros((3, self.image_size, self.image_size))

    def __len__(self) -> int:
        return len(self.image_groups)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        paths = self.image_groups[idx]
        caption = self.captions[idx]
        is_video = self.is_video[idx]

        if is_video:
            # Extract frames from video
            images = self.extract_video_frames(paths[0])
        else:
            # Load and transform all images in the group
            images = [self.load_and_transform_image(path) for path in paths]

        # Get optimal grid dimensions
        rows, cols = self.get_grid_dimensions(len(images))

        # Create image grid
        grid = self.create_image_grid(images, rows, cols)

        # Resize grid to original image size while maintaining aspect ratio
        if grid.shape[1] != self.image_size or grid.shape[2] != self.image_size:
            resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_size, antialias=True),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()
            ])
            grid = resize_transform(grid)

        return {
            'pixel_values': grid,
            'caption': caption,
            'num_images': len(images),
            'is_video': is_video,
            'dataset': self.datasets[idx]
        }


def train(
        model_path: str,
        output_path: str,
        meta_train_path: str,
        meta_val_path: str,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        batch_size: int = 64,
        epochs: int = 100,
        mask_ratio: float = 0.75,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5
):
    # Initialize wandb
    wandb.init(project='pretraining')

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'pretrain.log')
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Load base model
    # base_model = InternVLChatModel.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16
    # )

    if 'sbb2' in model_path.lower():
        base_model = ViTSBBClassifier(vision_output_size=1).cuda()
        # model = model.to(torch.bfloat16)
    elif 'convnext' in model_path.lower():
        base_model = ConvNextV2Classifier.from_pretrained(
            model_path,  # or any other ConvNeXtV2 checkpoint
            vision_output_size=1
        ).cuda()
        base_model = base_model.to(torch.bfloat16)
    elif 'clip' in model_path.lower():
        base_model = OpenCLIPClassifier.from_pretrained(
            model_path,
            vision_output_size=1,
            dtype=torch.bfloat16
        ).cuda()
    elif 'eva' in model_path.lower():
        base_model = EVA02Classifier(
            vision_output_size=1,
            checkpoint_path="eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt"
        ).cuda()
        base_model = base_model.to(torch.bfloat16)
    else:
        base_model = InternVLChatModel.from_pretrained(model_path, vision_only=True, vision_output_size=1,
                                                  torch_dtype=torch.bfloat16)

    # Create MAE + Contrastive model
    model: MAEContrastiveWrapper = MAEContrastiveWrapper(
        base_model=base_model,
        mask_ratio=mask_ratio,
        temperature=temperature,
        contrastive_weight=contrastive_weight
    )
    model = model.to(torch.bfloat16)
    model = model.cuda()

    # Create tokenizer for text encoding
    tokenizer = model.tokenizer

    # Create dataset and dataloader
    dataset = PretrainingDataset(meta_train_path, tokenizer)
    val_dataset = PretrainingDataset(meta_val_path, tokenizer, is_train=False)
    # Optimizer and scheduler setup remains the same
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    ]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=64,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer)
    )

    optimizer = SOAP(param_groups, lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    warmup_iters = 500
    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_iters,
        num_training_steps=total_iters,
        num_cycles=1.5
    )

    # Training loop
    step = 0
    for epoch in range(epochs):
        model.train()

        for batch in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
            pixel_values = batch['pixel_values'].cuda().to(torch.bfloat16)
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            # Forward pass with negative samples
            mae_pred, image_embeds, text_embeds = model(
                pixel_values,
                input_ids,
                attention_mask
            )

            # Compute loss
            total_loss, mae_loss, contrastive_loss = model.get_loss(
                mae_pred,
                pixel_values,
                image_embeds,
                text_embeds,
            )

            # Rest of training loop (backward pass, optimizer step, etc.)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log additional metrics
            current_lr = lr_scheduler.get_last_lr()[0]
            with torch.no_grad():
                stats = {
                    'total_loss': total_loss.item(),
                    'mae_loss': mae_loss.item(),
                    'contrastive_loss': contrastive_loss.item(),
                    'logit_scale': model.logit_scale.exp().item(),
                    'lr': current_lr,
                    'step': step,
                    'epoch': epoch
                }
                wandb.log(stats)

            # Evaluation and checkpointing
            if step % 500 == 0 and step > 0:
                # Run zero-shot evaluation
                evaluate_zero_shot(
                    model,
                    val_dataset,
                    model.device,
                    'val',
                    batch_size,
                    step,
                    epoch,
                    dataloader=val_loader
                )
            if step % 1000 == 0 and step > 0:
                # Save checkpoint
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step,
                    },
                    os.path.join(output_path, f'checkpoint_{step}.pt')
                )
                logger.info(f'Saved checkpoint at step {step}')

            step += 1


def collate_fn(batch, tokenizer):
    """Custom collate function to handle both images and text"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])

    # Tokenize captions
    texts = [item['caption'] for item in batch]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,  # Standard CLIP text length
        return_tensors="pt"
    )

    return {
        'pixel_values': pixel_values,
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask,
        'dataset': [item['dataset'] for item in batch],
        'caption': texts
    }


def unpatchify(x: torch.Tensor, img_size: int) -> torch.Tensor:
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 14  # patch size
    h = w = img_size // p
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_path: str
) -> Tuple[int, int]:
    """Load checkpoint and return current epoch and step."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return checkpoint['epoch'], checkpoint['step']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Pretraining')

    # Model parameters
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL2-8B',
                        help='Path to base model')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Ratio of patches to mask')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')

    # Data parameters
    parser.add_argument('--meta_train_path', type=str, default='../../../processing/meta_pretrain_local.json',
                        help='Path to training metadata')
    parser.add_argument('--meta_valid_path', type=str, default='../../../processing/meta_pretrain_valid_local.json',
                        help='Path to training metadata')

    # Output parameters
    parser.add_argument('--output_path', type=str, default='/home/dvd/data/outputs/internvl_pretrain',
                        help='Path to save checkpoints')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Start training
    if args.resume:
        logger.info(f'Resuming training from checkpoint: {args.resume}')

    train(
        model_path=args.model_path,
        output_path=args.output_path,
        meta_train_path=args.meta_train_path,
        meta_val_path=args.meta_valid_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio
    )