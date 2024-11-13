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
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
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
from heavyball import PrecondScheduleForeachSOAP

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
            tokenizer,
            mask_ratio: float = 0.75,
            decoder_embed_dim: int = 512,
            temperature: float = 0.07,
            contrastive_weight: float = 0.4,
            generative_weight: float = 0.4,
            label_smoothing: float = 0.1,
            device: str = "cuda"
    ):
        super().__init__()
        self.base_model = base_model.to(device)
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.generative_weight = generative_weight
        self.label_smoothing = label_smoothing
        self.device = device
        self.tokenizer = tokenizer

        # Initialize log temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([], dtype=torch.bfloat16) * np.log(1 / temperature))

        # Get encoder from base model
        self.encoder = base_model.vision_model

        # MAE decoder
        encoder_embed_dim = self.encoder.config.hidden_size
        self.decoder = MAEDecoder(
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )
        # init weights for decoder
        self.decoder.apply(self._init_weights)
        self.decoder = self.decoder.to(dtype=torch.bfloat16)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        """Forward pass with MAE, contrastive, and generative learning"""
        # MAE forward pass
        vision_outputs = self.encoder(pixel_values, output_hidden_states=True)
        x = vision_outputs.last_hidden_state[:, 1:, :]
        x, self.ids_restore = self.random_masking(x, self.mask_ratio)
        mae_pred = self.decoder(x, self.ids_restore)

        # Contrastive and generative learning through language model
        image_embeds, text_embeds, gen_loss = self.base_model.forward_contrastive(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return mae_pred, image_embeds, text_embeds, gen_loss

    def get_loss(self, mae_pred, pixel_values, image_embeds, text_embeds, gen_loss):
        # Convert pixel_values to bfloat16 if needed
        if pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        # MAE reconstruction loss
        target = patchify(pixel_values).to(mae_pred.device, dtype=torch.bfloat16)
        mae_loss = F.mse_loss(mae_pred, target)

        # Contrastive loss
        contrastive_loss = self.get_contrastive_loss(image_embeds, text_embeds)

        # Ensure gen_loss is bfloat16
        gen_loss = gen_loss.to(torch.bfloat16)

        # Combined loss with all three components
        total_loss = (
                (1 - self.contrastive_weight - self.generative_weight) * mae_loss +
                self.contrastive_weight * contrastive_loss +
                self.generative_weight * gen_loss
        )

        return total_loss, mae_loss, contrastive_loss, gen_loss

    def get_contrastive_loss(self, image_embeds, text_embeds):
        """Compute contrastive loss with temperature scaling"""
        # Ensure embeddings are in bfloat16
        image_embeds = image_embeds.to(torch.bfloat16)
        text_embeds = text_embeds.to(torch.bfloat16)

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

        # Compute similarity matrices with temperature
        logit_scale = self.logit_scale.exp()
        sim_i2t = logit_scale * (image_embeds @ text_embeds_all.t())
        sim_t2i = logit_scale * (text_embeds @ image_embeds_all.t())

        # Create targets for contrastive learning
        targets = torch.linspace(rank * batch_size,
                                 rank * batch_size + batch_size - 1,
                                 batch_size,
                                 dtype=torch.long,
                                 device=device)

        # Compute symmetric cross entropy loss with label smoothing
        loss_i2t = F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing)

        return (loss_i2t + loss_t2i) / 2


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
        samples_per_caption: int = 64
) -> Dict:
    """
    Zero-shot evaluation using generation-based metrics on a subset of samples.
    Evaluates fixed number of samples per unique caption.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from collections import defaultdict

    # Initialize smoothing function
    smoothie = SmoothingFunction().method3

    model.eval()
    dataset_outputs = defaultdict(list)
    dataset_bleu = defaultdict(list)

    # Sample indices using pre-computed mapping
    selected_indices = []
    for caption, indices in dataset.caption_to_indices.items():
        # Sample min(samples_per_caption, len(indices)) samples for each caption
        num_samples = min(samples_per_caption, len(indices))
        selected_indices.extend(random.sample(indices, num_samples))

    # Create subset dataset
    subset = Subset(dataset, selected_indices)

    # Create dataloader for subset
    eval_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=24,
        pin_memory=True
    )

    # Evaluate images
    with torch.no_grad():
        bar = tqdm.tqdm(eval_loader, desc=f'Evaluating {split}')

        for batch in bar:
            pixel_values = batch['pixel_values'].to(device)
            datasets = batch['dataset']
            ground_truth_captions = batch['caption']

            # Generate captions
            generated_captions = model.base_model.forward_generate(
                pixel_values=pixel_values,
                tokenizer=model.tokenizer,
            )

            # Compare generated captions with ground truth
            for gen_cap, gt_cap, ds in zip(generated_captions, ground_truth_captions, datasets):
                # Clean and normalize both captions
                gen_cap = gen_cap.strip().lower()
                gt_cap = gt_cap.strip().lower()

                # Exact match (1 if matched, 0 if not)
                match = float(gen_cap == gt_cap)

                # Calculate BLEU score
                gen_tokens = word_tokenize(gen_cap)
                gt_tokens = word_tokenize(gt_cap)

                # Calculate BLEU score with smoothing
                try:
                    bleu = sentence_bleu(
                        [gt_tokens],
                        gen_tokens,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie
                    )
                except Exception as e:
                    logger.warning(f"Error calculating BLEU score: {e}")
                    logger.warning(f"Generated: {gen_cap}")
                    logger.warning(f"Ground truth: {gt_cap}")
                    bleu = 0.0

                dataset_outputs[ds].append(match)
                dataset_bleu[ds].append(bleu)

    # Calculate metrics by dataset and modality
    overall_stats = {
        'accuracy': [],
        'bleu': []
    }

    by_modality_stats = defaultdict(lambda: defaultdict(list))

    # Log number of samples evaluated
    logger.info(f"Evaluated {len(selected_indices)} total samples")
    logger.info(f"Number of unique captions: {len(dataset.caption_to_indices)}")
    logger.info(f"Samples per caption: {samples_per_caption}")

    for ds in dataset_outputs.keys():
        ds_outputs = np.array(dataset_outputs[ds])
        ds_bleu = np.array(dataset_bleu[ds])

        # Calculate metrics
        accuracy = np.mean(ds_outputs)
        bleu = np.mean(ds_bleu)

        # Log dataset-specific metrics
        stats = {
            f'{ds}/{split}/zero_shot_accuracy': accuracy,
            f'{ds}/{split}/zero_shot_bleu': bleu,
            f'{ds}/{split}/num_samples': len(ds_outputs),
            'step': step
        }
        wandb.log(stats)
        logger.info(f"Dataset {ds}: {stats}")

        # Accumulate overall stats
        overall_stats['accuracy'].append(accuracy)
        overall_stats['bleu'].append(bleu)

        # Accumulate modality stats
        modality = ds.split('_')[0]
        by_modality_stats[modality]['accuracy'].append(accuracy)
        by_modality_stats[modality]['bleu'].append(bleu)

    # Calculate and log overall metrics
    overall_metrics = {
        f'{split}/zero_shot_overall_accuracy': np.mean(overall_stats['accuracy']),
        f'{split}/zero_shot_overall_bleu': np.mean(overall_stats['bleu']),
        f'{split}/total_samples': len(selected_indices),
        'step': step,
        'epoch': epoch
    }

    # Add modality-specific metrics
    for modality, stats in by_modality_stats.items():
        modality_metrics = {
            f'{split}/{modality}_zero_shot_accuracy': np.mean(stats['accuracy']),
            f'{split}/{modality}_zero_shot_bleu': np.mean(stats['bleu']),
        }
        overall_metrics.update(modality_metrics)

    wandb.log(overall_metrics)
    logger.info(f"Overall zero-shot metrics: {overall_metrics}")

    model.train()
    return overall_metrics


class PretrainingDataset(Dataset):
    def __init__(
            self,
            meta_path: str,
            tokenizer: Any,
            image_size: int = 448,
            max_length: int = 77,
            frames_per_video: int = 4,
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

        # Create caption to indices mapping during initialization
        self.caption_to_indices = defaultdict(list)

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        for ds_name, ds_meta in meta.items():
            root = ds_meta['root']
            with open(ds_meta['annotation'], 'r') as f:
                for idx, line in enumerate(f):
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

                    # Add to caption_to_indices mapping
                    self.caption_to_indices[caption].append(len(self.captions) - 1)

        self.unique_captions = list(set(self.captions))
        logger.info(f'Loaded {len(self.image_groups)} image/video groups for pretraining')
        logger.info(f'Found {len(self.unique_captions)} unique captions')

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
        base_model = InternVLChatModel.from_pretrained(model_path, vision_only=False, vision_output_size=1,
                                                  torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
        tokenizer.tokenizer_path = model_path

    # freeze language model
    for param in base_model.language_model.parameters():
        param.requires_grad = False

    # Create MAE + Contrastive model
    model: MAEContrastiveWrapper = MAEContrastiveWrapper(
        tokenizer=tokenizer,
        base_model=base_model,
        mask_ratio=mask_ratio,
        temperature=temperature,
        contrastive_weight=contrastive_weight
    )
    model = model.to(torch.bfloat16)
    model = model.cuda()

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
        num_workers=32,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer)
    )

    optimizer = PrecondScheduleForeachSOAP(param_groups, lr=lr, weight_decay=weight_decay)
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
            mae_pred, image_embeds, text_embeds, gen_loss = model(
                pixel_values,
                input_ids,
                attention_mask
            )

            # Compute loss
            total_loss, mae_loss, contrastive_loss, gen_loss = model.get_loss(
                mae_pred, batch['pixel_values'], image_embeds, text_embeds, gen_loss
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
                    'gen_loss': gen_loss.item(),
                    'logit_scale': model.logit_scale.exp().item(),
                    'lr': current_lr,
                    'step': step,
                    'epoch': epoch
                }
                wandb.log(stats)

            # Evaluation and checkpointing
            if step % 500 == 0:
                # Run zero-shot evaluation
                evaluate_zero_shot(
                    model,
                    val_dataset,
                    model.device,
                    'val',
                    batch_size,
                    step,
                    epoch
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
    parser.add_argument('--bs', type=int, default=64,
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
        batch_size=args.bs,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio
    )