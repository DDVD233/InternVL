import argparse
import json
import logging
import os
from typing import Dict, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from torchvision import transforms

from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from internvl.train.pretrain_utils import get_2d_sincos_pos_embed, patchify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            contrastive_weight: float = 0.5
    ):
        super().__init__()
        self.encoder = base_model.vision_model
        self.text_encoder = base_model.text_model
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

        # MAE decoder
        encoder_embed_dim = self.encoder.config.hidden_size
        self.decoder = MAEDecoder(
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )

        # Projection heads for contrastive learning
        self.image_projection = nn.Linear(encoder_embed_dim, 512)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, 512)

        # Initialize projection heads
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use [CLS] token embedding
        text_embeds = text_outputs.last_hidden_state[:, 0]
        text_embeds = self.text_projection(text_embeds)
        return F.normalize(text_embeds, dim=-1)

    def encode_image(self, pixel_values):
        # Get vision encoder output (CLS token)
        vision_outputs = self.encoder(
            pixel_values,
            output_hidden_states=True
        )
        image_embeds = vision_outputs.last_hidden_state[:, 0]
        image_embeds = self.image_projection(image_embeds)
        return F.normalize(image_embeds, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        batch_size = pixel_values.shape[0]

        # MAE forward pass
        x = self.encoder(pixel_values, output_hidden_states=True).last_hidden_state[:, 1:, :]
        x, self.ids_restore = self.random_masking(x, self.mask_ratio)
        mae_pred = self.decoder(x, self.ids_restore)

        # Contrastive learning forward pass
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        return mae_pred, image_embeds, text_embeds

    def get_contrastive_loss(self, image_embeds, text_embeds):
        # Cosine similarity as logits
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature

        # Symmetric loss
        labels = torch.arange(len(image_embeds), device=image_embeds.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2

    def get_loss(self, mae_pred, pixel_values, image_embeds, text_embeds):
        # MAE reconstruction loss
        target = patchify(pixel_values)
        mae_loss = F.mse_loss(mae_pred, target)

        # Contrastive loss
        contrastive_loss = self.get_contrastive_loss(image_embeds, text_embeds)

        # Combined loss
        total_loss = (1 - self.contrastive_weight) * mae_loss + self.contrastive_weight * contrastive_loss

        return total_loss, mae_loss, contrastive_loss

    def random_masking(self, x, mask_ratio):
        """Keep the random_masking method from the original MAEPretrainingWrapper"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, ids_restore


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
                    self.captions.append(caption)

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

        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'pixel_values': grid,
            'input_ids': tokenized.input_ids.squeeze(0),
            'attention_mask': tokenized.attention_mask.squeeze(0),
            'num_images': len(images),
            'is_video': is_video
        }


def train(
        model_path: str,
        output_path: str,
        meta_train_path: str,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        batch_size: int = 64,
        epochs: int = 100,
        warmup_epochs: int = 10,
        mask_ratio: float = 0.75,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5
):
    # Initialize wandb
    wandb.init(project='pretraining')

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load base model
    base_model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Create MAE + Contrastive model
    model = MAEContrastiveWrapper(
        base_model=base_model,
        mask_ratio=mask_ratio,
        temperature=temperature,
        contrastive_weight=contrastive_weight
    )
    model = model.to(torch.bfloat16)
    model = model.cuda()

    # Create tokenizer for text encoding
    tokenizer = base_model.text_tokenizer

    # Create dataset and dataloader
    dataset = PretrainingDataset(meta_train_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)  # Custom collate function needed
    )

    # Optimizer and scheduler setup remains the same
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay
    )

    iters_per_epoch = len(dataloader)
    warmup_iters = warmup_epochs * iters_per_epoch
    total_iters = epochs * iters_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_iters,
        pct_start=warmup_iters / total_iters,
        anneal_strategy='cos'
    )

    # Training loop
    step = 0
    for epoch in range(epochs):
        model.train()

        for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
            pixel_values = batch['pixel_values'].cuda().to(torch.bfloat16)
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            # Forward pass
            mae_pred, image_embeds, text_embeds = model(pixel_values, input_ids, attention_mask)
            total_loss, mae_loss, contrastive_loss = model.get_loss(
                mae_pred, pixel_values, image_embeds, text_embeds
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log metrics
            stats = {
                'total_loss': total_loss.item(),
                'mae_loss': mae_loss.item(),
                'contrastive_loss': contrastive_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'step': step,
                'epoch': epoch
            }
            wandb.log(stats)

            if step % 100 == 0:
                logger.info(f'Step {step}: {stats}')

            # Checkpoint saving logic remains the same
            if step % 1000 == 0:
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step,
                    },
                    os.path.join(output_path, f'checkpoint_{step}.pt')
                )

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
        'attention_mask': tokenized.attention_mask
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Number of warmup epochs')

    # Data parameters
    parser.add_argument('--meta_train_path', type=str, default='../../../processing/meta_train.json',
                        help='Path to training metadata')
    parser.add_argument('--meta_valid_path', type=str, default='../../../processing/meta_valid.json',
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        mask_ratio=args.mask_ratio
    )