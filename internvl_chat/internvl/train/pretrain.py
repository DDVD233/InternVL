import argparse
import json
import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
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


class MAEPretrainingWrapper(nn.Module):
    def __init__(
            self,
            base_model: torch.nn.modules,
            mask_ratio: float = 0.75,
            decoder_embed_dim: int = 512,
    ):
        super().__init__()
        self.encoder = base_model.vision_model
        self.mask_ratio = mask_ratio

        encoder_embed_dim = self.encoder.config.hidden_size
        self.decoder = MAEDecoder(
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )

    def random_masking(
            self,
            x: torch.Tensor,
            mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # Get patch embeddings (exclude CLS token)
        x = self.encoder(x, output_hidden_states=False).last_hidden_state[:, 1:, :]

        # Add random mask
        x, self.ids_restore = self.random_masking(x, self.mask_ratio)
        return x

    def forward_decoder(self, x: torch.Tensor) -> torch.Tensor:
        # Forward decoder
        x = self.decoder(x, self.ids_restore)
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        latent = self.forward_encoder(pixel_values)
        pred = self.forward_decoder(latent)
        return pred

    def get_loss(self, pred: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss between predicted and target patches
        """
        target = patchify(pixel_values)
        loss = F.mse_loss(pred, target)
        return loss


class PretrainingDataset(Dataset):
    def __init__(
            self,
            meta_path: str,
            image_size: int = 448,
            is_train: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.is_train = is_train

        # Load metadata
        self.images = []
        self.captions = []
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        for ds_name, ds_meta in meta.items():
            root = ds_meta['root']
            with open(ds_meta['annotation'], 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'images' in data:
                        for image in data['images']:
                            self.images.append(os.path.join(root, image))
                    target = data['conversations'][1]['value'].lower()
                    modality = ds_name.split('_')[0]  # chest, derm, mammo, etc
                    modality = modality.replace('chest', 'chest x-ray').replace('mammo', 'mammography').replace('derm', 'dermoscopy')
                    modality = modality.replace('mri', 'MRI').replace('ct', 'CT')
                    caption = modality + ', ' + target
                    self.captions.append(caption)

        self.unique_captions = list(set(self.captions))

        logger.info(f'Loaded {len(self.images)} images for pretraining')
        self.transform = build_transform(
            is_train=is_train,
            input_size=image_size
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f'Error loading image {image_path}: {e}')
            image = torch.zeros((3, self.image_size, self.image_size))

        return {'pixel_values': image, 'caption': self.captions[idx]}


def train(
        model_path: str,
        output_path: str,
        meta_train_path: str,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        batch_size: int = 64,
        epochs: int = 100,
        warmup_epochs: int = 10,
        mask_ratio: float = 0.75
):
    # Initialize wandb
    wandb.init(project='mae_pretraining')

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load base model
    base_model = InternVLChatModel.from_pretrained(
        model_path,
        vision_only=True,
        vision_output_size=1,  # Dummy value since we don't use classification head
        torch_dtype=torch.bfloat16
    )

    # Create MAE model
    model = MAEPretrainingWrapper(
        base_model=base_model,
        mask_ratio=mask_ratio
    )
    model = model.to(torch.bfloat16)
    model = model.cuda()

    # Create dataset and dataloader
    dataset = PretrainingDataset(meta_train_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )

    # Create optimizer
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
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

            # Forward pass
            pred = model(pixel_values)
            loss = model.get_loss(pred, pixel_values)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log metrics
            stats = {
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'step': step,
                'epoch': epoch
            }
            wandb.log(stats)

            if step % 100 == 0:
                logger.info(f'Step {step}: {stats}')

            # Save checkpoint
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

                # only keep the last 5 checkpoints
                checkpoints = sorted([f for f in os.listdir(output_path) if f.startswith('checkpoint_')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
                for f in checkpoints[:-5]:
                    os.remove(os.path.join(output_path, f))

            step += 1

        # Save epoch checkpoint
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'step': step,
            },
            os.path.join(output_path, f'checkpoint_epoch_{epoch}.pt')
        )

        # Visualization of reconstruction (once per epoch)
        model.eval()
        with torch.no_grad():
            # Get a small batch for visualization
            vis_batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
            pixel_values = vis_batch['pixel_values'].cuda()

            # Get reconstruction
            pred = model(pixel_values)

            # Convert predictions and targets back to images
            pred = unpatchify(pred, pixel_values.shape[2])

            # Log images to wandb
            wandb.log({
                'reconstructions': [
                    wandb.Image(
                        torch.cat([
                            pixel_values[i].cpu(),
                            pred[i].cpu()
                        ], dim=2),
                        caption=f'Original vs Reconstruction'
                    )
                    for i in range(min(8, len(pred)))
                ],
                'epoch': epoch
            })


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