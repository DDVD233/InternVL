import argparse
import os
import sys

from internvl.model.eva import DropPath
from internvl.train.pretrain_dataset import PretrainingDataset
from internvl.model.convnext import ConvNextV2Classifier

import torch
import torch.nn as nn
import logging
import wandb
from typing import Any, Dict, List
import math
from tqdm import tqdm
from heavyball import PrecondScheduleForeachSOAP
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder wrapper for ConvNextV2Classifier"""

    def __init__(
            self,
            backbone: ConvNextV2Classifier,
            decoder_depth: int = 1,
            decoder_embed_dim: int = 512,
            patch_size: int = 32,
            mask_ratio: float = 0.6,
            norm_pix_loss: bool = False
    ):
        super().__init__()
        self.encoder = backbone
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Get encoder's output dimension
        self.encoder_dim = self.encoder.vision_model.config.hidden_sizes[-1]

        # Projection layer
        self.proj = nn.Conv2d(
            in_channels=self.encoder_dim,
            out_channels=decoder_embed_dim,
            kernel_size=1
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))

        # Decoder blocks
        decoder = []
        for i in range(decoder_depth):
            decoder.append(
                ConvNextBlock(
                    dim=decoder_embed_dim,
                    drop_path=0.
                )
            )
        self.decoder = nn.Sequential(*decoder)

        # Prediction head
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * 3,  # Reconstruct RGB patches
            kernel_size=1
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize decoder weights
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize projection and prediction layers
        for m in [self.proj, self.pred]:
            if isinstance(m, nn.Conv2d):
                w = m.weight.data
                torch.nn.init.trunc_normal_(w.view([w.shape[0], -1]), std=.02)
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches
        Args:
            imgs: (N, 3, H, W)
        Returns:
            x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images
        Args:
            x: (N, L, patch_size**2 *3)
        Returns:
            imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def generate_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Generate random mask
        Args:
            x: (N, 3, H, W)
            mask_ratio: percentage of patches to mask
        Returns:
            mask: (N, L), 0 is keep, 1 is remove
        """
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_encoder(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder with masking
        Args:
            imgs: (N, 3, H, W)
        Returns:
            x: encoded features
            mask: binary mask
        """
        # Generate mask
        mask = self.generate_mask(imgs, self.mask_ratio)

        # Get features from backbone
        features = self.encoder.vision_model(imgs).last_hidden_state

        return features, mask

    def forward_decoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder
        Args:
            x: encoded features
            mask: binary mask
        Returns:
            pred: reconstructed patches
        """
        # Project features
        x = self.proj(x)

        # Apply mask token
        B, C, H, W = x.shape
        mask = mask.reshape(B, H, W).unsqueeze(1)
        mask_tokens = self.mask_token.expand(B, -1, H, W)
        x = x * (1. - mask) + mask_tokens * mask

        # Decode
        x = self.decoder(x)
        pred = self.pred(x)

        return pred

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss
        Args:
            imgs: original images
            pred: reconstructed patches
            mask: binary mask
        Returns:
            loss: reconstruction loss
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass
        Args:
            imgs: (N, 3, H, W)
            mask_ratio: optional override for default mask ratio
        Returns:
            loss: reconstruction loss
            pred: reconstructed patches
            mask: binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        x, mask = self.forward_encoder(imgs)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


class ConvNextBlock(nn.Module):
    """Simple ConvNeXt block for decoder"""

    def __init__(self, dim: int, drop_path: float = 0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


def train_one_epoch(
        model: FCMAE,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        epoch: int,
        max_norm: float = 1.0,
) -> Dict[str, float]:
    """Training loop for one epoch with wandb logging using bfloat16"""
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # Get pixel values and convert to bfloat16
        images = batch['pixel_values'].to(device).to(torch.bfloat16)

        # Forward pass in bfloat16
        loss, _, _ = model(images)

        # Check for invalid loss
        if not math.isfinite(loss.item()):
            logger.error(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / (batch_idx + 1)

        # Calculate statistics for the current batch
        batch_stats = {
            'loss': loss.item(),
            'avg_loss': avg_loss,
            'lr': current_lr,
            'num_images': batch['num_images'].mean().item(),
            'video_ratio': batch['is_video'].float().mean().item(),
        }

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_stats["loss"]:.4f}',
            'avg_loss': f'{batch_stats["avg_loss"]:.4f}',
            'lr': f'{batch_stats["lr"]:.6f}',
            'imgs/batch': f'{batch_stats["num_images"]:.1f}'
        })

        # Log to wandb with additional metrics
        wandb.log({
            'train/loss': batch_stats["loss"],
            'train/avg_loss': batch_stats["avg_loss"],
            'train/learning_rate': batch_stats["lr"],
            'train/epoch': epoch,
            'train/step': epoch * num_batches + batch_idx,
            'train/images_per_batch': batch_stats["num_images"],
            'train/video_ratio': batch_stats["video_ratio"],
        })

        # Log dataset distribution (once per epoch)
        if batch_idx == 0:
            dataset_counts = {}
            for ds_name in batch['dataset']:
                if ds_name not in dataset_counts:
                    dataset_counts[ds_name] = 0
                dataset_counts[ds_name] += 1

            wandb.log({
                f'train/dataset_{k}': v / len(batch['dataset'])
                for k, v in dataset_counts.items()
            })

    # Compute epoch metrics
    avg_loss = total_loss / num_batches

    # Log epoch metrics
    wandb.log({
        'train/epoch_loss': avg_loss,
        'train/epoch': epoch
    })

    return {'loss': avg_loss}


def train_model(
        model: FCMAE,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        num_epochs: int,
        save_dir: str,
        project_name: str = "fcmae_pretrain",
):
    """Full training procedure with wandb logging using bfloat16"""

    # Initialize wandb
    wandb.init(project=project_name)
    wandb.config.update({
        "model_type": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size,
    })

    # Convert model to bfloat16 and move to device
    model = model.to(device).to(torch.bfloat16)

    # Training loop
    for epoch in range(num_epochs):
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
        )

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{save_dir}/checkpoint_epoch{epoch}.pt")
        # remove previous checkpoint
        if epoch > 0:
            prev_checkpoint = f"{save_dir}/checkpoint_epoch{epoch - 1}.pt"
            if os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_train_path', type=str, default='../../../processing/meta_pretrain_local.json',
                        help='Path to training metadata')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default='facebook/convnextv2-base-22k-224',
                        help='Path to ConvNeXtV2 model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output_path', type=str, default='/home/dvd/data/outputs/convnext_pretrain',
                        help='Path to save checkpoints')

    args = parser.parse_args()

    if "convnextv2-base" in args.model_path:
        img_size = 224
    elif "convnextv2-large" in args.model_path:
        img_size = 384
    else:
        raise ValueError("Invalid model path")

    dataset = PretrainingDataset(
        meta_path=args.meta_train_path,
        tokenizer=None,
        image_size=img_size,
        max_length=77,
        frames_per_video=4,
        is_train=True
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=16
    )

    # Initialize models and optimizer
    backbone = ConvNextV2Classifier.from_pretrained(args.model_path)
    model = FCMAE(
        backbone=backbone,
        decoder_depth=1,
        decoder_embed_dim=512,
        patch_size=32,
        mask_ratio=0.6
    )

    # Setup optimizer and scheduler
    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Can use any PyTorch scheduler, for example:
    warmup_steps = 500
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(loader) * args.num_epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train model
    train_model(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.output_path,
        project_name="fcmae_pretrain"
    )
