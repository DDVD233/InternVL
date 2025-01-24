import argparse
import gc
import json
import logging
import math
import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from internvl.model.convnext import ConvNextV2Classifier
from internvl.train.dataset import build_transform
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, hamming_loss, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, Subset
from transformers import get_cosine_schedule_with_warmup
from heavyball import PrecondScheduleForeachSOAP
import traceback

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_frames(video_path: str, num_frames: int, is_train: bool) -> List[Image.Image]:
    """
    Sample frames from a video file, excluding first and last frames for uniform sampling.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample
        is_train: If True, use random sampling; if False, use uniform sampling excluding first/last frames

    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    if is_train:
        # Random sampling for training
        frame_indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))
    else:
        # Sample num_frames + 2 points uniformly, then remove first and last
        frame_indices = np.linspace(0, total_frames - 1, num_frames + 2, dtype=int)[1:-1]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            raise ValueError(f"Failed to read frame {idx} from video: {video_path}")

    cap.release()
    return frames


def collate_fn(batch):
    # Find max length of pixel_values in the batch
    max_length = max(item['pixel_values'].size(0) for item in batch)

    # Pad pixel_values and create attention masks
    padded_pixel_values = []
    attention_masks = []

    for item in batch:
        pixel_vals = item['pixel_values']
        current_len = pixel_vals.size(0)

        # Create padding
        padding_size = max_length - current_len
        padding = torch.zeros((padding_size,) + pixel_vals.size()[1:])

        # Pad pixel values
        padded = torch.cat([pixel_vals, padding], dim=0)
        padded_pixel_values.append(padded)

        # Create attention mask
        mask = torch.ones(current_len, dtype=torch.long)
        mask_padding = torch.zeros(padding_size, dtype=torch.long)
        attention_mask = torch.cat([mask, mask_padding], dim=0)
        attention_masks.append(attention_mask)

    # Stack all padded values and masks
    pixel_values = torch.stack(padded_pixel_values)
    attention_mask = torch.stack(attention_masks)
    labels = torch.stack([torch.tensor(item['labels'], dtype=torch.float32) for item in batch])

    if 'positive' in batch[0]:
        # Handle positive examples similarly with padding
        max_pos_length = max(item['positive'].size(0) for item in batch)
        padded_positive_values = []
        positive_attention_masks = []

        for item in batch:
            pos_vals = item['positive']
            current_len = pos_vals.size(0)

            padding_size = max_pos_length - current_len
            padding = torch.zeros((padding_size,) + pos_vals.size()[1:])

            padded = torch.cat([pos_vals, padding], dim=0)
            padded_positive_values.append(padded)

            mask = torch.ones(current_len, dtype=torch.long)
            mask_padding = torch.zeros(padding_size, dtype=torch.long)
            pos_attention_mask = torch.cat([mask, mask_padding], dim=0)
            positive_attention_masks.append(pos_attention_mask)

        positive_values = torch.stack(padded_positive_values)
        positive_attention_mask = torch.stack(positive_attention_masks)

        # Generate negative indices
        batch_size = len(batch)
        negative_indices = []
        for i in range(batch_size):
            neg_indices = [j for j in range(batch_size) if not torch.all(labels[i] == labels[j])]
            if not neg_indices:  # If no negatives found, use all other indices except self
                neg_indices = [j for j in range(batch_size) if j != i]
            if not neg_indices:  # If still empty (batch size 1 or all same labels)
                neg_indices = [i]  # Use self as negative if no other options
            negative_indices.append(neg_indices)

        max_neg_count = max(len(indices) for indices in negative_indices)
        padded_negative_indices = [
            indices + [indices[0] if indices else i] * (max_neg_count - len(indices))
            for i, indices in enumerate(negative_indices)
        ]
        padded_negative_indices = torch.tensor(padded_negative_indices)
    else:
        positive_values = None
        positive_attention_mask = None
        padded_negative_indices = None

    return {
        'pixel_values': pixel_values,
        'attention_mask': attention_mask,
        'positive': positive_values,
        'positive_attention_mask': positive_attention_mask,
        'labels': labels,
        'negative_indices': padded_negative_indices,
    }


class SingleDatasetLazySupervisedDataset(Dataset):
    def __init__(
            self,
            meta_path,
            dataset_name,
            output_path,
            image_size=384,
            is_train=False,
            pad2square=False,
            normalize_type='imagenet',
            random_seed=0,
    ):
        super(SingleDatasetLazySupervisedDataset, self).__init__()
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.normalize_type = normalize_type

        vocabs = []
        data_items = []

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        if dataset_name not in meta:
            raise ValueError(f"Dataset {dataset_name} not found in metadata")

        ds_meta = meta[dataset_name]
        root = ds_meta['root']

        with open(ds_meta['annotation'], 'r') as f:
            bar = tqdm.tqdm(f, desc=f'Loading {dataset_name}', total=ds_meta['length'])
            for line in f:
                data = json.loads(line)
                if 'images' in data:
                    for index, image in enumerate(data['images']):
                        real_image_path = os.path.join(root, image)
                        data['images'][index] = real_image_path

                target = data['conversations'][1]['value'].lower()
                targets = [label.strip() for label in target.split(',')]
                vocabs.extend(targets)
                data['targets'] = targets
                data['dataset'] = dataset_name
                data_items.append(data)
                bar.update(1)

        vocab_path = os.path.join(output_path, f'{dataset_name}_vocabs.json')
        vocabs = self.build_vocab(vocab_path, vocabs)
        logger.info(f'Vocab size for {dataset_name}: {len(vocabs)}')

        vocabs_to_index = {v: i for i, v in enumerate(vocabs)}
        self.dataset_multilabel = []
        all_labels = []

        for data in data_items:
            labels = [0] * len(vocabs)
            if len(data['targets']) > 1:
                self.dataset_multilabel.append(dataset_name)
            for target in data['targets']:
                if target in vocabs_to_index:
                    labels[vocabs_to_index[target]] = 1
            all_labels.append(labels)
            data['labels'] = labels

        # Calculate pos_weight for the dataset
        agg_labels = torch.tensor(all_labels, dtype=torch.float)
        self.pos_weight = 1 / agg_labels.mean(dim=0).clamp(min=1e-6)

        self.data_items = data_items
        self.vocabs = vocabs
        self.vocabs_to_index = vocabs_to_index

        self.rng = np.random.default_rng(seed=random_seed)
        if is_train:
            self.rng.shuffle(self.data_items)

    def build_vocab(self, vocab_path, vocabs):
        vocabs = list(set(vocabs))
        vocabs.sort()
        with open(vocab_path, 'w') as f:
            json.dump(vocabs, f)
        return vocabs

    def __len__(self):
        return len(self.data_items)

    def load_image(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f'Error loading image {image_path}: {e}')
            return Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

    def get_transform(self):
        return build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type
        )

    def __getitem__(self, i):
        data_item = self.data_items[i]
        transform = self.get_transform()

        images = []
        try:
            if 'images' in data_item and len(data_item['images']) > 0:
                for image_path in data_item['images']:
                    image = self.load_image(image_path)
                    images.append(image)
            elif 'videos' in data_item and len(data_item['videos']) > 0:
                for video_path in data_item['videos']:
                    video = sample_frames(video_path, 1, self.is_train)
                    images.extend(video)
            else:
                images = [torch.zeros((3, self.image_size, self.image_size))]
        except Exception as e:
            logger.error(f"Error processing item {i}: {e}")
            images = [torch.zeros((3, self.image_size, self.image_size))]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        ret = {
            'pixel_values': pixel_values,
            'labels': data_item['labels'],
        }
        return ret


def contrastive_loss(anchor, positive, negative_indices, temperature=0.07):
    batch_size = anchor.size(0)
    sim_positive = F.cosine_similarity(anchor, positive, dim=1) / temperature

    neg_sims = torch.empty(batch_size, negative_indices.size(1), device=anchor.device)
    for i in range(batch_size):
        neg_sims[i] = F.cosine_similarity(
            anchor[i].unsqueeze(0),
            anchor[negative_indices[i]],
            dim=1
        ) / temperature

    all_sims = torch.cat([sim_positive.unsqueeze(1), neg_sims], dim=1)
    losses = -sim_positive + torch.logsumexp(all_sims, dim=1)
    return losses.mean()


def evaluate_classifier(model, dataset, device, split, bs, step, epoch):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc=f'Evaluating {split}'):
            pixel_values = batch['pixel_values'].to(device).to(model.dtype)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            features = model.forward_vision(pixel_values, attention_mask=attention_mask, classify=False)
            outputs = model.classify(features)
            all_outputs.append(outputs.float().cpu())
            all_labels.append(labels.float().cpu())

    outputs = torch.cat(all_outputs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Handle multilabel vs multiclass differently
    is_multilabel = len(dataset.dataset_multilabel) > 0

    if is_multilabel:
        # Convert to numpy first to ensure shape consistency
        labels_np = labels.numpy()
        outputs_np = outputs.numpy()

        # Apply sigmoid and get predictions
        predictions_np = (1 / (1 + np.exp(-outputs_np))) > 0.5

        accuracy = 1 - hamming_loss(labels_np, predictions_np)

        # Handle case where all predictions are 0
        zero_pred_samples = (predictions_np.sum(axis=1) == 0)
        if any(zero_pred_samples):
            highest_prob_indices = outputs_np[zero_pred_samples].argmax(axis=1)
            sample_indices = np.where(zero_pred_samples)[0]
            predictions_np[sample_indices, highest_prob_indices] = True

        try:
            auc = roc_auc_score(labels_np, 1 / (1 + np.exp(-outputs_np)), average='macro')
        except ValueError:
            auc = None

        sensitivity, specificity = calculate_multilabel_metrics(labels_np, predictions_np)
        f1 = f1_score(labels_np, predictions_np, average='macro')

    else:
        outputs = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
        labels = labels.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

        try:
            auc = roc_auc_score(
                torch.nn.functional.one_hot(labels, num_classes=outputs.size(1)).numpy(),
                outputs.numpy(),
                average='macro'
            )
        except ValueError:
            auc = None

        # Convert to one-hot for binary metrics
        y_true_binary = torch.nn.functional.one_hot(labels, num_classes=outputs.size(1)).numpy()
        y_pred_binary = torch.nn.functional.one_hot(predictions, num_classes=outputs.size(1)).numpy()

        sensitivity, specificity = calculate_multilabel_metrics(y_true_binary, y_pred_binary)
        f1 = f1_score(labels, predictions, average='macro')

    stats = {
        f'{split}/auc': auc,
        f'{split}/accuracy': accuracy,
        f'{split}/sensitivity': sensitivity,
        f'{split}/specificity': specificity,
        f'{split}/f1': f1,
        'step': step,
        'epoch': epoch
    }

    wandb.log(stats)
    logger.info(f"Evaluation results for {split}: {stats}")

    model.train()
    return stats


def calculate_multilabel_metrics(y_true, y_pred):
    n_classes = y_true.shape[1]
    sensitivities = []
    specificities = []

    for i in range(n_classes):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return np.mean(sensitivities), np.mean(specificities)


def train_single_dataset(
        model_path: str,
        dataset_name: str,
        output_path: str,
        meta_train_path: str,
        meta_valid_path: str,
        lr: float = 1.5e-4,
        bs: int = 32,
        wd: float = 0.01,
        epochs: int = 3,
        contrastive_weight: float = 0.5,
        eval_every: int = 1000,
):
    dataset_output_path = os.path.join(output_path, dataset_name)
    os.makedirs(dataset_output_path, exist_ok=True)

    # Setup logging for this dataset
    file_handler = logging.FileHandler(os.path.join(dataset_output_path, 'train.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Initialize wandb for this dataset
    wandb.init(
        project='single_modality_training',
        name=dataset_name,
        config={
            'model_path': model_path,
            'dataset': dataset_name,
            'learning_rate': lr,
            'batch_size': bs,
            'weight_decay': wd,
            'epochs': epochs,
            'contrastive_weight': contrastive_weight,
        }
    )

    logger.info(f'Starting training for dataset: {dataset_name}')
    logger.info(f'Output path: {dataset_output_path}')

    # Initialize datasets
    train_dataset = SingleDatasetLazySupervisedDataset(
        meta_train_path,
        dataset_name,
        dataset_output_path,
        is_train=True,
    )

    val_dataset = SingleDatasetLazySupervisedDataset(
        meta_valid_path,
        dataset_name,
        dataset_output_path,
        is_train=False,
    )

    # Initialize model
    model = ConvNextV2Classifier.from_pretrained(
        model_path,
        vision_output_size=len(train_dataset.vocabs)
    ).cuda()
    model = model.to(torch.bfloat16)
    model.train()

    # Initialize dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Calculate total steps and warmup steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = min(500, total_steps // 10)

    # Initialize optimizer and scheduler
    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params}')
    logger.info(f'Total steps: {total_steps}, Warmup steps: {warmup_steps}')

    # Training loop
    best_val_accuracy = 0
    step = 0
    device = torch.device('cuda')

    for epoch in range(epochs):
        for batch in train_dataloader:
            pixel_values = batch['pixel_values'].to(device).to(model.dtype)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass for main classification
            features = model.forward_vision(pixel_values, attention_mask=attention_mask, classify=False)
            outputs = model.classify(features)
            classification_loss = F.binary_cross_entropy_with_logits(
                outputs,
                labels.to(model.dtype),
                pos_weight=train_dataset.pos_weight.to(device).to(model.dtype)
            )

            # Contrastive learning if positive examples are provided
            if batch['positive'] is not None:
                positive_values = batch['positive'].to(device).to(model.dtype)
                positive_attention_mask = batch['positive_attention_mask'].to(device)
                negative_indices = batch['negative_indices'].to(device)

                positive_features = model.forward_vision(positive_values, attention_mask=positive_attention_mask,
                                                         classify=False)
                contr_loss = contrastive_loss(features, positive_features, negative_indices)
                total_loss = classification_loss + contrastive_weight * contr_loss
            else:
                contr_loss = torch.tensor(0.0, device=device)
                total_loss = classification_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                stats = {
                    'train/loss': total_loss.item(),
                    'train/classification_loss': classification_loss.item(),
                    'train/contrastive_loss': contr_loss.item(),
                    'train/learning_rate': lr,
                    'epoch': epoch,
                    'step': step,
                }
                wandb.log(stats)
                logger.info(f'Step {step}: {stats}')

            # Evaluation
            if step % eval_every == 0:
                # Evaluate on validation set
                val_stats = evaluate_classifier(
                    model,
                    val_dataset,
                    device,
                    'val',
                    bs,
                    step,
                    epoch
                )

                # Save model if validation accuracy improved
                if val_stats['val/accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_stats['val/accuracy']
                    torch.save(
                        model.state_dict(),
                        os.path.join(dataset_output_path, 'best_model.pt')
                    )
                    logger.info(f'Saved new best model with validation accuracy: {best_val_accuracy}')

            step += 1

    # Final evaluation
    final_val_stats = evaluate_classifier(
        model,
        val_dataset,
        device,
        'val',
        bs,
        step,
        epoch
    )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(dataset_output_path, 'final_model.pt')
    )

    # Close wandb run
    wandb.finish()

    # Remove file handler
    logger.removeHandler(file_handler)

    return final_val_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='facebook/convnextv2-large-1k-224')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--meta_train_path', type=str, required=True)
    parser.add_argument('--meta_valid_path', type=str, required=True)
    parser.add_argument('--eval_every', type=int, default=1000)
    args = parser.parse_args()

    # Create main output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load existing results if they exist
    results_path = os.path.join(args.output_path, 'all_results.json')
    if os.path.exists(results_path):
        logger.info("Loading existing results from all_results.json")
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            all_results = existing_results.get('dataset_results', {})
            logger.info(f"Found results for {len(all_results)} datasets")
        except json.JSONDecodeError:
            logger.warning("Error loading existing results, starting fresh")
            all_results = {}
    else:
        all_results = {}

    # Get list of datasets from meta files
    with open(args.meta_train_path, 'r') as f:
        meta_train = json.load(f)
    dataset_names = list(meta_train.keys())

    # Train on each dataset separately
    for dataset_name in dataset_names:
        # Skip if already processed
        if dataset_name in all_results:
            logger.info(f'Skipping dataset {dataset_name} - already processed')
            continue

        logger.info(f'\n{"=" * 50}\nTraining on dataset: {dataset_name}\n{"=" * 50}')

        modality_averages = {}
        try:
            results = train_single_dataset(
                model_path=args.model_path,
                dataset_name=dataset_name,
                output_path=args.output_path,
                meta_train_path=args.meta_train_path,
                meta_valid_path=args.meta_valid_path,
                lr=args.lr,
                bs=args.bs,
                wd=args.wd,
                epochs=args.epochs,
                eval_every=args.eval_every,
            )
            all_results[dataset_name] = results

            # Calculate and save intermediate results after each dataset
            modality_results = defaultdict(list)
            modality_metrics = defaultdict(lambda: defaultdict(list))

            for ds_name, res in all_results.items():
                modality = ds_name.split('_')[0]  # Get modality from dataset name
                modality_results[modality].append((ds_name, res))

                # Collect metrics for each modality
                for metric, value in res.items():
                    if isinstance(value, float):
                        modality_metrics[modality][metric].append(value)

            # Calculate averages for each modality
            for modality, metrics in modality_metrics.items():
                modality_averages[modality] = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }

            # Save intermediate results
            final_results = {
                'dataset_results': all_results,
                'modality_averages': modality_averages
            }

            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2)

            logger.info(f'Saved intermediate results after completing dataset {dataset_name}')

        except Exception as e:
            logger.error(f'Error training on dataset {dataset_name}: {str(e)}')
            logger.error(traceback.format_exc())

            # Save results even if there's an error
            final_results = {
                'dataset_results': all_results,
                'modality_averages': modality_averages
            }
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2)

            continue

    # Print summary of results
    logger.info('\n\nTraining Complete!\n')

    # Print individual dataset results
    logger.info('Dataset Results:')
    logger.info('=' * 50)
    for dataset_name, results in all_results.items():
        logger.info(f'\nDataset: {dataset_name}')
        for metric, value in results.items():
            if isinstance(value, float):
                logger.info(f'{metric}: {value:.4f}')

    # Print modality averages
    logger.info('\n\nModality Averages:')
    logger.info('=' * 50)
    for modality, averages in modality_averages.items():
        logger.info(f'\nModality: {modality}')
        logger.info(f'Number of datasets: {len(modality_metrics[modality]["val/accuracy"])}')
        for metric, value in averages.items():
            logger.info(f'{metric}: {value:.4f}')


if __name__ == '__main__':
    main()