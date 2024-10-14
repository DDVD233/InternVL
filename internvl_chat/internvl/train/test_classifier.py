import argparse
from typing import Dict, List

import wandb
from torch import nn

from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
import gc
import json
import math
import os
import warnings
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess)
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Subset
import random
from sklearn.metrics import roc_auc_score, hamming_loss, confusion_matrix
from collections import defaultdict
import tqdm
import wandb
import logging
import torch.nn.functional as F
import libauc.losses as auc_losses

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([torch.tensor(item['labels'], dtype=torch.float32) for item in batch])

    if 'positive' in batch[0]:
        positive_values = torch.stack([item['positive'] for item in batch])
        # Generate multiple negative indices within the batch
        batch_size = len(batch)
        negative_indices = []
        for i in range(batch_size):
            # Find indices of samples with different labels
            neg_indices = [j for j in range(batch_size) if not torch.all(labels[i] == labels[j])]
            if not neg_indices:  # If no different labels found, use all other indices
                neg_indices = [j for j in range(batch_size) if j != i]
            negative_indices.append(neg_indices)

        # Pad negative indices to the same length
        max_neg_count = max(len(indices) for indices in negative_indices)
        padded_negative_indices = [indices + [indices[-1]] * (max_neg_count - len(indices)) for indices in
                                   negative_indices]
        padded_negative_indices = torch.tensor(padded_negative_indices)
    else:
        positive_values = None
        padded_negative_indices = None

    # Padding questions to the same length
    questions = [item['question'] for item in batch]
    dataset = [item['dataset'] for item in batch]
    max_question_length = max(len(q) for q in questions)
    padded_questions = [q.ljust(max_question_length) for q in questions]

    # Convert targets to a list of lists
    targets = [item['targets'] for item in batch]

    return_dict = {
        'pixel_values': pixel_values,
        'positive': positive_values,
        'labels': labels,
        'questions': padded_questions,
        'targets': targets,
        'dataset': dataset,
        'negative_indices': padded_negative_indices
    }

    return return_dict


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            meta_path,
            num_image_token,
            image_size=448,
            is_train=False,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=False,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=9,
            min_num_frame=4,  # for video data
            max_num_frame=12,  # for video data
            sampling_method='rand',  # for video data
            normalize_type='imagenet',
            random_seed=0,
            rebuild_vocab=False
    ):
        super(LazySupervisedDataset, self).__init__()
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.normalize_type = normalize_type

        logger.info('Formatting inputs...Skip in lazy mode')

        vocabs = []
        data_items = []
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        for ds_name, ds_meta in meta.items():
            root = ds_meta['root']
            with open(ds_meta['annotation'], 'r') as f:  # jsonl
                bar = tqdm.tqdm(f, desc=f'Loading {ds_name}', total=ds_meta['length'])
                for line in f:
                    data = json.loads(line)
                    for index, image in enumerate(data['images']):
                        real_image_path = os.path.join(root, image)
                        data['images'][index] = real_image_path
                    target = data['conversations'][1]['value'].lower()
                    targets = [label.strip() for label in target.split(',')]
                    vocabs.extend(targets)
                    data['targets'] = targets
                    data['dataset'] = ds_name
                    data_items.append(data)
                    bar.update(1)

        vocab_path = os.path.join(os.path.dirname(meta_path), 'vocabs.json')
        if not os.path.exists(vocab_path) or rebuild_vocab:
            logger.info('Building vocab...')
            vocabs = self.build_vocab(vocab_path, vocabs)
            logger.info(f'Vocab size: {len(vocabs)}')
            logger.info(f'Vocab: {vocabs}')
        else:
            with open(vocab_path, 'r') as f:
                vocabs = json.load(f)
        vocabs_to_index = {v: i for i, v in enumerate(vocabs)}
        for data in data_items:
            labels = [0] * len(vocabs)
            for target in data['targets']:
                if target in vocabs_to_index:
                    labels[vocabs_to_index[target]] = 1
                else:
                    labels[-1] = 1  # unk
            data['labels'] = labels
        with open(vocab_path, 'w') as f:
            json.dump(vocabs, f)

        # calculate positive ratio
        agg_labels = torch.tensor([data['labels'] for data in data_items])
        self.positive_ratio = agg_labels.sum(dim=0) / len(agg_labels)
        self.positive_ratio = self.positive_ratio.cpu().tolist()

        self.data_items: List[Dict] = data_items
        self.vocabs = vocabs
        self.vocabs_to_index = vocabs_to_index

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.data_items)

        gc.collect()
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        gc.collect()

    def build_vocab(self, vocab_path, vocabs):
        vocabs = list(set(vocabs))
        vocabs.sort()
        vocabs.append('unknown')
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
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['images'])
        for image_path in data_item['images']:
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // num_image,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        question = data_item['conversations'][0]['value']
        dataset = data_item['dataset']

        # Create the final return dictionary
        ret = dict(
            pixel_values=pixel_values,
            labels=data_item['labels'],
            targets=data_item['targets'],
            question=question,
            dataset=dataset,
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.data_items)
        data_item = self.data_items[i]
        ret = self.multi_modal_get_item(data_item)
        return ret


class ContrastiveLazySupervisedDataset(LazySupervisedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_label_to_indices = defaultdict(dict)
        self.label_set = set()

        for idx, item in enumerate(self.data_items):
            label_tuple = tuple(item['labels'])
            dataset = item['dataset']
            if label_tuple not in self.dataset_label_to_indices[dataset]:
                self.dataset_label_to_indices[dataset][label_tuple] = []
            self.dataset_label_to_indices[dataset][label_tuple].append(idx)
            self.label_set.add(label_tuple)

        # Pre-compute different labels for each label
        self.different_labels = {
            label: list(self.label_set - {label}) for label in self.label_set
        }

        # Log some statistics
        logger.info(f"Total number of samples: {len(self.data_items)}")
        logger.info(f"Number of unique labels: {len(self.label_set)}")
        logger.info(f"Number of datasets: {len(self.dataset_label_to_indices)}")

    def __getitem__(self, i):
        i = i % len(self.data_items)
        anchor_item = self.data_items[i]
        anchor_label = tuple(anchor_item['labels'])

        # Get the anchor image
        anchor_ret = self.multi_modal_get_item(anchor_item)
        anchor_dataset = anchor_item['dataset']

        # Get a positive sample (same label, can be from any dataset)
        positive_indices = []
        for dataset_labels in self.dataset_label_to_indices.values():
            if anchor_label in dataset_labels:
                positive_indices.extend(dataset_labels[anchor_label])

        if not positive_indices:
            logger.warning(f"No positive samples found for label {anchor_label}")
            positive_idx = i  # Use the anchor as positive if no other positives found
        else:
            positive_idx = random.choice(positive_indices)
        positive_item = self.data_items[positive_idx]
        positive_ret = self.multi_modal_get_item(positive_item)

        return {
            'pixel_values': anchor_ret['pixel_values'],
            'positive': positive_ret['pixel_values'],
            'labels': anchor_ret['labels'],
            'targets': anchor_ret['targets'],
            'question': anchor_ret['question'],
            'dataset': anchor_ret['dataset'],
        }


def contrastive_loss(anchor, positive, negative_indices, temperature=0.07):
    batch_size = anchor.size(0)

    # Compute similarity between anchor and positive
    sim_positive = F.cosine_similarity(anchor, positive, dim=1) / temperature

    # Compute similarities between anchor and all negative samples
    neg_sims = torch.empty(batch_size, negative_indices.size(1), device=anchor.device)
    for i in range(batch_size):
        neg_sims[i] = F.cosine_similarity(
            anchor[i].unsqueeze(0),
            anchor[negative_indices[i]],
            dim=1
        ) / temperature

    # Concatenate positive and negative similarities
    all_sims = torch.cat([sim_positive.unsqueeze(1), neg_sims], dim=1)

    # Compute the loss using logsumexp
    losses = -sim_positive + torch.logsumexp(all_sims, dim=1)

    return losses.mean()


def evaluate_classifier(model, dataset, device, split, bs, step, epoch, num_samples=-1):
    model.eval()
    dataset_outputs = defaultdict(list)
    dataset_labels = defaultdict(list)

    if len(dataset) > num_samples > 0:
        subset_indices = random.sample(range(len(dataset)), num_samples)
        subset = Subset(dataset, subset_indices)
    else:
        subset = dataset

    dataloader = torch.utils.data.DataLoader(subset, batch_size=bs, shuffle=False, collate_fn=collate_fn,
                                             num_workers=32, pin_memory=True)

    with torch.no_grad():
        bar = tqdm.tqdm(dataloader, desc=f'Evaluating {split}')
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            datasets = batch['dataset']
            outputs = model.forward_vision(pixel_values)
            outputs = torch.sigmoid(outputs)

            for output, label, ds in zip(outputs, labels, datasets):
                dataset_outputs[ds].append(output.float().cpu().numpy())
                dataset_labels[ds].append(label.float().cpu().numpy())

            bar.update(1)

    overall_stats = {
        'auc': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'perfect_match': []
    }

    for ds in dataset_outputs.keys():
        ds_outputs = np.stack(dataset_outputs[ds])
        ds_labels = np.stack(dataset_labels[ds])

        valid_classes = np.where((ds_labels.sum(axis=0) > 0) & (ds_labels.sum(axis=0) < len(ds_labels)))[0]
        ds_outputs = ds_outputs[:, valid_classes]
        ds_labels = ds_labels[:, valid_classes]

        # Calculate metrics
        try:
            auc = roc_auc_score(ds_labels, ds_outputs, average='macro')
        except ValueError:
            auc = None

        # Convert outputs to binary predictions
        predictions = (ds_outputs > 0.5).astype(int)
        try:
            accuracy = 1 - hamming_loss(ds_labels, predictions)
            # also calculate perfect match accuracy
            perfect_match = (ds_labels == predictions).all(axis=1).mean()
            # Calculate sensitivity and specificity
            cm = confusion_matrix(ds_labels.ravel(), predictions.ravel())
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

            # Log dataset-specific metrics to wandb
            stats = {
                f'{ds}/{split}/auc': auc,
                f'{ds}/{split}/accuracy': accuracy,
                f'{ds}/{split}/sensitivity': sensitivity,
                f'{ds}/{split}/specificity': specificity,
                f'{ds}/{split}/perfect_match': perfect_match,
            }
            wandb.log(stats)
            logger.info(f"Dataset {ds}: {stats}")

            # Accumulate for overall statistics
            ds_length = len(ds_labels)
            if auc is not None:
                overall_stats['auc'].extend([auc] * ds_length)
            overall_stats['accuracy'].extend([accuracy] * ds_length)
            overall_stats['sensitivity'].extend([sensitivity] * ds_length)
            overall_stats['specificity'].extend([specificity] * ds_length)
            overall_stats['perfect_match'].extend([perfect_match] * ds_length)
        except ValueError:
            logger.error(f'Error calculating accuracy for {ds}')
            logger.error(f'Labels: {ds_labels}, predictions: {predictions}')

    try:
        # Calculate and log overall statistics
        overall_auc = np.mean(overall_stats['auc']) if overall_stats['auc'] else None
        overall_accuracy = np.mean(overall_stats['accuracy'])
        overall_sensitivity = np.mean(overall_stats['sensitivity'])
        overall_specificity = np.mean(overall_stats['specificity'])

        overall_stats = {
            f'{split}/overall_auc': overall_auc,
            f'{split}/overall_accuracy': overall_accuracy,
            f'{split}/overall_sensitivity': overall_sensitivity,
            f'{split}/overall_specificity': overall_specificity,
            f'{split}/overall_perfect_match': np.mean(overall_stats['perfect_match']),
            'step': step,
            'epoch': epoch
        }
        wandb.log(overall_stats)
        logger.info(f"Overall: {overall_stats}")
    except ValueError:
        logger.error(f'Error calculating overall statistics')

    model.train()


def train_classifier(model_path, output_path, lr=1e-5, bs=16, wd=1e-3, epochs=10, freeze_vision=False,
                     max_grad_norm=2.0, contrastive_weight=0.5, auc_margin=4.0, sampling_rate=0.5):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    wandb.init(project='high_modality')
    train_dataset = ContrastiveLazySupervisedDataset('../../../processing/meta_train.json',
                                                     num_image_token=256, rebuild_vocab=True, is_train=True)
    # sampler = DualSampler(train_dataset, batch_size=bs, sampling_rate=sampling_rate, shuffle=True)

    train_val_dataset = LazySupervisedDataset('../../../processing/meta_train.json', num_image_token=256)
    val_dataset = LazySupervisedDataset('../../../processing/meta_valid.json', num_image_token=256)
    vocab_size = len(train_dataset.vocabs)

    # Load the model
    model = InternVLChatModel.from_pretrained(model_path,
                                              vision_only=True, vision_output_size=vocab_size,
                                              torch_dtype=torch.bfloat16)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if freeze_vision:
        logger.info('Freezing vision model')
        for param in model.vision_model.parameters():
            param.requires_grad = False

    # Load the dataset
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                                             num_workers=32, pin_memory=True, persistent_workers=True,
                                             prefetch_factor=8)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    auc_loss_fn = auc_losses.MultiLabelAUCMLoss(margin=auc_margin, version='v1', num_labels=vocab_size,
                                                device=device, imratio=train_dataset.positive_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params}')

    # save vocab to output path
    with open(os.path.join(output_path, 'vocabs.json'), 'w') as f:
        json.dump(train_dataset.vocabs, f)

    # Train the model
    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            pixel_values = batch['pixel_values'].cuda().to(torch.bfloat16)
            positive_values = batch['positive'].cuda().to(torch.bfloat16)
            labels = batch['labels'].cuda().to(torch.bfloat16)
            negative_indices = batch['negative_indices'].cuda()

            features = model.forward_vision(pixel_values, classify=False)
            positive_outputs = model.forward_vision(positive_values, classify=False)
            outputs = model.classify(features)

            classification_loss = loss_fn(outputs, labels)
            contr_loss = contrastive_loss(features, positive_outputs, negative_indices)

            if step > 5050:
                auc_loss = auc_loss_fn(outputs, labels)
            else:
                auc_loss = torch.tensor(0.0).to(device)

            # Backward pass
            total_loss = classification_loss + contrastive_weight * contr_loss + auc_loss
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            stats = {
                'loss': total_loss.item(),
                'classification_loss': classification_loss.item(),
                'contrastive_loss': contr_loss.item(),
                'auc_loss': auc_loss.item(),
                'step': step,
                'lr': lr,
                'epoch': epoch
            }
            wandb.log(stats)
            if step % 100 == 0:
                logger.info(f'Step {step}: {stats}')
            if step % 500 == 0 and step > 0:
                evaluate_classifier(model, train_val_dataset, device, 'train', step=step, epoch=epoch, bs=bs,
                                    num_samples=8000)
                evaluate_classifier(model, val_dataset, device, 'val', step=step, epoch=epoch, bs=bs, num_samples=-1)
            if step == 100:
                evaluate_classifier(model, train_val_dataset, device, 'train', step=step, epoch=epoch, bs=bs,
                                    num_samples=8000)
            # save every 1000 steps
            if step % 1000 == 0:
                checkpoint_path = os.path.join(output_path, f'checkpoint_{step}.pt')
                torch.save(model.state_dict(), checkpoint_path)
            step += 1

    # Save the model via huggingface
    model.save_pretrained(output_path)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL2-8B')
    arg_parser.add_argument('--output_path', type=str, required=True)
    arg_parser.add_argument('--lr', type=float, default=1e-5)
    arg_parser.add_argument('--wd', type=float, default=1e-3)
    arg_parser.add_argument('--bs', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=10)
    arg_parser.add_argument('--freeze_vision', action='store_true')

    args = arg_parser.parse_args()
    train_classifier(model_path=args.model_path, output_path=args.output_path,
                     lr=args.lr, wd=args.wd, bs=args.bs, epochs=args.epochs, freeze_vision=args.freeze_vision)
