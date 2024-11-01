import argparse
import copy
from typing import Dict, List

import cv2
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

        # Create attention mask (1 for real values, 0 for padding)
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
            if not neg_indices:
                neg_indices = [j for j in range(batch_size) if j != i]
            negative_indices.append(neg_indices)

        max_neg_count = max(len(indices) for indices in negative_indices)
        padded_negative_indices = [indices + [indices[-1]] * (max_neg_count - len(indices))
                                   for indices in negative_indices]
        padded_negative_indices = torch.tensor(padded_negative_indices)
    else:
        positive_values = None
        positive_attention_mask = None
        padded_negative_indices = None

    questions = [item['question'] for item in batch]
    dataset = [item['dataset'] for item in batch]
    max_question_length = max(len(q) for q in questions)
    padded_questions = [q.ljust(max_question_length) for q in questions]
    targets = [item['targets'] for item in batch]

    return_dict = {
        'pixel_values': pixel_values,
        'attention_mask': attention_mask,
        'positive': positive_values,
        'positive_attention_mask': positive_attention_mask,
        'labels': labels,
        'questions': padded_questions,
        'targets': targets,
        'dataset': dataset,
        'negative_indices': padded_negative_indices
    }

    return return_dict


def sample_frames(video_path: str, num_frames: int, is_train: bool) -> List[Image.Image]:
    """
    Sample frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample
        is_train: If True, use random sampling; if False, use uniform sampling

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
        # Uniform sampling for testing
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

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
            dynamic_image_size=True,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=8,
            min_num_frame=4,  # for video data
            max_num_frame=8,  # for video data
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
                    if 'images' in data:
                        for index, image in enumerate(data['images']):
                            real_image_path = os.path.join(root, image)
                            data['images'][index] = real_image_path
                    if 'videos' in data:
                        for index, video in enumerate(data['videos']):
                            real_video_path = os.path.join(root, video)
                            data['videos'][index] = real_video_path
                    target = data['conversations'][1]['value'].lower()
                    targets = [label.strip() for label in target.split(',')]
                    vocabs.extend(targets)
                    data['targets'] = targets
                    data['dataset'] = ds_name
                    data_items.append(data)
                    bar.update(1)

        meta_name = os.path.basename(meta_path).replace('.json', '').replace('_train', '').replace('_valid', '')

        vocab_path = os.path.join(os.path.dirname(meta_path), f'{meta_name}_vocabs.json')
        if not os.path.exists(vocab_path) or rebuild_vocab:
            logger.info('Building vocab...')
            vocabs = self.build_vocab(vocab_path, vocabs)
            logger.info(f'Vocab size: {len(vocabs)}')
            logger.info(f'Vocab: {vocabs}')
        else:
            with open(vocab_path, 'r') as f:
                vocabs = json.load(f)
            logger.info(f'Loaded vocab size: {len(vocabs)}')
            logger.info(f'Loaded vocab: {vocabs}')
        vocabs_to_index = {v: i for i, v in enumerate(vocabs)}
        self.dataset_multilabel = []
        all_labels = []
        for data in data_items:
            labels = [0] * len(vocabs)
            if data['dataset'] not in self.dataset_multilabel and len(data['targets']) > 1:
                self.dataset_multilabel.append(data['dataset'])
            for target in data['targets']:
                if target in vocabs_to_index:
                    labels[vocabs_to_index[target]] = 1
                else:
                    labels[-1] = 1  # unk
                    logger.warning(f'Unknown label: {target}')
            all_labels.append(labels)
            data['labels'] = labels
        with open(vocab_path, 'w') as f:
            json.dump(vocabs, f)

        # calculate pos_weight
        agg_labels = torch.tensor(all_labels, dtype=torch.float)
        self.pos_weight = 1 / agg_labels.mean(dim=0).clamp(min=1e-6)
        logger.info(f'pos_weight: {self.pos_weight}')

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
        # vocabs.append('unknown')
        with open(vocab_path, 'w') as f:
            json.dump(vocabs, f)
        return vocabs

    def save_annotation(self, path, rel_path):
        # save all the data items to a json file
        relative_data = []

        for data in self.data_items:
            relative_images = [os.path.relpath(image, rel_path) for image in data['images']]
            copy_data = copy.deepcopy(data)
            copy_data['images'] = relative_images
            relative_data.append(copy_data)

        with open(path, 'w') as f:
            json.dump(rel_path, f)

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
        if 'images' in data_item:
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

        if 'videos' in data_item:
            for video_path in data_item['videos']:
                try:
                    # Sample frames from the video
                    video_frames = sample_frames(
                        video_path,
                        num_frames=self.max_num_frame,
                        is_train=self.is_train
                    )

                    if self.dynamic_image_size:
                        for frame in video_frames:
                            processed_frames = dynamic_preprocess(
                                frame,
                                min_num=self.min_dynamic_patch,
                                max_num=self.max_dynamic_patch // len(video_frames),
                                image_size=self.image_size,
                                use_thumbnail=self.use_thumbnail
                            )
                            images += processed_frames
                            num_tiles.append(len(processed_frames))
                    else:
                        images.extend(video_frames)
                        num_tiles.extend([1] * len(video_frames))

                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {e}")
                    # Add black frames as fallback
                    black_frames = [Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))] * 8
                    images.extend(black_frames)
                    num_tiles.extend([1] * 8)

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


def evaluate_classifier(model, dataset, device, split, bs, step, epoch, num_samples=-1, dataloader=None):
    model.eval()
    dataset_outputs = defaultdict(list)
    dataset_labels = defaultdict(list)

    if dataloader is None:
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
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            datasets = batch['dataset']
            outputs = model.forward_vision(pixel_values, attention_mask)
            # outputs = torch.sigmoid(outputs)

            for output, label, ds in zip(outputs, labels, datasets):
                dataset_outputs[ds].append(output.float().cpu())
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
        ds_outputs = torch.stack(dataset_outputs[ds])
        ds_labels = np.stack(dataset_labels[ds])

        valid_classes = np.where((ds_labels.sum(axis=0) > 0) & (ds_labels.sum(axis=0) < len(ds_labels)))[0]
        ds_outputs = ds_outputs[:, valid_classes]
        ds_labels = ds_labels[:, valid_classes]

        if ds in dataset.dataset_multilabel:
            # use sigmoid
            ds_outputs = torch.sigmoid(ds_outputs).numpy()
        else:
            # use softmax
            ds_outputs = torch.softmax(ds_outputs, dim=1).numpy()

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
                     max_grad_norm=2.0, contrastive_weight=0.5,
                     meta_train_path='../../../processing/meta_train.json',
                     meta_valid_path='../../../processing/meta_valid.json',
                     eval_only=False, load_checkpoint=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    wandb.init(project='high_modality')
    train_dataset = ContrastiveLazySupervisedDataset(meta_train_path,
                                                     num_image_token=256, rebuild_vocab=True, is_train=True)
    # sampler = DualSampler(train_dataset, batch_size=bs, sampling_rate=sampling_rate, shuffle=True)

    train_val_dataset = LazySupervisedDataset(meta_train_path, num_image_token=256)
    val_dataset = LazySupervisedDataset(meta_valid_path, num_image_token=256)
    # train_dataset.save_annotation(meta_train_path,
    #                               rel_path='/home/dvd/Datasets/high_modality/')
    # val_dataset.save_annotation(meta_valid_path,
    #                             rel_path='/home/dvd/Datasets/high_modality/')
    vocab_size = len(train_dataset.vocabs)

    # Load the model
    model = InternVLChatModel.from_pretrained(model_path, vision_only=True, vision_output_size=vocab_size,
                                              torch_dtype=torch.bfloat16)
    if load_checkpoint is not None:
        model.load_state_dict(torch.load(load_checkpoint))
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if freeze_vision:
        logger.info('Freezing vision model')
        for param in model.vision_model.parameters():
            param.requires_grad = False

    # Load the dataset
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                                             num_workers=32, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn,
                                                 num_workers=32, pin_memory=True)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=train_dataset.pos_weight.cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if eval_only:
        evaluate_classifier(model, train_val_dataset, device, 'train', bs=bs, step=0, epoch=0, num_samples=8000)
        evaluate_classifier(model, val_dataset, device, 'val', bs=bs, step=0, epoch=0, num_samples=-1)
        return

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params}')

    # save vocab to output path
    meta_train_name = os.path.basename(meta_train_path)
    meta_name = meta_train_name.replace('_train', '').replace('.json', '')
    with open(os.path.join(output_path, f'{meta_name}_vocabs.json'), 'w') as f:
        json.dump(train_dataset.vocabs, f)

    # Train the model
    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            pixel_values = batch['pixel_values'].cuda().to(torch.bfloat16)
            attention_mask = batch['attention_mask'].cuda()
            positive_values = batch['positive'].cuda().to(torch.bfloat16)
            positive_attention_mask = batch['positive_attention_mask'].cuda()
            labels = batch['labels'].cuda().to(torch.bfloat16)
            negative_indices = batch['negative_indices'].cuda()

            features = model.forward_vision(pixel_values, attention_mask=attention_mask, classify=False)
            positive_outputs = model.forward_vision(positive_values, attention_mask=positive_attention_mask,
                                                    classify=False)
            outputs = model.classify(features)

            classification_loss = loss_fn(outputs, labels) * 5
            contr_loss = contrastive_loss(features, positive_outputs, negative_indices)

            # Backward pass
            total_loss = classification_loss + contrastive_weight * contr_loss
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
                evaluate_classifier(model, val_dataset, device, 'val', step=step, epoch=epoch, bs=bs,
                                    num_samples=-1, dataloader=val_dataloader)
            if step == 100:
                evaluate_classifier(model, train_val_dataset, device, 'train', step=step, epoch=epoch, bs=bs,
                                    num_samples=8000)
                evaluate_classifier(model, val_dataset, device, 'val', step=step, epoch=epoch, bs=bs,
                                    num_samples=-1, dataloader=val_dataloader)
            # save every 1000 steps
            if step % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(output_path, f'model_{step}.pt'))
            step += 1

    # Save the model via huggingface
    torch.save(model.state_dict(), os.path.join(output_path, 'model.pt'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL2-8B')
    arg_parser.add_argument('--output_path', type=str, required=True)
    arg_parser.add_argument('--lr', type=float, default=1e-5)
    arg_parser.add_argument('--wd', type=float, default=0)
    arg_parser.add_argument('--bs', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=10)
    arg_parser.add_argument('--freeze_vision', action='store_true')
    arg_parser.add_argument('--meta_train_path', type=str, default='../../../processing/meta_train.json')
    arg_parser.add_argument('--meta_valid_path', type=str, default='../../../processing/meta_valid.json')
    arg_parser.add_argument('--eval_only', action='store_true')
    arg_parser.add_argument('--load_checkpoint', type=str, default=None)

    args = arg_parser.parse_args()
    train_classifier(model_path=args.model_path, output_path=args.output_path,
                     lr=args.lr, wd=args.wd, bs=args.bs, epochs=args.epochs, freeze_vision=args.freeze_vision,
                     meta_train_path=args.meta_train_path, meta_valid_path=args.meta_valid_path,
                     eval_only=args.eval_only,
                     load_checkpoint=args.load_checkpoint)
