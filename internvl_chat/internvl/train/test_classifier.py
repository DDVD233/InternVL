import argparse
from typing import Dict

import wandb

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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([torch.tensor(item['labels'], dtype=torch.float32) for item in batch])

    # Padding questions to the same length
    questions = [item['question'] for item in batch]
    dataset = [item['dataset'] for item in batch]
    max_question_length = max(len(q) for q in questions)
    padded_questions = [q.ljust(max_question_length) for q in questions]

    # Convert targets to a list of lists
    targets = [item['targets'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'questions': padded_questions,
        'targets': targets,
        'dataset': dataset
    }


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

        self.data_items = data_items
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
        self.normalize_type = normalize_type
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
        'specificity': []
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
            'step': step,
            'epoch': epoch
        }
        wandb.log(overall_stats)
        logger.info(f"Overall: {overall_stats}")
    except ValueError:
        logger.error(f'Error calculating overall statistics')

    model.train()


def train_classifier(model_path, output_path, lr=1e-5, bs=16, wd=1e-3, epochs=10, freeze_vision=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    wandb.init(project='high_modality')
    train_dataset = LazySupervisedDataset('../../../processing/meta_train.json',
                                          num_image_token=256, rebuild_vocab=True)
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
                                             num_workers=32, pin_memory=True, persistent_workers=True)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Count trainable parameters
    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f'Trainable parameters: {trainable_params}')

    # Train the model
    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            pixel_values = batch['pixel_values'].cuda()
            outputs = model.forward_vision(pixel_values)
            labels = batch['labels'].cuda().to(outputs.dtype)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            stats = {
                'loss': loss.item(),
                'step': step,
                'lr': lr,
                'epoch': epoch
            }
            wandb.log(stats)
            if step % 100 == 0:
                logger.info(f'Step {step}: {stats}')
            if step % 500 == 0 and step > 0:
                evaluate_classifier(model, train_dataset, device, 'train', step=step, epoch=epoch, bs=bs, num_samples=8000)
                evaluate_classifier(model, val_dataset, device, 'val', step=step, epoch=epoch, bs=bs, num_samples=-1)
            if step == 10:
                evaluate_classifier(model, train_dataset, device, 'train', step=step, epoch=epoch, bs=bs, num_samples=8000)
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
