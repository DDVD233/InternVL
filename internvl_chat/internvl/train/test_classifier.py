import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from internvl.model.clip import OpenCLIPClassifier
from internvl.model.convnext import ConvNextV2Classifier
from internvl.model.eva_classifier import EVA02Classifier
from internvl.model.swin_transformer import SwinV2Classifier
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.sbb_vit import ViTSBBClassifier
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.model.internvl_chat.internvl_moe import MoEVisionModel
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, hamming_loss, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, Subset
from transformers import get_cosine_schedule_with_warmup, AutoConfig
from heavyball import PrecondScheduleForeachSOAP

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    captions = [item['caption'] for item in batch]
    dataset = [item['dataset'] for item in batch]
    max_question_length = max(len(q) for q in questions)
    padded_questions = [q.ljust(max_question_length) for q in questions]
    targets = [item['targets'] for item in batch]
    modalities = [item['modality'] for item in batch]
    positive_modalities = [item['positive_modality'] for item in batch] if 'positive_modality' in batch[0] else None

    return_dict = {
        'pixel_values': pixel_values,
        'attention_mask': attention_mask,
        'positive': positive_values,
        'positive_attention_mask': positive_attention_mask,
        'labels': labels,
        'questions': padded_questions,
        'caption': captions,
        'targets': targets,
        'dataset': dataset,
        'negative_indices': padded_negative_indices,
        'modality': modalities,
        'positive_modality': positive_modalities
    }

    return return_dict


def build_dataset_label_indices(data_items, vocabs):
    """Build a mapping of which labels appear in which datasets."""
    dataset_label_indices = {}
    dataset_label_counts = defaultdict(lambda: defaultdict(int))

    # First pass: count label occurrences per dataset
    for item in data_items:
        dataset = item['dataset']
        for target in item['targets']:
            if target in vocabs:
                label_idx = vocabs.index(target)
                dataset_label_counts[dataset][label_idx] += 1

    # Second pass: keep only labels that appear in each dataset
    for dataset, label_counts in dataset_label_counts.items():
        valid_indices = []
        for label_idx, count in label_counts.items():
            # You might want to adjust this threshold based on your needs
            if count > 0:  # or use a higher threshold like count > 10
                valid_indices.append(label_idx)
        dataset_label_indices[dataset] = sorted(valid_indices)

    return dataset_label_indices


class MixedClassificationLoss(nn.Module):
    def __init__(self, dataset_label_indices, multilabel_datasets, pos_weight=None):
        super().__init__()
        self.dataset_label_indices = dataset_label_indices
        self.multilabel_datasets = multilabel_datasets
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets, datasets):
        """
        Args:
            predictions: (batch_size, num_classes) model predictions
            targets: (batch_size, num_classes) target labels
            datasets: list of dataset names for each sample
        """
        total_loss = 0
        batch_size = len(datasets)
        valid_samples = 0

        # Group samples by dataset type
        multilabel_mask = torch.tensor([d in self.multilabel_datasets for d in datasets],
                                       device=predictions.device)

        if multilabel_mask.any():
            # Handle multi-label samples
            ml_preds = predictions[multilabel_mask]
            ml_targets = targets[multilabel_mask]
            ml_datasets = [d for i, d in enumerate(datasets) if multilabel_mask[i]]

            for i, (pred, target, dataset) in enumerate(zip(ml_preds, ml_targets, ml_datasets)):
                valid_indices = self.dataset_label_indices[dataset]
                if not valid_indices:
                    continue

                valid_pred = pred[valid_indices]
                valid_target = target[valid_indices]

                if self.pos_weight is not None:
                    valid_pos_weight = self.pos_weight[valid_indices]
                    sample_loss = self.bce_loss(valid_pred, valid_target) * valid_pos_weight
                else:
                    sample_loss = self.bce_loss(valid_pred, valid_target)

                total_loss += sample_loss.mean()
                valid_samples += 1

        if (~multilabel_mask).any():
            # Handle multi-class samples
            mc_preds = predictions[~multilabel_mask]
            mc_targets = targets[~multilabel_mask]
            mc_datasets = [d for i, d in enumerate(datasets) if not multilabel_mask[i]]

            for i, (pred, target, dataset) in enumerate(zip(mc_preds, mc_targets, mc_datasets)):
                valid_indices = self.dataset_label_indices[dataset]
                if not valid_indices:
                    continue

                valid_pred = pred[valid_indices]
                valid_target = target[valid_indices]

                # Convert one-hot to class index for CrossEntropyLoss
                target_idx = valid_target.argmax()
                sample_loss = self.ce_loss(valid_pred.unsqueeze(0), target_idx.unsqueeze(0))

                total_loss += sample_loss.mean()
                valid_samples += 1

        if valid_samples == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return total_loss / valid_samples


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


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            meta_path,
            num_image_token,
            output_path,
            image_size=448,
            is_train=False,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=True,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=8,
            min_num_frame=4,  # for video data
            # max_num_frame=8,  # for video data
            normalize_type='imagenet',
            random_seed=0,
            rebuild_vocab=False,
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
        self.max_num_frame = max_dynamic_patch
        self.min_num_frame = min_num_frame
        self.normalize_type = normalize_type

        logger.info('Formatting inputs...Skip in lazy mode')

        vocabs = []
        data_items = []
        self.unique_modalities = set()
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

                    # Extract and standardize modality
                    modality = ds_name.split('_')[0]  # chest, derm, mammo, etc
                    # modality = modality.replace('chest', 'chest x-ray') \
                    #     .replace('mammo', 'mammography') \
                    #     .replace('derm', 'dermoscopy') \
                    #     .replace('mri', 'MRI') \
                    #     .replace('ct', 'CT')
                    data['modality'] = modality
                    self.unique_modalities.add(modality)

                    data['caption'] = modality + ', ' + target
                    if 'images' in data:
                        data_items.append(data)
                    elif 'videos' in data:
                        # Oversampling 20x for video data
                        data_items.extend([data] * 20)
                    bar.update(1)

        logger.info(f"Found modalities: {sorted(list(self.unique_modalities))}")

        # meta_name = os.path.basename(meta_path).replace('.json', '').replace('_train', '').replace('_valid', '')
        vocab_path = os.path.join(output_path, f'vocabs.json')
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

    def get_modalities(self):
        """Return the list of unique modalities in the dataset."""
        return sorted(list(self.unique_modalities))

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
                    black_frames = [Image.new('RGB', (self.image_size, self.image_size),
                                              (0, 0, 0))] * self.max_dynamic_patch
                    images.extend(black_frames)
                    num_tiles.extend([1] * self.max_dynamic_patch)

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
            caption=data_item['caption'],
            dataset=dataset,
            modality=data_item['modality']
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
            'caption': anchor_ret['caption'],
            'modality': anchor_ret['modality'],
            'positive_modality': positive_ret['modality']
        }


def filter_by_modality(data_items, test_modality, in_mod_pct, out_mod_pct, few_shot_datasets, vocabs):
    """
    Filter data items based on modality percentages while preserving few-shot datasets.

    Args:
        data_items: List of data items
        test_modality: Target modality to filter
        in_mod_pct: Percentage of in-modality data to keep (0.0-1.0)
        out_mod_pct: Percentage of out-of-modality data to keep (0.0-1.0)
        few_shot_datasets: List of dataset names to preserve
        vocabs: List of vocabulary items for stratification
    """
    if test_modality is None:
        return data_items

    # Organize data by modality, dataset, and label
    modality_dataset_label_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in tqdm.tqdm(data_items, desc='Organizing data for filtering'):
        dataset = item['dataset']
        modality = item['modality']
        label_tuple = tuple(item['labels'])
        modality_dataset_label_data[modality][dataset][label_tuple].append(item)

    filtered_items = []

    # Process each modality
    for modality, dataset_dict in modality_dataset_label_data.items():
        for dataset, label_dict in dataset_dict.items():
            # Skip filtering for few-shot datasets
            if dataset in few_shot_datasets and modality == test_modality:
                for items in label_dict.values():
                    filtered_items.extend(items)
                continue

            # Determine percentage to keep based on modality
            keep_pct = in_mod_pct if modality == test_modality else out_mod_pct

            # If keeping all data, skip filtering
            if keep_pct >= 1.0:
                for items in label_dict.values():
                    filtered_items.extend(items)
                continue
            elif keep_pct <= 0.0:
                continue

            # Stratified sampling for each label
            for label_tuple, items in label_dict.items():
                num_to_keep = max(1, int(len(items) * keep_pct))
                kept_items = random.sample(items, num_to_keep)
                filtered_items.extend(kept_items)

    return filtered_items


class FewShotContrastiveLazySupervisedDataset(ContrastiveLazySupervisedDataset):
    def __init__(
            self,
            meta_path,
            num_image_token,
            output_path,
            few_shot_datasets=None,
            shots_per_class=32,
            test_modality=None,
            in_mod_pct=1.0,
            out_mod_pct=1.0,
            image_size=448,
            is_train=False,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=True,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=8,
            min_num_frame=4,
            normalize_type='imagenet',
            random_seed=0,
            rebuild_vocab=False,
    ):
        # Initialize with parent class first
        super().__init__(
            meta_path=meta_path,
            num_image_token=num_image_token,
            output_path=output_path,
            image_size=image_size,
            is_train=is_train,
            pad2square=pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            min_num_frame=min_num_frame,
            normalize_type=normalize_type,
            random_seed=random_seed,
            rebuild_vocab=rebuild_vocab,
        )

        self.few_shot_datasets = few_shot_datasets or []
        self.shots_per_class = shots_per_class

        if test_modality is not None:
            logger.info(f'Filtering data based on modality: {test_modality}')
            logger.info(f'In-modality percentage: {in_mod_pct}')
            logger.info(f'Out-modality percentage: {out_mod_pct}')

        # Apply modality filtering
        self.data_items = filter_by_modality(
            self.data_items,
            test_modality,
            in_mod_pct,
            out_mod_pct,
            self.few_shot_datasets,
            self.vocabs
        )

        if self.few_shot_datasets and self.is_train:
            logger.info(f'Applying few-shot sampling for datasets: {few_shot_datasets}')
            logger.info(f'Shots per class: {shots_per_class}')

            # Organize data by dataset and label combination
            dataset_label_data = defaultdict(lambda: defaultdict(list))
            for item in self.data_items:
                dataset_label_data[item['dataset']][tuple(item['labels'])].append(item)

            # Apply few-shot sampling
            final_data_items = []
            for dataset, label_data in tqdm.tqdm(dataset_label_data.items(), desc='Few-shot sampling'):
                if dataset in self.few_shot_datasets:
                    # Few-shot sampling
                    dataset_size = 0
                    sampled_items = []
                    for label_tuple, items in label_data.items():
                        samples = random.sample(items, min(self.shots_per_class, len(items)))
                        sampled_items.extend(samples)
                        dataset_size += len(items)

                    # Oversample to match original dataset size
                    if sampled_items:
                        num_repeats = max(1, dataset_size // len(sampled_items))
                        final_data_items.extend(sampled_items * num_repeats)
                else:
                    # Keep all samples for non-few-shot datasets
                    for items in label_data.values():
                        final_data_items.extend(items)

            # Update dataset
            self.data_items = final_data_items
            # shuffle data
            self.rng.shuffle(self.data_items)

        # Update indices and label sets for contrastive learning
        self.rebuild_indices()

        # Log dataset statistics
        self._log_dataset_statistics()

    def rebuild_indices(self):
        """Rebuild indices for contrastive learning after data filtering"""
        self.dataset_label_to_indices = defaultdict(dict)
        for idx, item in enumerate(self.data_items):
            label_tuple = tuple(item['labels'])
            dataset = item['dataset']
            if label_tuple not in self.dataset_label_to_indices[dataset]:
                self.dataset_label_to_indices[dataset][label_tuple] = []
            self.dataset_label_to_indices[dataset][label_tuple].append(idx)

        # Rebuild label set and different labels mapping
        self.label_set = set(tuple(item['labels']) for item in self.data_items)
        self.different_labels = {
            label: list(self.label_set - {label}) for label in self.label_set
        }

    def _log_dataset_statistics(self):
        """Log detailed dataset statistics after filtering"""
        logger.info(f'Dataset size after processing: {len(self.data_items)}')

        # Count samples by modality
        modality_counts = defaultdict(int)
        for item in self.data_items:
            modality_counts[item['modality']] += 1

        logger.info("Samples per modality:")
        for modality, count in modality_counts.items():
            logger.info(f"  {modality}: {count}")

        # Count samples in few-shot datasets
        if self.few_shot_datasets:
            logger.info("\nSamples in few-shot datasets:")
            for dataset in self.few_shot_datasets:
                count = sum(1 for item in self.data_items if item['dataset'] == dataset)
                logger.info(f"  {dataset}: {count}")


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


def calculate_multilabel_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate sensitivity (recall) and specificity for multilabel classification.

    Parameters:
    y_true: numpy array of shape (n_samples, n_classes) with true binary labels
    y_pred: numpy array of shape (n_samples, n_classes) with predicted probabilities
    threshold: float, classification threshold (default 0.5)

    Returns:
    dict containing per-class and macro-averaged metrics
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
        num_batches = len(dataloader)
        if num_samples > 0:
            num_batches = math.ceil(num_samples / bs)

        bar = tqdm.tqdm(dataloader, desc=f'Evaluating {split}', total=num_batches)
        for index, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device).to(model.dtype)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            datasets = batch['dataset']
            modalities = batch['modality']
            outputs = model.forward_vision(pixel_values, attention_mask, modalities=modalities)
            # outputs = torch.sigmoid(outputs)

            for output, label, ds in zip(outputs, labels, datasets):
                dataset_outputs[ds].append(output.float().cpu())
                dataset_labels[ds].append(label.float().cpu().numpy())

            bar.update(1)
            if num_samples > 0 and index >= num_batches:
                break

    overall_stats = {
        'auc': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'perfect_match': [],
        'f1': []
    }
    by_modality_stats = defaultdict(lambda: defaultdict(list))

    for ds in dataset_outputs.keys():
        ds_outputs = torch.stack(dataset_outputs[ds])
        ds_labels = np.stack(dataset_labels[ds])

        valid_classes = np.where((ds_labels.sum(axis=0) > 0) & (ds_labels.sum(axis=0) < len(ds_labels)))[0]

        # If no valid class
        if len(valid_classes) == 0:
            logger.warning(f"No valid classes found for dataset {ds}")
            continue

        ds_outputs = ds_outputs[:, valid_classes]
        ds_labels = ds_labels[:, valid_classes]

        if ds in dataset.dataset_multilabel:  # This is multilabel
            # use sigmoid
            ds_outputs = torch.sigmoid(ds_outputs).numpy()

            # Convert outputs to binary predictions
            predictions = (ds_outputs > 0.5).astype(int)
            zero_pred_samples = (predictions.sum(axis=1) == 0)
            if any(zero_pred_samples):
                # Get indices of highest probability for each sample
                highest_prob_indices = predictions[zero_pred_samples].argmax(axis=1)

                # Use advanced indexing to set the highest probability class to 1
                sample_indices = np.where(zero_pred_samples)[0]
                predictions[sample_indices, highest_prob_indices] = 1

            accuracy = 1 - hamming_loss(ds_labels, predictions)
            # also calculate perfect match accuracy
            perfect_match = (ds_labels == predictions).all(axis=1).mean()

        else:  # This is multiclass
            # use softmax
            ds_outputs = torch.softmax(ds_outputs, dim=1).numpy()
            predictions = np.zeros_like(ds_outputs)
            predictions[np.arange(len(ds_outputs)), ds_outputs.argmax(1)] = 1

            perfect_match = accuracy = (predictions == ds_labels).mean()

        try:
            auc = roc_auc_score(ds_labels, ds_outputs, average='macro')
        except ValueError:
            auc = None

        sensitivity, specificity = calculate_multilabel_metrics(ds_labels, predictions)
        f1_scores = f1_score(ds_labels, predictions, average='macro')

        # Log dataset-specific metrics to wandb
        stats = {
            f'{ds}/{split}/auc': auc,
            f'{ds}/{split}/accuracy': accuracy,
            f'{ds}/{split}/sensitivity': sensitivity,
            f'{ds}/{split}/specificity': specificity,
            f'{ds}/{split}/perfect_match': perfect_match,
            f'{ds}/{split}/f1': f1_scores,
            'step': step,
        }
        wandb.log(stats)
        logger.info(f"Dataset {ds}: {stats}")

        # Accumulate for overall statistics
        if auc is not None:
            overall_stats['auc'].extend([auc])
        overall_stats['accuracy'].extend([accuracy])
        overall_stats['sensitivity'].extend([sensitivity])
        overall_stats['specificity'].extend([specificity])
        overall_stats['perfect_match'].extend([perfect_match])
        overall_stats['f1'].extend([f1_scores])

        # Accumulate for modality-specific statistics
        modality = ds.split('_')[0]
        by_modality_stats[modality]['auc'].append(auc)
        by_modality_stats[modality]['accuracy'].append(accuracy)
        by_modality_stats[modality]['sensitivity'].append(sensitivity)
        by_modality_stats[modality]['specificity'].append(specificity)
        by_modality_stats[modality]['perfect_match'].append(perfect_match)
        by_modality_stats[modality]['f1'].append(f1_scores)

    try:
        # Calculate and log overall statistics
        overall_auc = np.mean(overall_stats['auc']) if overall_stats['auc'] else None
        overall_accuracy = np.mean(overall_stats['accuracy'])
        overall_sensitivity = np.mean(overall_stats['sensitivity'])
        overall_specificity = np.mean(overall_stats['specificity'])
        overall_f1 = np.mean(overall_stats['f1'])

        overall_stats = {
            f'{split}/overall_auc': overall_auc,
            f'{split}/overall_accuracy': overall_accuracy,
            f'{split}/overall_sensitivity': overall_sensitivity,
            f'{split}/overall_specificity': overall_specificity,
            f'{split}/overall_perfect_match': np.mean(overall_stats['perfect_match']),
            f'{split}/overall_f1': overall_f1,
            'step': step,
            'epoch': epoch
        }

        # Calculate and log modality-specific statistics
        for modality, stats in by_modality_stats.items():
            modality_auc = np.mean(stats['auc']) if stats['auc'] else None
            modality_accuracy = np.mean(stats['accuracy'])
            modality_sensitivity = np.mean(stats['sensitivity'])
            modality_specificity = np.mean(stats['specificity'])

            overall_stats[f'{split}/{modality}_auc'] = modality_auc
            overall_stats[f'{split}/{modality}_accuracy'] = modality_accuracy
            overall_stats[f'{split}/{modality}_sensitivity'] = modality_sensitivity
            overall_stats[f'{split}/{modality}_specificity'] = modality_specificity
            overall_stats[f'{split}/{modality}_perfect_match'] = np.mean(stats['perfect_match'])
            overall_stats[f'{split}/{modality}_f1'] = np.mean(stats['f1'])

        wandb.log(overall_stats)
        logger.info(f"Overall: {overall_stats}")
    except ValueError:
        logger.error(f'Error calculating overall statistics')

    model.train()


def train_classifier(model_path, output_path, lr=1e-5, bs=16, wd=1e-3, epochs=5, freeze_vision=False,
                     max_grad_norm=2.0, contrastive_weight=0.5,
                     meta_train_path='../../../processing/meta_train.json',
                     meta_valid_path='../../../processing/meta_valid.json',
                     eval_only=False, load_checkpoint=None, no_contrastive=False, unfreeze_vit_layers=0,
                     few_shot=False, shots_per_class=32, all_separate=False, eval_every=2000,
                     test_modality=None, in_mod_pct=1.0, out_mod_pct=1.0, moe=-1):
    set_random_seed(42)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    wandb.init(project='high_modality', config=locals(), name=os.path.basename(output_path))

    # Create and setup file handler
    log_file = os.path.join(output_path, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log all args passed into this function
    logger.info(f'Arguments: {locals()}')

    rebuild_vocab = True
    if load_checkpoint is not None:
        checkpoint_dir = os.path.dirname(load_checkpoint)
        if 'vocabs.json' in os.listdir(checkpoint_dir):  # Not a pretrain model
            rebuild_vocab = False
    if 'internvl' in model_path.lower():
        # dynamic_image_size = True
        image_size = 448
        max_dynamic_patch = 8
    if 'sbb2' in model_path.lower():
        image_size = 384
        max_dynamic_patch = 1
    elif 'eva' in model_path.lower():
        image_size = 448
        max_dynamic_patch = 2
    elif 'swintransformer' in model_path.lower():
        image_size = 192
        max_dynamic_patch = 8
    elif 'convnextv2-large' in model_path.lower():
        image_size = 384
        max_dynamic_patch = 1
    else:
        # dynamic_image_size = False
        image_size = 224
        max_dynamic_patch = 2
    logger.info(f'Image size: {image_size}, Max dynamic patch: {max_dynamic_patch}')

    if few_shot:
        few_shot_datasets = []
        with open(meta_valid_path, 'r') as f:
            valid_meta = json.load(f)
            few_shot_datasets = list(valid_meta.keys())
        logger.info(f'Few-shot datasets from validation: {few_shot_datasets}')
    else:
        few_shot_datasets = None

    train_dataset = FewShotContrastiveLazySupervisedDataset(
        meta_train_path,
        output_path=output_path,
        num_image_token=256,
        rebuild_vocab=rebuild_vocab,
        is_train=True,
        dynamic_image_size=False,
        image_size=image_size,
        max_dynamic_patch=max_dynamic_patch,
        few_shot_datasets=few_shot_datasets,
        shots_per_class=shots_per_class,
        test_modality=test_modality,
        in_mod_pct=in_mod_pct,
        out_mod_pct=out_mod_pct
    )
    train_val_dataset = LazySupervisedDataset(meta_train_path, num_image_token=256,
                                              output_path=output_path, dynamic_image_size=False,
                                              is_train=False, image_size=image_size,
                                              max_dynamic_patch=max_dynamic_patch)

    # Filter validation dataset if test_modality is specified
    if test_modality is not None:
        val_dataset = FewShotContrastiveLazySupervisedDataset(
            meta_valid_path,
            output_path=output_path,
            num_image_token=256,
            is_train=False,
            dynamic_image_size=False,
            image_size=image_size,
            max_dynamic_patch=max_dynamic_patch,
            test_modality=test_modality,
            in_mod_pct=1.0,
            out_mod_pct=0.0
        )
    else:
        val_dataset = LazySupervisedDataset(
            meta_valid_path,
            num_image_token=256,
            output_path=output_path,
            dynamic_image_size=False,
            is_train=False,
            image_size=image_size,
            max_dynamic_patch=max_dynamic_patch
        )
    vocab_size = len(train_dataset.vocabs)

    # Load the model
    if 'sbb2' in model_path.lower():
        model = ViTSBBClassifier(vision_output_size=vocab_size).cuda()
        # model = model.to(torch.bfloat16)
    elif 'convnext' in model_path.lower():
        model = ConvNextV2Classifier.from_pretrained(
            model_path,  # or any other ConvNeXtV2 checkpoint
            vision_output_size=vocab_size
        ).cuda()
        model = model.to(torch.bfloat16)
    elif 'clip' in model_path.lower():
        model = model = OpenCLIPClassifier.from_pretrained(
            model_path,
            vision_output_size=vocab_size,
            dtype=torch.bfloat16
        ).cuda()
    elif 'eva' in model_path.lower():
        model = EVA02Classifier(
            vision_output_size=vocab_size,
            checkpoint_path="eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt"
        ).cuda()
        model = model.to(torch.bfloat16)
    elif 'swintransformer' in model_path.lower():
        model = SwinV2Classifier(
            vision_output_size=vocab_size,
            model_name="microsoft/swinv2-large-patch4-window12-192-22k"
        ).cuda()
        model = model.to(torch.bfloat16)
    elif all_separate:
        modalities = train_dataset.get_modalities()
        model = InternVLChatModel.from_pretrained(model_path, vision_only=True, vision_output_size=vocab_size,
                                                  torch_dtype=torch.bfloat16, all_separate=True, modalities=modalities)
        model.init_encoder_array()
    elif moe > 0:
        config = InternVLChatConfig.from_pretrained(model_path).vision_config
        config.num_experts = 4
        model = MoEVisionModel(config=config, vision_output_size=vocab_size)
        model.load_from_internvl(model_path)
    else:
        model = InternVLChatModel.from_pretrained(model_path, vision_only=True, vision_output_size=vocab_size,
                                                  torch_dtype=torch.bfloat16)
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        if 'model' in checkpoint and 'encoder.embeddings.position_embedding' in checkpoint['model']:
            # This is loaded from pretrain, process the keys
            new_checkpoint = {}
            for key, value in checkpoint['model'].items():
                if 'encoder' in key:
                    new_key: str = key.replace('encoder.', 'vision_model.', 1)
                    new_checkpoint[new_key] = value
            checkpoint = new_checkpoint
        incompatible_keys = model.load_state_dict(checkpoint, strict=False)
        logger.info(f'Loaded checkpoint from {load_checkpoint}')
        if incompatible_keys.missing_keys:
            logger.warning(f'Missing keys: {incompatible_keys.missing_keys}')
        if incompatible_keys.unexpected_keys:
            logger.warning(f'Unexpected keys: {incompatible_keys.unexpected_keys}')
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if freeze_vision:
        logger.info('Freezing vision model')
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # Load the dataset
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                                             num_workers=64, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn,
                                                 num_workers=32, pin_memory=True)
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=bs, shuffle=True,
                                                   collate_fn=collate_fn,
                                                   num_workers=32, pin_memory=True)

    # Calculate total steps for the scheduler
    total_steps = len(dataloader) * epochs
    warmup_steps = 200

    dataset_label_indices = build_dataset_label_indices(train_dataset.data_items, train_dataset.vocabs)

    # Add dataset_label_indices to the dataset object
    train_dataset.dataset_label_indices = dataset_label_indices

    # Initialize the dataset-specific loss function
    loss_fn = MixedClassificationLoss(
        dataset_label_indices=dataset_label_indices,
        multilabel_datasets=train_dataset.dataset_multilabel,
        pos_weight=train_dataset.pos_weight.cuda()
    )

    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=train_dataset.pos_weight.cuda())
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=lr, weight_decay=wd)
    # optimizer = ForeachDelayedPSGD(model.parameters(), lr=lr, weight_decay=wd)
    # optimizer.train()

    # Initialize the learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=1.5
    )

    if eval_only:
        evaluate_classifier(model, train_val_dataset, device, 'train', bs=bs, step=0, epoch=0, num_samples=8000)
        evaluate_classifier(model, val_dataset, device, 'val', bs=bs, step=0, epoch=0, num_samples=-1)
        return

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params}')
    logger.info(f'Total steps: {total_steps}, Warmup steps: {warmup_steps}')

    with open(os.path.join(output_path, f'vocabs.json'), 'w') as f:
        json.dump(train_dataset.vocabs, f)

    # Train the model
    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            pixel_values = batch['pixel_values'].cuda().to(model.dtype)
            attention_mask = batch['attention_mask'].cuda()
            positive_values = batch['positive'].cuda().to(model.dtype)
            positive_attention_mask = batch['positive_attention_mask'].cuda()
            labels = batch['labels'].cuda().to(model.dtype)
            negative_indices = batch['negative_indices'].cuda()
            datasets = batch['dataset']
            modalities = batch['modality']
            positive_modalities = batch['positive_modality']

            features = model.forward_vision(pixel_values, attention_mask=attention_mask,
                                            classify=False, modalities=modalities)
            if isinstance(model, MoEVisionModel):
                features, load_loss = features
            else:
                load_loss = torch.tensor(0.0, device=features.device)
            outputs = model.classify(features)

            classification_loss = loss_fn(outputs, labels, datasets)
            if no_contrastive:
                contr_loss = torch.tensor(0.0, device=features.device)
            else:
                positive_outputs = model.forward_vision(positive_values, attention_mask=positive_attention_mask,
                                                        classify=False, modalities=positive_modalities)
                if isinstance(model, MoEVisionModel):
                    positive_outputs, load_loss_pos = positive_outputs
                    load_loss += load_loss_pos
                contr_loss = contrastive_loss(features, positive_outputs, negative_indices)

            # Backward pass
            total_loss = classification_loss + contrastive_weight * contr_loss + load_loss
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]  # Get current learning rate
            stats = {
                'loss': total_loss.item(),
                'classification_loss': classification_loss.item(),
                'contrastive_loss': contr_loss.item(),
                'load_loss': load_loss.item(),
                'step': step,
                'lr': current_lr,  # Log current learning rate
                'epoch': epoch
            }
            wandb.log(stats)
            if step % 100 == 0:
                logger.info(f'Step {step}: {stats}')
            if (step % eval_every == 0 and step > 0) or step == 300:
                evaluate_classifier(model, train_val_dataset, device, 'train', step=step, epoch=epoch, bs=bs,
                                    num_samples=8000, dataloader=train_val_loader)
                evaluate_classifier(model, val_dataset, device, 'val', step=step, epoch=epoch, bs=bs,
                                    num_samples=-1, dataloader=val_dataloader)

                # Save the model
                torch.save(model.state_dict(), os.path.join(output_path, f'model_{step}.pt'))
                # Leave only 3
                for file in os.listdir(output_path):
                    if 'model_' in file and int(file.split('_')[1].split('.')[0]) < step - 3000:
                        os.remove(os.path.join(output_path, file))
            step += 1

    # Save the model via huggingface
    torch.save(model.state_dict(), os.path.join(output_path, 'model.pt'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL2-8B')
    arg_parser.add_argument('--output_path', type=str, required=True)
    arg_parser.add_argument('--lr', type=float, default=1.5e-4)
    arg_parser.add_argument('--wd', type=float, default=0.01)
    arg_parser.add_argument('--bs', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--freeze_vision', action='store_true')
    arg_parser.add_argument('--meta_train_path', type=str, default='../../../processing/meta_train_local.json')
    arg_parser.add_argument('--meta_valid_path', type=str, default='../../../processing/meta_valid_local.json')
    arg_parser.add_argument('--eval_only', action='store_true')
    arg_parser.add_argument('--load_checkpoint', type=str, default=None)
    arg_parser.add_argument('--no_contrastive', action='store_true')
    arg_parser.add_argument('--unfreeze_vit_layers', type=int, default=0)
    arg_parser.add_argument('--eval_every', type=int, default=2000)
    arg_parser.add_argument('--few_shot', action='store_true',
                            help='Enable few-shot learning mode')
    arg_parser.add_argument('--shots_per_class', type=int, default=8,
                            help='Number of samples per class in few-shot mode')
    arg_parser.add_argument('--all_separate', action='store_true', help='Use all separate model')
    arg_parser.add_argument('--test_modality', type=str, default=None,
                            help='Modality to test on (e.g., mammography, chest x-ray)')
    arg_parser.add_argument('--in_mod_pct', type=float, default=1.0,
                            help='Percentage of in-modality data to keep (0.0-1.0)')
    arg_parser.add_argument('--out_mod_pct', type=float, default=1.0,
                            help='Percentage of out-of-modality data to keep (0.0-1.0)')
    arg_parser.add_argument('--moe', type=int, default=-1,
                            help='Number of experts for MoE model')

    args = arg_parser.parse_args()
    train_classifier(
        model_path=args.model_path,
        output_path=args.output_path,
        lr=args.lr,
        wd=args.wd,
        bs=args.bs,
        epochs=args.epochs,
        freeze_vision=args.freeze_vision,
        meta_train_path=args.meta_train_path,
        meta_valid_path=args.meta_valid_path,
        eval_only=args.eval_only,
        load_checkpoint=args.load_checkpoint,
        no_contrastive=args.no_contrastive,
        unfreeze_vit_layers=args.unfreeze_vit_layers,
        few_shot=args.few_shot,
        shots_per_class=args.shots_per_class,
        all_separate=args.all_separate,
        eval_every=args.eval_every,
        test_modality=args.test_modality,
        in_mod_pct=args.in_mod_pct,
        out_mod_pct=args.out_mod_pct,
        moe=args.moe
    )
