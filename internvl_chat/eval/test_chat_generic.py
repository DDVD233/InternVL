import ujson as json
import os
from collections import defaultdict
from typing import List, Literal

import torch
from transformers import AutoTokenizer
from models.InternVL.internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from torchmetrics import Accuracy, F1Score, Specificity, Recall
from models.InternVL.internvl_chat.internvl.train.internvl_chat_finetune import LazySupervisedDataset, \
    concat_pad_data_collator
from models.InternVL.internvl_chat.internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                                                    IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                                                    IMG_START_TOKEN, QUAD_END_TOKEN,
                                                                    QUAD_START_TOKEN, REF_END_TOKEN,
                                                                    REF_START_TOKEN)
from transformers.modeling_outputs import CausalLMOutputWithPast
# prog bar
from tqdm import tqdm

import argparse


def test_model(meta_path, dataset_name, path, modality='all'):
    print(f'Path: {path}')

    # load in bfloat16

    if '26B' in path:
        device_map = {
            'vision_model': 1,
            'mlp1': 1,
            'mlp2': 1,
            'language_model': 0,
        }
        model: InternVLChatModel = InternVLChatModel.from_pretrained(
            path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        model: InternVLChatModel = model.eval()
    else:
        model: InternVLChatModel = InternVLChatModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    if '26B' not in path:
        model = model.to('cuda')

    with open(meta_path, 'r') as f:
        ds_collections = json.load(f)

    template_name = 'internlm2-chat' if ('8B' in path or '26B' in path) else 'phi3-chat'

    dataset = LazySupervisedDataset(
        template_name=template_name,
        meta=ds_collections[dataset_name],
        tokenizer=tokenizer,
        tcs_loader=None,
        ds_name=dataset_name,
        num_image_token=model.num_image_token,
        image_size=448,
        is_train=False,
        pad2square=False,
        group_by_length=True,
        dynamic_image_size=True,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=4,
        repeat_time=1,
        normalize_type="imagenet",
    )
    generation_config = dict(
        num_beams=1,
        max_new_tokens=50,
        do_sample=False,
    )
    dataset.ds_name = dataset_name
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                             collate_fn=concat_pad_data_collator)
    length = len(dataset)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    categories = []
    # task: Literal["binary", "multiclass", "multilabel"] = 'multiclass'
    for index, sample in enumerate(dataset.raw_data):
        answer = json.loads(sample)['conversations'][1]['value']
        if ',' in answer:  # this is multilabel
            answers = answer.split(',')
            answers = [a.strip().lower() for a in answers]
            categories.extend(answers)
        else:
            categories.append(answer.lower())
    categories = list(set(categories))
    print(f'Categories: {categories}')
    counts = defaultdict(int)
    metrics = {
        'category_specificity': Specificity(task='multilabel', num_labels=len(categories), average='none'),
        'overall_specificity': Specificity(task='multilabel', num_labels=len(categories), average='micro'),
        'category_sensitivity': Recall(task='multilabel', num_labels=len(categories), average='none'),
        'overall_sensitivity': Recall(task='multilabel', num_labels=len(categories), average='micro'),
    }

    for index, sample in enumerate(dataloader):
        # move to cuda
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].to('cuda')

        question = sample['question'][0]

        response = model.chat(tokenizer, sample['pixel_values'], question,
                              generation_config=generation_config)
        pred_tensor = torch.zeros(len(categories))
        if ',' in response:
            answers = response.split(',')
            answers = [a.strip().lower() for a in answers]
            for answer in answers:
                for category in categories:
                    if answer.endswith(category):
                        pred_tensor[categories.index(category)] = 1
        else:
            for category in categories:
                if response.lower().endswith(category):
                    pred_tensor[categories.index(category)] = 1
        target = sample['target'][0]
        if ',' in target:
            targets = target.split(',')
            targets = [a.strip().lower() for a in targets]
            target_tensor = torch.zeros(len(categories))
            for target in targets:
                for category in categories:
                    if target.endswith(category):
                        target_tensor[categories.index(category)] = 1
        else:
            target_category = categories.index(target.lower())
            target_tensor = torch.zeros(len(categories))
            target_tensor[target_category] = 1
        counts[target] += 1
        print(f'Predicted: {response}, Target: {target}. The answer is correct: {response == target}')
        for metric in metrics.values():
            metric(pred_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
        if index % 10 == 0:
            print(f'Processed {index}/{length} samples')
            # print metrics
            for metric_name, metric in metrics.items():
                metric_computed = metric.compute()
                if 'category' in metric_name:
                    submetric_name = metric_name.split('_')[1]
                    for i, category in enumerate(categories):
                        print(f'{category} {submetric_name}: {metric_computed[i]}')
                else:
                    print(f'{metric_name}: {metric_computed}')

    # print final metrics
    computed_metrics = {}
    for metric_name, metric in metrics.items():
        metric_computed = metric.compute()
        if 'category' in metric_name:
            submetric_name = metric_name.split('_')[1]
            for i, category in enumerate(categories):
                print(f'{submetric_name}/{category}: {metric_computed[i]}')
                computed_metrics[f'{submetric_name}/{category}'] = metric_computed[i].item()
        else:
            print(f'{metric_name}: {metric_computed}')
            computed_metrics[metric_name] = metric_computed.item()
    return computed_metrics, counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    path = 'OpenGVLab/InternVL2-8B'
    print(f'Path: {path}')
    meta_path = 'processing/meta_valid.json'
    with open(meta_path, 'r') as f:
        ds_collections = json.load(f)
    datasets = list(ds_collections.keys())
    dataset_metrics = {}
    for dataset in datasets:
        print(f'Dataset: {dataset}')
        metrics, counts = test_model(meta_path=meta_path,
                                     dataset_name=dataset,
                                     path=path,
                                     modality='all')
        dataset_metrics[dataset] = metrics
        for key, value in counts.items():
            dataset_metrics[dataset][f"{key}_count"] = value

        print(dataset_metrics)
