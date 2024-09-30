import ujson as json
import os
from collections import defaultdict
from typing import List

import torch
from transformers import AutoTokenizer
from models.InternVL.internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from torchmetrics import Accuracy, F1Score
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
            'audio': 1,
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
    audio_context_token_id = tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    model.audio_context_token_id = audio_context_token_id

    # get categories
    # if 'What is the emotion of the speaker in this video?\n' in dataset[0]['question']:
    #     sample = (dataset[0]['question'].split('What is the emotion of the speaker in this video?\n')[1]
    #               .split('\nAnswer with one word or phrase.')[0])
    #     categories = sample.split('\n')
    # elif 'yes\nno\n' in dataset[0]['question']:
    #     categories = ['yes', 'no']
    # else:
    #     # Go through answers and get categories
    categories = []
    for index, sample in enumerate(dataset.raw_data):
        answer = json.loads(sample)['conversations'][1]['value']
        if answer not in categories:
            categories.append(answer)
    print(f'Categories: {categories}')
    counts = defaultdict(int)
    categories.append('unknown')
    metrics = {
        'category_accuracy': Accuracy(task='multiclass', num_classes=len(categories), average='none'),
        'overall_accuracy': Accuracy(task='multiclass', num_classes=len(categories), average='micro'),
        'f1': F1Score(task='multiclass', num_classes=len(categories), average='macro')
    }

    for index, sample in enumerate(dataloader):
        included_modality = []
        if 'mosei' in dataset_name and modality != 'all':
            if modality == 'image':
                sample['question'][0] = sample['question'][0].replace('<audio>\n', '')
                sample['question'][0] = sample['question'][0].split('The speaker said: \'')[0] + \
                                        sample['question'][0].split('\'')[-1]
                sample['audio_flags'] = torch.zeros_like(sample['audio_flags'])
            if modality == 'audio':
                sample['question'][0] = sample['question'][0].replace('<image>\n', '')
                sample['question'][0] = sample['question'][0].split('The speaker said: \'')[0] + \
                                        sample['question'][0].split('\'')[-1]
                sample['image_flags'] = torch.zeros_like(sample['image_flags'])
            if modality == 'text':
                sample['question'][0] = sample['question'][0].replace('<audio>\n', '').replace('<image>\n', '')
        if sample['image_flags'][0].item() == 1:
            included_modality.append('image')
        if sample['audio_flags'].size(0) > 1 and sample['audio_flags'][0].item() == 1:
            included_modality.append('audio')
        if 'The speaker said' in sample['question'][0]:
            included_modality.append('text')
        if modality == 'all' and len(included_modality) != 3:
            continue
        if modality == 'image' and ('image' not in included_modality or len(included_modality) > 1):
            continue
        if modality == 'audio' and ('audio' not in included_modality or len(included_modality) > 1):
            continue
        if modality == 'text' and ('text' not in included_modality or len(included_modality) > 1):
            continue
        # move to cuda
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].to('cuda')

        question = sample['question'][0]

        response = model.chat(tokenizer, sample['pixel_values'], question,
                              generation_config=generation_config)
        response_category = categories.index('unknown')
        for category in categories:
            if response.endswith(category):
                response_category = categories.index(category)
        # try:
        #     response_category = categories.index(response)
        # except ValueError:
        #     response_category = categories.index('unknown')
        target = sample['target'][0]
        target_category = categories.index(target)
        counts[target] += 1
        print(f'Predicted: {response}, Target: {target}. The answer is correct: {response == target}')
        response = torch.tensor([response_category])
        target = torch.tensor([target_category])
        for metric in metrics.values():
            metric(response, target)
        if index % 10 == 0:
            print(f'Processed {index}/{length} samples')
            # print metrics
            for metric_name, metric in metrics.items():
                metric_computed = metric.compute()
                if metric_name == 'category_accuracy':
                    for i, category in enumerate(categories):
                        print(f'{category}: {metric_computed[i]}')
                else:
                    print(f'{metric_name}: {metric_computed}')

    # print final metrics
    computed_metrics = {}
    for metric_name, metric in metrics.items():
        metric_computed = metric.compute()
        if metric_name == 'category_accuracy':
            for i, category in enumerate(categories):
                print(f'{category}: {metric_computed[i]}')
                computed_metrics[category] = metric_computed[i].item()
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
