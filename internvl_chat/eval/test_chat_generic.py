import json
import os
from collections import defaultdict
from typing import List

import torch
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLChatModel
from torchmetrics import Accuracy, F1Score
from internvl.train.internvl_chat_audio_finetune import LazySupervisedDataset, concat_pad_data_collator
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.model.internvl_chat import InternVLChatModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import argparse


AUDIO_START_TOKEN = '<audio>'
AUDIO_END_TOKEN = '</audio>'
AUDIO_CONTEXT_TOKEN = '<AUDIO_CONTEXT>'


def main(meta_path, dataset_name, path):
    print(f'Path: {path}')

    # load in bfloat16
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
    # model = model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if 'OpenGVLab' in path:  # vanilla model
        tokenizer.add_tokens([AUDIO_START_TOKEN, AUDIO_END_TOKEN, AUDIO_CONTEXT_TOKEN], special_tokens=True)
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.audio.load_state_dict(torch.load('audio.pth'), strict=False)

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
    if 'What is the emotion of the speaker in this video?\n' in dataset[0]['question']:
        sample = (dataset[0]['question'].split('What is the emotion of the speaker in this video?\n')[1]
                  .split('\nAnswer with one word or phrase.')[0])
        categories = sample.split('\n')
    elif 'yes\nno\n' in dataset[0]['question']:
        categories = ['yes', 'no']
    else:
        raise ValueError('Unexpected question format')
    counts = defaultdict(int)
    categories.append('unknown')
    metrics = {
        'category_accuracy': Accuracy(task='multiclass', num_classes=len(categories), average='none'),
        'overall_accuracy': Accuracy(task='multiclass', num_classes=len(categories), average='micro'),
        'f1': F1Score(task='multiclass', num_classes=len(categories), average='macro')
    }


    # Iterate over data
    output_path = '/home/dvd/data/depression_interview/behavioral_output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for index, sample in enumerate(dataloader):
        if '<image>\n<audio>' not in sample['question'][0]:
            continue
        if 'The speaker said' not in sample['question'][0]:
            continue
        # move to cuda
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].to('cuda')

        if 'audio' in sample:
            audio_info = {
                "input_audios": sample['audios'],
                "audio_span_tokens": sample['audio_span_tokens'],
                "input_audio_lengths": sample['input_audio_lengths'],
            }
        else:
            audio_info = None

        question = sample['question'][0]
        # question = question.replace('Answer with one word or phrase.',
        #                             'Provide an explanation first. Then answer with one word or phrase from the following: yes/no.')

        response = model.chat(tokenizer, sample['pixel_values'], question,
                              generation_config=generation_config,
                              audio_info=audio_info)
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
    for metric_name, metric in metrics.items():
        metric_computed = metric.compute()
        if metric_name == 'category_accuracy':
            for i, category in enumerate(categories):
                print(f'{category}: {metric_computed[i]}')
        else:
            print(f'{metric_name}: {metric_computed}')
    print(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription', action='store_true')
    args = parser.parse_args()

    # path = "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"  # Vanilla model
    # path = "OpenGVLab/InternVL2-4B"  # Vanilla model
    path = "OpenGVLab/InternVL2-26B"  # Vanilla model
    # path = "/home/data/outputs/all_public_backbone_2"  # Backbone trained on public
    # path = "/home/data/outputs/phq9_lora"  # Pretrained on PHQ9 with lora
    # path = "/home/dvd/data/outputs/phq9_pretrain_nonlora/"  # Pretrained on PHQ9 without lora *
    # path = '/home/dvd/data/outputs/behavioral_pretrain'  # Pretrained on behavioral
    # path = '/home/dvd/data/outputs/both_phq9'  # Pretrained on both PHQ9 and behavioral
    # path = '/home/dvd/data/outputs/phq9_full_pretrain_nonlora'
    # path = '/home/dvd/data/outputs/phq9_full_pretrain_lora'
    # path = '/home/dvd/data/outputs/phq9_on_vanilla_lora/'  # Pretrained on PHQ9 on vanilla model with lora
    # path = '/home/dvd/data/outputs/phq9_binary_pretrain'
    # path = '/home/dvd/data/outputs/phq9_binary_lora'
    # path = '/home/dvd/data/outputs/phq9_binary_pretrain_on_vanilla_8B'
    # path = '/home/data/outputs/all_public_backbone_3'
    print(f'Path: {path}')
    main(meta_path='shell/data/behavioral_val.json',
         dataset_name='behavioral_phq_binary',
         path=path)
