import os
from typing import List

import numpy
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPProcessor
from transformers import AutoTokenizer
from datasets.IemocapDataset import IemocapDataset
from internvl.model.internvl_chat import InternVLChatModel
from torchmetrics import Accuracy, F1Score
from utils.audio_utils import process_audio

import argparse


AUDIO_START_TOKEN = '<audio>'
AUDIO_END_TOKEN = '</audio>'
AUDIO_CONTEXT_TOKEN = '<AUDIO_CONTEXT>'


def main(transcription_on=False):
    # Print configs
    print(f'Transcription: {transcription_on}')

    # Initialize IemocapDataset
    iemocap_dataset = IemocapDataset(
        '/home/dvd/data/depression_interview/dataset/IEMOCAP_full_release/IEMOCAP_full_release',
        sessions=[5])
    data_loader = torch.utils.data.DataLoader(iemocap_dataset, batch_size=1, shuffle=True, pin_memory=True,
                                              num_workers=8)
    length = len(data_loader)
    # path = "OpenGVLab/InternVL-Chat-V1-5-Int8"
    # path = "OpenGVLab/InternVL-Chat-V1-5"
    # path = "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"
    # path = "/home/data/phi3_iemocap"
    path = "/home/dvd/data/phi3_backbone_lora/"

    device_map = {
        'audio': 1,
        'vision_model': 1,
        'mlp1': 1,
        'mlp2': 1,
        'language_model': 0,
    }
    # load in bfloat16
    model: InternVLChatModel = InternVLChatModel.from_pretrained(
        path,
        # low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        # device_map=device_map
    )
    model = model.eval()
    model = model.to('cuda')
    # model.template = 'internvl_zh'

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if 'OpenGVLab' in path:  # vanilla model
        tokenizer.add_tokens([AUDIO_START_TOKEN, AUDIO_END_TOKEN, AUDIO_CONTEXT_TOKEN], special_tokens=True)
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.audio.load_state_dict(torch.load('audio.pth'), strict=False)

    categories = ['happy', 'sad', 'neutral', 'angry', 'frustrated', 'unknown']
    metrics = {
        'category_accuracy': Accuracy(task='multiclass', num_classes=len(categories), average='none'),
        'overall_accuracy': Accuracy(task='multiclass', num_classes=len(categories), average='micro'),
        'f1': F1Score(task='multiclass', num_classes=len(categories), average='macro')
    }

    # Iterate over data
    output_path = '/home/dvd/data/depression_interview/iemocap_output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for index, sample in enumerate(data_loader):
        frames: torch.Tensor = sample['frames'][0]  # 2H, 2W, C
        audio_path = sample['audio_path'][0]
        processed_audio = process_audio(audio_path)

        generation_config = dict(
            num_beams=1,
            max_new_tokens=50,
            do_sample=False,
        )

        question = "Above are 4 frames and an audio clip from a video. "
        if transcription_on:
            question += "The speaker said, '" + sample['transcription'][0] + "'"
        question += (" What is the emotion of the speaker?"
                     "\nhappy\nsad\nneutral\nangry\nfrustrated\nAnswer with one word or phrase.")

        response: str = model.chat(tokenizer, frames.to(model.device), question, generation_config, audio_info=processed_audio)
        # print('-' * 50)
        target = sample['emotion_str'][0]
        if response.startswith(target):
            response = target
        print(f'Response: {response}, Target: {target}. The answer is correct: {response == target}')
        # print('-' * 50)
        # fill metrics
        try:
            response_category = categories.index(response)
        except ValueError:
            response_category = categories.index('unknown')
        target_category = categories.index(target)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription', action='store_true')
    args = parser.parse_args()
    main(transcription_on=args.transcription)
