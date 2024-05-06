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


def make_grid(images):
    # Reshape the input tensor to (B, 2, 2, H, W, C)
    images = images.view(images.shape[0], 2, 2, images.shape[2], images.shape[3], images.shape[4])

    # Swap the second and third dimensions to move the images next to each other
    images = images.permute(0, 3, 1, 4, 2, 5)

    # Reshape to (B, 2H, 2W, C)
    images = images.contiguous().view(images.shape[0], images.shape[3] * 2, images.shape[1] * 2, images.shape[5])

    return images


def main():
    # Initialize IemocapDataset
    iemocap_dataset = IemocapDataset(
        '/home/dvd/data/depression_interview/dataset/IEMOCAP_full_release/IEMOCAP_full_release',
        sessions=[5])
    data_loader = torch.utils.data.DataLoader(iemocap_dataset, batch_size=1, shuffle=True, pin_memory=True,
                                              num_workers=8)
    length = len(data_loader)
    path = "OpenGVLab/InternVL-Chat-V1-5-Int8"
    model = InternVLChatModel.from_pretrained(
        path,
        low_cpu_mem_usage=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    categories = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated', 'unknown']
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
        frames: List[numpy.ndarray] = sample['frames']  # 4, H, W, C
        # make 2x2 grid of frames, shape = 2H, 2W, C
        # frame = frames[0]
        # grid = make_grid(frames)

        # for index_f, frame in enumerate(frames):

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        question = ("This is a set of 4 frames from a video. What is the emotion of the speaker? "
                    "Answer with one word from the following: happy, sad, neutral, angry, excited, and frustrated.")
        response = model.chat(tokenizer, frames[0], question, generation_config)
        # print('-' * 50)
        target = sample['emotion_str'][0]
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
    main()
