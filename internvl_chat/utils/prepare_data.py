import os
from typing import List

import numpy
import torch
from PIL import Image
from datasets.IemocapDataset import IemocapDataset
import shutil
import ujson as json
import tqdm

import argparse


def prepare_iemocap():
    iemocap_dataset = IemocapDataset(
        '/home/dvd/data/depression_interview/dataset/IEMOCAP_full_release/IEMOCAP_full_release',
        sessions=[1, 2, 3, 4])
    save_path = '/home/data/datasets/iemocap'
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    audio_base_path = os.path.join(save_path, 'audio')
    os.makedirs(audio_base_path, exist_ok=True)
    annotations = []

    for i, data in enumerate(tqdm.tqdm(iemocap_dataset)): # (H, W, C)
        if data is None:
            break
        target: str = data['emotion_str']
        transcription: str = data['transcription']
        audio_path: str = data['audio_path']

        # Save image
        image_name = f"{i}.jpg"
        # skip if exists
        if not os.path.exists(os.path.join(image_path, image_name)):
            frames: numpy.ndarray = data['raw_frames']
            image = Image.fromarray(frames)
            image.save(os.path.join(image_path, image_name))

        # Save audio
        audio_name = f"{i}.wav"
        # skip if exists
        if not os.path.exists(os.path.join(audio_base_path, audio_name)):
            shutil.copy(audio_path, os.path.join(audio_base_path, audio_name))

        # transcription, image, audio
        annotations.append({
            "id": len(annotations),
            "image": f"images/{image_name}",
            "audio": f"audio/{audio_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Above are 4 frames and an audio clip from a video. "
                             f"The speaker said, '{transcription}' What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })

        # transcription, image
        annotations.append({
            "id": len(annotations),
            "image": f"images/{image_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Above are 4 frames from a video. "
                             f"The speaker said, '{transcription}' What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })

        # transcription, audio
        annotations.append({
            "id": len(annotations),
            "audio": f"audio/{audio_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Above is an audio clip from a video. "
                             f"The speaker said, '{transcription}' What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })

        # transcription
        annotations.append({
            "id": len(annotations),
            "conversations": [
                {
                    "from": "human",
                    "value": f"The speaker said, '{transcription}' What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })

        # image, audio
        annotations.append({
            "id": len(annotations),
            "image": f"images/{image_name}",
            "audio": f"audio/{audio_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Above are 4 frames and an audio clip from a video. What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })

        # image
        annotations.append({
            "id": len(annotations),
            "image": f"images/{image_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Above are 4 frames from a video. What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })

        # audio
        annotations.append({
            "id": len(annotations),
            "audio": f"audio/{audio_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Above is an audio clip from a video. What is the emotion of the speaker in this video?"
                },
                {
                    "from": "gpt",
                    "value": target
                }
            ]
        })


    with open(os.path.join(save_path, 'annotations.jsonl'), 'w') as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + '\n')


if __name__ == '__main__':
    prepare_iemocap()