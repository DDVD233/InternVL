import base64
import os
from collections import defaultdict
from typing import List

import numpy
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor, CLIPProcessor
from transformers import AutoTokenizer
from datasets.IemocapDataset import IemocapDataset
from internvl.model.internvl_chat import InternVLChatModel
from torchmetrics import Accuracy, F1Score
from utils.audio_utils import process_audio
import openai
import cv2

import argparse
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import io

GPT_MODEL = "gpt-4o"


def base64_encode_image(image):
    return base64.b64encode(image).decode('utf-8')


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


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

    # read the API key from the environment
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # create a model object

    counts = defaultdict(int)
    for index, sample in enumerate(tqdm(data_loader)):
        counts[sample['emotion_str'][0]] += 1
    print(counts)

    for index, sample in enumerate(data_loader):
        message_base = [
            {"role": "system", "content": "You are a helpful assistant. Answer the questions concisely."}
        ]

        frame: numpy.ndarray = numpy.array(sample['raw_frames'][0])  # 2H, 2W, C
        image = Image.fromarray(frame)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        frame_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # decoded = base64.b64decode(frame_encoded)
        # with open('test.jpg', 'wb') as file:
        #     file.write(decoded)

        question = "These are 4 frames from a video. "
        if transcription_on:
            question += "The speaker said: \"" + sample['transcription'][0] + "\" "
        question += ("What is the emotion of the speaker? "
                     "Using both the frames and the transcription, answer with one word from the following: "
                     "happy, sad, neutral, angry, excited, and frustrated.")

        question_llm = {"role": "user", "content": [
            *map(lambda x: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + x}}, [frame_encoded]),
            {"type": "text", "text": question},
        ]
        }
        message_base.append(question_llm)

        chat_response = chat_completion_request(message_base)
        while chat_response.status_code != 200:
            print("Retrying")
            chat_response = chat_completion_request(message_base)

        assistant_message = chat_response.json()["choices"][0]["message"]
        response = assistant_message['content'].lower()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription', action='store_true')
    args = parser.parse_args()
    main(transcription_on=args.transcription)
