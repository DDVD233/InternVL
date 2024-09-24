import os
from PIL import Image
import tqdm
import h5py
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import ujson as json
import copy
import random


def main():
    input_dir = '/home/dvd/data/datasets/meld/images'
    output_dir = '/home/dvd/data/datasets/meld/images_resized'
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    for filename in tqdm.tqdm(files):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            img = Image.open(input_path)
            new_size = (img.width // 2, img.height // 2)
            # Resize the image
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            # Construct the output file path
            output_path = os.path.join(output_dir, filename)
            # Save the resized image as a PNG
            img_resized.save(output_path, format='PNG')


def diversify_data(path, no_audio=True, no_image=True, no_audio_no_image=True):
    with open(path, 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    new_data = []
    for d in tqdm.tqdm(data):
        d['id'] = len(new_data)
        new_data.append(d)
        if no_audio:
            data_no_audio = copy.deepcopy(d)
            data_no_audio.pop('audio', None)
            data_no_audio['conversations'][0]['value'] = data_no_audio['conversations'][0]['value'].replace('<audio>\n',
                                                                                                            '')
            data_no_audio['id'] = len(new_data)
            new_data.append(data_no_audio)
        if no_image:
            data_no_image = copy.deepcopy(d)
            data_no_image.pop('image', None)
            data_no_image['conversations'][0]['value'] = data_no_image['conversations'][0]['value'].replace('<image>\n',
                                                                                                            '')
            data_no_image['id'] = len(new_data)
            new_data.append(data_no_image)
        if no_audio_no_image:
            data_no_audio_no_image = copy.deepcopy(d)
            data_no_audio_no_image.pop('audio', None)
            data_no_audio_no_image.pop('image', None)
            data_no_audio_no_image['conversations'][0]['value'] = data_no_audio_no_image['conversations'][0][
                'value'].replace('<audio>\n', '')
            data_no_audio_no_image['conversations'][0]['value'] = data_no_audio_no_image['conversations'][0][
                'value'].replace('<image>\n', '')
            data_no_audio_no_image['conversations'][0]['value'] \
                = (data_no_audio_no_image['conversations'][0]['value']
                   .replace('Above are 4 frames and an audio clip from a video. ', ''))

            data_no_audio_no_image['id'] = len(new_data)
            new_data.append(data_no_audio_no_image)
    with open(path.replace('.jsonl', '_diversified.jsonl'), 'w') as f:
        for d in new_data:
            f.write(json.dumps(d) + '\n')


def format_questions(path):
    with open(path, 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    new_data = []
    for d in tqdm.tqdm(data):
        target = d['conversations'][1]['value']
        # reduce neutral portion by 60%
        if target == 'neutral' and random.random() < 0.6:
            continue
        conversation = d['conversations'][0]['value']
        to_append = '\nanger\ndisgust\nfear\njoy\nneutral\nsadness\nsurprise\nAnswer with one word or phrase.'
        if to_append not in conversation:
            d['conversations'][0]['value'] += to_append
        new_data.append(d)
    with open(path.replace('_diversified.jsonl', '_formatted.jsonl'), 'w') as f:
        for d in new_data:
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    data_path = '/home/dvd/data/datasets/cmu_mosei/'
    diversify_data(os.path.join(data_path, 'annotation_train.jsonl'),
                   no_audio=True, no_image=True, no_audio_no_image=False)
    # diversify_data(os.path.join(data_path, 'annotations_no_transcript.jsonl'),
    #                no_audio=True, no_image=True, no_audio_no_image=False)
    # format_questions(os.path.join(data_path, 'annotations_with_transcript_diversified.jsonl'))
    # format_questions(os.path.join(data_path, 'annotations_no_transcript_diversified.jsonl'))
    # mosei_recover()
