import json
import os
import numpy as np
from PIL import Image
import tqdm


def remove_overlaps(captions):
    unique_captions = [captions[0]]  # Start with the first caption

    for i in range(1, len(captions)):
        previous_caption = unique_captions[-1]
        current_caption = captions[i]

        # Find the overlap between the previous caption and the current caption
        overlap_index = -1
        for j in range(len(previous_caption)):
            if current_caption.startswith(previous_caption[j:]):
                overlap_index = j
                break

        # Remove the overlap part from the current caption
        if overlap_index != -1:
            unique_part = current_caption[len(previous_caption) - overlap_index:]
            unique_captions.append(unique_part)
        else:
            unique_captions.append(current_caption)

    return unique_captions


def read_vtt(filename):
    captions = []
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Skip header and empty lines
    entries = [line.strip() for line in lines if line.strip() and not line.strip().startswith('WEBVTT')]

    # Process entries
    i = 0
    while i < len(entries):
        if '-->' in entries[i]:
            start_time, end_time = entries[i].split(' --> ')
            text = ''
            i += 1
            while i < len(entries) and '-->' not in entries[i]:
                text += entries[i].strip() + ' '
                i += 1
            if len(captions) > 0 and captions[-1] in text:
                captions[-1] = text.strip().lower()
            else:
                captions.append(text.strip().lower())
        else:
            i += 1

    if len(captions) == 0:
        return ''
    captions = remove_overlaps(captions)
    captions = [caption for caption in captions if caption.strip()]
    captions = remove_overlaps(captions)
    captions = [caption for caption in captions if caption.strip()]
    captions = ' '.join(captions)

    return captions


def preprocess_social_iq():
    root_path = '/home/dvd/data/datasets/Social-IQ-2.0-Challenge/siq2/'
    grid_path = os.path.join(root_path, 'images')
    annotation_path = os.path.join(root_path, 'qa', 'qa_train.json')  # This is actually jsonl
    new_annotation_path = os.path.join(root_path, 'annotations_train.jsonl')

    if not os.path.exists(grid_path):
        os.makedirs(grid_path)

    raw_data = []
    with open(annotation_path, 'r') as f:
        for line in f:
            raw_data.append(json.loads(line))

    # Preprocess data
    processed_data = []
    for data in tqdm.tqdm(raw_data):
        video_name = data['vid_name']
        frame_path = os.path.join(root_path, 'frames', video_name)
        if not os.path.exists(frame_path):
            print(f'Frames not found for {video_name}')
            continue
        frame_names = os.listdir(frame_path)
        # Sample 4 frames uniformly
        step = len(frame_names) // 4
        frame_names = [frame_names[i] for i in range(0, len(frame_names), step)]
        frame_names = frame_names[:4]
        assert len(frame_names) == 4
        # Make a 2x2 grid of images
        images = []
        # read images to numpy frames
        for frame_name in frame_names:
            image = Image.open(os.path.join(frame_path, frame_name))
            image = np.array(image)
            images.append(image)
        # stack images to 2x2 grid
        images = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])
        # save the grid
        save_grid_path = os.path.join(grid_path, f'{video_name}.png')
        Image.fromarray(images).save(save_grid_path)

        audio_path = os.path.join(root_path, 'audio', 'mp3', f'{video_name}.mp3')

        transcript_path = os.path.join(root_path, 'transcript', f'{video_name}.vtt')
        # read transcript
        captions = read_vtt(transcript_path)

        question = (f"<image>\n<audio>\nAbove are 4 frames and an audio clip from a video. "
                    f"The speaker said, '{captions}' "
                    f"{data['q']}\n"
                    f"A. {data['a0']}\n"
                    f"B. {data['a1']}\n"
                    f"C. {data['a2']}\n"
                    f"D. {data['a3']}\n"
                    f"Answer with A, B, C, or D.")
        answer_index = data['answer_idx']
        answer = ['A', 'B', 'C', 'D'][answer_index]
        processed_data.append({
            'id': len(processed_data),
            'image': os.path.relpath(save_grid_path, root_path),
            'audio': os.path.relpath(audio_path, root_path),
            'conversations': [
                {'from': 'human', 'value': question},
                {'from': 'gpt', 'value': answer}
            ]
        })

    with open(new_annotation_path, 'w') as f:
        for data in processed_data:
            f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    preprocess_social_iq()







