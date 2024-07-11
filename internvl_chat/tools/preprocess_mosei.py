import json

import h5py
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import pandas as pd
import tqdm
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def extract_number(filename):
    return int(filename.split('.')[0])


def sample_grid(video_path, grid_path):
    """
    Uniformly sample 4 frames from the video, arrange them in a 2x2 grid.
    Save the grid as an image to the grid_path.
    @param video_path: Path to the video file.
    @param grid_path: Path to save the grid image.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Error: Could not open video file {video_path}.")
        return

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all frames
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Error: Could not read frame {i}.")
            cap.release()
            break

        # Append the frame to the list
        frames.append(frame)


    # Sample 4 frames uniforml
    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
    frames = [frames[i] for i in indices]

    # Create a 2x2 grid of frames
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top_row, bottom_row))

    # Save the grid as an image
    cv2.imwrite(grid_path, grid)

    # Release the video capture object
    cap.release()


def extract_audio(video_path, audio_path):
    """
    Extract audio from the video and save it as a WAV file.
    @param video_path: Path to the video file.
    @param audio_path: Path to save the audio file.
    """
    # Load the video clip
    clip = VideoFileClip(video_path)

    # Extract the audio
    clip.audio.write_audiofile(audio_path)

    # Close the clip
    clip.close()


def map_sentiment(sentiment: int):
    if sentiment <= -1:
        return "negative"
    elif sentiment >= 1:
        return "positive"
    elif sentiment == 0:
        return "neutral"
    elif sentiment < 0:
        return "weakly negative"
    elif sentiment > 0:
        return "weakly positive"

def preprocess_mosei():
    label_path = 'CMU_MOSEI_Labels.csd'
    mosei_path = '/home/dvd/data/datasets/cmu_mosei/'
    video_path = '/home/dvd/data/datasets/cmu_mosei/Raw/'
    image_path = '/home/dvd/data/datasets/cmu_mosei/images/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    audio_path = '/home/dvd/data/datasets/cmu_mosei/audio/'
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    csv_label_path = 'mosei_label.csv'
    with open(csv_label_path, 'r') as f:
        csv_label = pd.read_csv(f)  # video_id,clip_id,text,label,annotation,mode,label_T,label_A,label_V

    question_prefix = '<image>\n<audio>\nAbove are 4 frames and an audio clip from a video. '
    transcription_prefix = 'The speaker said: \''
    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']

    annotations = []

    with h5py.File(label_path, 'r') as f:
        data = f['All Labels']['data']

        video_names = data.keys()
        for video_name in tqdm.tqdm(video_names):
            labels = data[video_name]['features'][:]  # (num_clips, 7)
            # Label format: [sentiment (-3 to 3), happy (0 to 3), sad (0 to 3), anger (0 to 3),
            # surprise (0 to 3), disgust (0 to 3), fear (0 to 3)]
            this_video_path = os.path.join(video_path, video_name)
            if not os.path.exists(this_video_path):
                logging.warning(f"Video {video_name} not found.")
                continue
            clip_names = os.listdir(this_video_path)
            clip_names.sort(key=extract_number)
            for clip_name, label in zip(clip_names, labels):
                # print(clip_name, label)
                clip_id = int(clip_name.split('.')[0])
                basename = f"{video_name}_{clip_name.split('.')[0]}"

                clip_path = os.path.join(this_video_path, clip_name)
                grid_path = os.path.join(image_path, f"{basename}.jpg")
                if not os.path.exists(grid_path):
                    logging.info(f"Sampling grid for {clip_path}.")
                    sample_grid(clip_path, grid_path)

                audio_name = f"{basename}.wav"
                this_audio_path = os.path.join(audio_path, audio_name)
                if not os.path.exists(this_audio_path):
                    logging.info(f"Extracting audio from {clip_path}.")
                    try:
                        extract_audio(clip_path, this_audio_path)
                    except Exception as e:
                        logging.warning(f"Error: Could not extract audio from {clip_path}. Error: {e}")
                        continue

                sentiment = label[0]
                sentiment_str = map_sentiment(sentiment)

                csv_entry = csv_label[(csv_label['video_id'] == video_name) & (csv_label['clip_id'] == clip_id)]
                transcription = csv_entry['text'].values[0]

                sentiment_suffix = ('What is the sentiment of the speaker in this video?\n'
                                    'negative\nweakly negative\nneutral\nweakly positive\npositive\n'
                                    'Answer with one word or phrase.')
                question_with_transcription = question_prefix + transcription_prefix + transcription + '\' ' + sentiment_suffix
                question_without_transcription = question_prefix + sentiment_suffix

                annotations.append({
                    'id': len(annotations),
                    'image': f'images/{basename}.jpg',
                    'audio': f'audio/{audio_name}',
                    'conversations': [
                        {'from': 'human', 'value': question_with_transcription},
                        {'from': 'gpt', 'value': sentiment_str}
                    ]
                })

                annotations.append({
                    'id': len(annotations),
                    'image': f'images/{basename}.jpg',
                    'audio': f'audio/{audio_name}',
                    'conversations': [
                        {'from': 'human', 'value': question_without_transcription},
                        {'from': 'gpt', 'value': sentiment_str}
                    ]
                })

                strongest_emotion_index = np.argmax(label[1:])
                strongest_emotion = emotions[strongest_emotion_index]
                strongest_emotion_value = label[1 + strongest_emotion_index]
                if strongest_emotion_value >= 1:
                    emotion_suffix = ('What is the emotion of the speaker in this video?\n'
                                      'happy\nsad\nanger\nsurprise\ndisgust\nfear\n'
                                      'Answer with one word or phrase.')
                    emo_question_with_transcription = question_prefix + transcription_prefix + transcription + '\' ' + emotion_suffix
                    emo_question_without_transcription = question_prefix + emotion_suffix

                    annotations.append({
                        'id': len(annotations),
                        'image': f'images/{basename}.jpg',
                        'audio': f'audio/{audio_name}',
                        'conversations': [
                            {'from': 'human', 'value': emo_question_with_transcription},
                            {'from': 'gpt', 'value': strongest_emotion}
                        ]
                    })

                    annotations.append({
                        'id': len(annotations),
                        'image': f'images/{basename}.jpg',
                        'audio': f'audio/{audio_name}',
                        'conversations': [
                            {'from': 'human', 'value': emo_question_without_transcription},
                            {'from': 'gpt', 'value': strongest_emotion}
                        ]
                    })

                for emotion, value in zip(emotions, label[1:]):
                    if value >= 1:
                        binary_emotion_suffix = ('Is the speaker feeling ' + emotion + ' in this video?\n'
                                                'yes\nno\n'
                                                'Answer with one word or phrase.')
                        binary_emo_question_with_transcription = question_prefix + transcription_prefix + transcription + '\' ' + binary_emotion_suffix
                        binary_emo_question_without_transcription = question_prefix + binary_emotion_suffix

                        annotations.append({
                            'id': len(annotations),
                            'image': f'images/{basename}.jpg',
                            'audio': f'audio/{audio_name}',
                            'conversations': [
                                {'from': 'human', 'value': binary_emo_question_with_transcription},
                                {'from': 'gpt', 'value': 'yes'}
                            ]
                        })

                        annotations.append({
                            'id': len(annotations),
                            'image': f'images/{basename}.jpg',
                            'audio': f'audio/{audio_name}',
                            'conversations': [
                                {'from': 'human', 'value': binary_emo_question_without_transcription},
                                {'from': 'gpt', 'value': 'yes'}
                            ]
                        })

    with open(os.path.join(mosei_path, 'annotation_train.jsonl'), 'w') as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + '\n')


if __name__ == '__main__':
    preprocess_mosei()
