import os

import cv2
import numpy
import torch
import torchaudio
import pandas as pd
import numpy as np
import torch.nn.functional as F
from PIL import Image

from utils.video_utils import sample_frames, process_image, make_grid


class IemocapDataset(object):
    """
        Create a Dataset for Iemocap. Each item is a tuple of the form:
        (waveform, sample_rate, emotion, activation, valence, dominance)
    """

    _ext_audio = '.wav'
    _emotions = {'ang': 0, 'hap': 1, 'exc': 1, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8}
    _emotion_str = {0: 'angry', 1: 'happy', 2: 'excited', 3: 'sad', 4: 'frustrated', 5: 'fearful', 6: 'surprised',
                    7: 'neutral', 8: 'unknown'}

    def __init__(self,
                 root='IEMOCAP_full_release',
                 emotions=['ang', 'hap', 'exc', 'sad', 'neu', 'fru'],
                 sessions=[1, 2, 3, 4, 5],
                 script_impro=['script', 'impro'],
                 genders=['M', 'F']):
        """
        Args:
            root (string): Directory containing the Session folders
        """
        self.root = root

        # Iterate through all 5 sessions
        data = []
        transcriptions = {}
        for i in range(1, 6):
            transcription_path = os.path.join(root, 'Session' + str(i), 'dialog', 'transcriptions')
            # Get list of transcription files
            transcription_files = [file for file in os.listdir(transcription_path) if file.endswith('.txt')]
            for file in transcription_files:
                with open(os.path.join(transcription_path, file), 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Split the line into parts on the ']: ' which
                        # marks the end of the key and the start of the transcription
                        parts = line.split(']: ')

                        # Check if the line was split into exactly two parts
                        if len(parts) == 2:
                            # The first part is further split by space to separate the key from the timestamp
                            key = parts[0].split(' [')[0]
                            # The second part is the transcription text
                            transcription = parts[1]

                            # Store the key and transcription in the dictionary
                            transcriptions[key] = transcription

            # Define path to evaluation files of this session
            path = os.path.join(root, 'Session' + str(i), 'dialog', 'EmoEvaluation')

            # Get list of evaluation files
            files = [file for file in os.listdir(path) if file.endswith('.txt')]

            # Iterate through evaluation files to get utterance-level data
            for file in files:
                # Open file
                f = open(os.path.join(path, file), 'r')

                # Get list of lines containing utterance-level data. Trim and split each line into individual string elements.
                data += [line.strip()
                             .replace('[', '')
                             .replace(']', '')
                             .replace(' - ', '\t')
                             .replace(', ', '\t')
                             .split('\t')
                         for line in f if line.startswith('[')]

        # Get session number, script/impro, speaker gender, utterance number
        data = [d + [d[2][4], d[2].split('_')[1], d[2][-4], d[2][-3:]] for d in data]

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['start', 'end', 'file', 'emotion', 'activation', 'valence', 'dominance', 'session', 'script_impro', 'gender', 'utterance'])
        # convert start and end to float
        self.df['start'] = self.df['start'].astype(np.float32)
        self.df['end'] = self.df['end'].astype(np.float32)
        # convert session to int
        self.df['session'] = self.df['session'].astype(np.int32)
        # Convert activation, valence, dominance to float
        self.df['activation'] = self.df['activation'].astype(np.float32)
        self.df['valence'] = self.df['valence'].astype(np.float32)
        self.df['dominance'] = self.df['dominance'].astype(np.float32)
        # Convert utterance to int
        self.df['utterance'] = self.df['utterance'].astype(np.int32)


        # Filter by emotions
        filtered_emotions = self.df['emotion'].isin(emotions)
        self.df = self.df[filtered_emotions]

        # Filter by sessions
        filtered_sessions = self.df['session'].isin(sessions)
        self.df = self.df[filtered_sessions]

        # Filter by script_impro
        filtered_script_impro = self.df['script_impro'].str.contains('|'.join(script_impro))
        self.df = self.df[filtered_script_impro]

        # Filter by gender
        filtered_genders = self.df['gender'].isin(genders)
        self.df = self.df[filtered_genders]

        # Reset indices
        self.df = self.df.reset_index()

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.float32)

        # Map file to correct path w.r.t to root
        self.df['audio_file'] = [os.path.join('Session' + file[4], 'sentences', 'wav', file[:-5], file + self._ext_audio) for file in self.df['file']]
        self.df['video_file'] = [os.path.join('Session' + file[4], 'dialog', 'avi', 'DivX', file[:-5] + '.avi') for file in self.df['file']]
        self.df['transcription'] = [transcriptions[file] for file in self.df['file']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.df):
            return None

        audio_name = os.path.join(self.root, self.df.loc[idx, 'audio_file'])
        video_name = os.path.join(self.root, self.df.loc[idx, 'video_file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        emotion = self.df.loc[idx, 'emotion']
        emotion_str = self._emotion_str[int(emotion)]
        activation = self.df.loc[idx, 'activation']
        valence = self.df.loc[idx, 'valence']
        dominance = self.df.loc[idx, 'dominance']
        start = self.df.loc[idx, 'start']
        end = self.df.loc[idx, 'end']
        frames = sample_frames(video_name, num_frames=4, start=start, end=end)
        grid = make_grid(frames)
        grid: numpy.ndarray = np.array(grid)
        # downsize the grid to 1280x720 maximum, but keep the aspect ratio
        if grid.shape[0] > 720 or grid.shape[1] > 1280:
            aspect_ratio = grid.shape[1] / grid.shape[0]
            if aspect_ratio > 1280 / 720:
                new_width = 1280
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = 720
                new_width = int(new_height * aspect_ratio)
            grid = cv2.resize(grid, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frames = process_image(Image.fromarray(grid))
        transcription = self.df.loc[idx, 'transcription']

        sample = {
            'audio_path': audio_name,
            'video_path': video_name,
            'transcription': transcription,
            'frames': frames,
            'raw_frames': grid,
            'start': start,
            'end': end,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion,
            'emotion_str': emotion_str,
            'activation': activation,
            'valence': valence,
            'dominance': dominance
        }

        return sample

    def collage_fn_vgg(self, batch):
        # Clip or pad each utterance audio into 4.020 seconds.
        sample_rate = 16000
        n_channels = 1
        frame_length = np.int(4.020 * sample_rate)

        # Initialize output
        waveforms = torch.zeros(0, n_channels, frame_length)
        emotions = torch.zeros(0)

        for item in batch:
            waveform = item['waveform']
            original_waveform_length = waveform.shape[1]
            padded_waveform = F.pad(waveform, (0, frame_length - original_waveform_length)) if original_waveform_length < frame_length else waveform[:, :frame_length]
            waveforms = torch.cat((waveforms, padded_waveform.unsqueeze(0)))
            emotions = torch.cat((emotions, torch.tensor([item['emotion']])), 0)

        return waveforms, emotions

    def collate_fn_segments(self, batch):
        # Segment each sample into 264ms frames and 25ms sliding window
        sample_rate = 16000
        segment_length = np.int(0.264 * sample_rate)
        step_length = np.int(0.025 * sample_rate)

        # Initialize output
        segments = torch.zeros(0, segment_length)
        n_segments = torch.zeros(0)
        emotions = torch.zeros(0)
        filenames = []

        # Iterate through samples in batch
        for item in batch:
            waveform = item['waveform']
            original_waveform_length = waveform.shape[1]

            # Compute number of segments given input waveform, segment, and step lengths
            item_n_segments = np.int(np.ceil((original_waveform_length - segment_length) / step_length) + 1)

            # Compute and apply padding to waveform
            padding_length = segment_length - original_waveform_length if original_waveform_length < segment_length else (segment_length + (item_n_segments - 1) * step_length - original_waveform_length)
            padded_waveform = F.pad(waveform, (0, padding_length))
            padded_waveform = padded_waveform.view(-1)

            # Construct tensor of segments
            item_segments = torch.zeros(item_n_segments, segment_length)
            for i in range(item_n_segments):
                item_segments[i] = padded_waveform[i*step_length:i*step_length+segment_length]
            segments = torch.cat((segments, item_segments), 0)

            # Construct tensor of emotion labels
            emotion = torch.tensor([item['emotion']])
            emotions = torch.cat((emotions, emotion.repeat(item_n_segments)), 0)

            # Construct list of
            filenames += [item['path'].split('/')[-1]]*item_n_segments

            # Construct tensor of n_frames (contains a list of number of frames per item)
            item_n_segments = torch.tensor([float(item_n_segments)])
            n_segments = torch.cat((n_segments, item_n_segments), 0)

        return segments, emotions, n_segments, filenames

    def collate_fn(self, batch):
        # Frame each sample into 25ms frames and 10ms sliding (step) window.
        # This means that the frame length for a 16kHz signal is 0.025 * 16000 = 400 samples.
        # Frame step is usually something like 10ms (160 samples), which allows some overlap to the frames.
        # The first 400 sample frame starts at sample 0, the next 400 sample frame starts at sample 160 etc until the end of the speech file is reached.
        # If the speech file does not divide into an even number, pad it with zeros so that it does.
        sample_rate = 16000
        # n_channels = 1
        frame_length = np.int(0.025 * sample_rate)
        step_length = np.int(0.01 * sample_rate)

        # Initialize output
        # frames = torch.zeros(0, n_channels, frame_length)
        frames = torch.zeros(0, frame_length)
        n_frames = torch.zeros(0)
        emotions = torch.zeros(0)

        for item in batch:
            waveform = item['waveform']
            original_waveform_length = waveform.shape[1]

            # Compute number of frames given input waveform, frame and step lengths
            item_n_frames = np.int(np.ceil((original_waveform_length - frame_length) / step_length) + 1)

            # Compute and apply padding to waveform
            padding_length = frame_length - original_waveform_length if original_waveform_length < frame_length else (frame_length + (item_n_frames - 1) * step_length - original_waveform_length)
            padded_waveform = F.pad(waveform, (0, padding_length))
            padded_waveform = padded_waveform.view(-1)

            # Construct tensor of frames
            # item_frames = torch.zeros(n_frames, n_channels, frame_length)
            item_frames = torch.zeros(item_n_frames, frame_length)
            for i in range(item_n_frames):
                item_frames[i] = padded_waveform[i*step_length:i*step_length+frame_length]
                # item_frames[i] = padded_waveform[:, i*step_length:i*step_length+frame_length]
            frames = torch.cat((frames, item_frames), 0)

            # Construct tensor of emotion labels
            emotion = torch.tensor([item['emotion']])
            emotions = torch.cat((emotions, emotion.repeat(item_n_frames)), 0)

            # Construct tensor of n_frames (contains a list of number of frames per item)
            item_n_frames = torch.tensor([float(item_n_frames)])
            n_frames = torch.cat((n_frames, item_n_frames), 0)

        return frames, emotions, n_frames

# Example: Load Iemocap dataset
# iemocap_dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')

# Example: Iterate through samples
# for i in range(len(iemocap_dataset)):
#     sample = iemocap_dataset[i]
#     print(i, sample)

# Number of audio by duration
# dataset_duration = np.ceil(iemocap_dataset.df['end'] - iemocap_dataset.df['start'])
# idx = np.where(dataset_duration == 35)
# durations = np.unique(dataset_duration)
# durations_count = [np.sum(dataset_duration == i) for i in durations]

# print('End')