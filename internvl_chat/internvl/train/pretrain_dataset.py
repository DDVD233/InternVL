import json
import os
from typing import Any, List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import logging

from internvl.train.dataset import build_transform


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    def __init__(
            self,
            meta_path: str,
            tokenizer: Any,
            image_size: int = 448,
            max_length: int = 77,
            frames_per_video: int = 4,  # Number of frames to extract from each video
            is_train: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.frames_per_video = frames_per_video

        # Load metadata
        self.image_groups = []  # List of lists of image paths or video paths
        self.captions = []
        self.datasets = []  # Track which dataset each entry belongs to
        self.is_video = []  # Track which entries are videos

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        for ds_name, ds_meta in meta.items():
            root = ds_meta['root']
            with open(ds_meta['annotation'], 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'images' in data:
                        this_images = [os.path.join(root, image) for image in data['images']]
                        self.image_groups.append(this_images)
                        self.is_video.append(False)
                    elif 'videos' in data and len(data['videos']) > 0:
                        video_path = os.path.join(root, data['videos'][0])
                        self.image_groups.append([video_path])  # Store as single-item list for consistency
                        self.is_video.append(True)
                    else:
                        continue

                    target = data['conversations'][1]['value'].lower()
                    modality = ds_name.split('_')[0]
                    modality = modality.replace('chest', 'chest x-ray').replace('mammo', 'mammography').replace('derm',
                                                                                                                'dermoscopy')
                    modality = modality.replace('mri', 'MRI').replace('ct', 'CT')
                    caption = modality + ', ' + target
                    caption = caption.replace('_', ' ')
                    self.captions.append(caption)

                    self.datasets.append(ds_name)

        self.unique_captions = list(set(self.captions))

        logger.info(f'Loaded {len(self.image_groups)} image/video groups for pretraining')
        self.transform = build_transform(
            is_train=is_train,
            input_size=image_size
        )

    def extract_video_frames(self, video_path: str) -> List[torch.Tensor]:
        """Extract frames from video and convert to tensors."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)

            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f'Empty video file: {video_path}')
                return [torch.zeros((3, self.image_size, self.image_size))]

            # Calculate frame indices to extract
            indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image and apply transforms
                    frame = Image.fromarray(frame)
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)
                else:
                    logger.error(f'Failed to read frame {idx} from video: {video_path}')
                    frames.append(torch.zeros((3, self.image_size, self.image_size)))

            cap.release()
            return frames

        except Exception as e:
            logger.error(f'Error processing video {video_path}: {e}')
            return [torch.zeros((3, self.image_size, self.image_size))] * self.frames_per_video

    def get_grid_dimensions(self, n_images: int) -> Tuple[int, int]:
        """
        Calculate the optimal grid dimensions for n images.
        Returns (rows, cols) that form the most square-like arrangement.
        """
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        return rows, cols

    def create_image_grid(self, images: List[torch.Tensor], rows: int, cols: int) -> torch.Tensor:
        """
        Create a grid of images with specified dimensions.
        Pads empty spaces with black if necessary.
        """
        n_images = len(images)
        c, h, w = images[0].shape

        # Create empty grid
        grid = torch.zeros((c, h * rows, w * cols))

        for idx, img in enumerate(images):
            i = idx // cols  # row
            j = idx % cols  # column
            grid[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = img

        return grid

    def load_and_transform_image(self, image_path: str) -> torch.Tensor:
        """Load and transform a single image, handling errors."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f'Error loading image {image_path}: {e}')
            return torch.zeros((3, self.image_size, self.image_size))

    def __len__(self) -> int:
        return len(self.image_groups)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        paths = self.image_groups[idx]
        caption = self.captions[idx]
        is_video = self.is_video[idx]

        if is_video:
            # Extract frames from video
            images = self.extract_video_frames(paths[0])
        else:
            # Load and transform all images in the group
            images = [self.load_and_transform_image(path) for path in paths]

        # Get optimal grid dimensions
        rows, cols = self.get_grid_dimensions(len(images))

        # Create image grid
        grid = self.create_image_grid(images, rows, cols)

        # Resize grid to original image size while maintaining aspect ratio
        if grid.shape[1] != self.image_size or grid.shape[2] != self.image_size:
            resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_size, antialias=True),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()
            ])
            grid = resize_transform(grid)

        return {
            'pixel_values': grid,
            'caption': caption,
            'num_images': len(images),
            'is_video': is_video,
            'dataset': self.datasets[idx]
        }
