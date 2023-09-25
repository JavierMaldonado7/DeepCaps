import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video, VideoMetaData


class OpenImagesDataset(Dataset):
    def __init__(self, data_dir, split):
        assert split in ["train", "test"]
        self.data_dir = os.path.join(data_dir, split)
        self.annotations_path = os.path.join(data_dir, f"{split}.csv")
        self.annotations = pd.read_csv(self.annotations_path)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        image_path = os.path.join(self.data_dir, row["ImageID"] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        label = row["LabelName"]
        return image, label

    def __len__(self):
        return len(self.annotations)


class Youtube8MDataset(Dataset):
    def __init__(self, data_dir, split):
        assert split in ["train", "test"]
        self.data_dir = os.path.join(data_dir, split)
        self.annotations_path = os.path.join(data_dir, f"{split}.csv")
        self.annotations = pd.read_csv(self.annotations_path)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        video_path = os.path.join(self.data_dir, row["VideoID"] + ".avi")

        # Read video frames using torchvision.io.read_video function
        video, _, _ = read_video(video_path)

        # Transpose the video tensor from (t, h, w, c) to (c, t, h, w) format expected by PyTorch models
        video = video.permute(3, 0, 1, 2)

        # Return video tensor and label
        label = row["Label"]
        return video, label

    def __len__(self):
        return len(self.annotations)


class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Read in all of the video files in the directory
        for video_file in os.listdir(data_dir):
            if video_file.endswith('.mp4'):
                # Extract the label from the file name
                label = int(video_file.split('_')[1])

                # Add a sample to the list of samples
                self.samples.append((os.path.join(data_dir, video_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Load the video frames as a list of PIL images
        video_path, label = self.samples[index]
        video_frames = []
        for frame_file in os.listdir(video_path):
            if frame_file.endswith('.jpg'):
                frame = Image.open(os.path.join(video_path, frame_file))
                video_frames.append(frame)

        # Apply any transforms to the video frames
        if self.transform:
            video_frames = [self.transform(frame) for frame in video_frames]

        # Stack the video frames into a 4D tensor
        video_tensor = torch.stack(video_frames, dim=0)

        return video_tensor, label
