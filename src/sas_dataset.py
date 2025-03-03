import torch
import pandas as pd
from collections import deque
from PIL import Image
from torch.utils.data import Dataset


class SasDataset(Dataset):
    def __init__(self, csv_file, transform=None, num_stacked_frames=4):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_stacked_frames (int): Number of stacked frames (default=4).
        )
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.num_stacked_frames = num_stacked_frames

    def __len__(self):
        return len(self.data) - self.num_stacked_frames

    def __getitem__(self, index):
        state_deque = deque(maxlen=self.num_stacked_frames)
        next_state_deque = deque(maxlen=self.num_stacked_frames)

        # First encoder's stacked frames: S_t, S_{t-1}, S_{t-2}, S_{t-3}
        for i in range(self.num_stacked_frames):
            frame_idx = max(index - i, 0)
            row = self.data.iloc[frame_idx]
            state_image = Image.open(row["state"])

            if self.transform:
                state_image = self.transform(state_image)

            state_deque.appendleft(state_image)

        # Second encoder's stacked frames with 75% overlap: S_{t+1}, S_t, S_{t-1}, S_{t-2}
        for i in range(self.num_stacked_frames):
            frame_idx = min(index + 1 - i, len(self.data) - 1)
            row = self.data.iloc[frame_idx]
            next_state_image = Image.open(row["next_state"])

            if self.transform:
                next_state_image = self.transform(next_state_image)

            next_state_deque.appendleft(next_state_image)

        # Stack frames across the channel dimension
        stacked_state = torch.cat(
            list(state_deque), dim=0
        )  # Shape: [num_stacked_frames * channels, H, W]
        stacked_next_state = torch.cat(list(next_state_deque), dim=0)

        # Get the last action and done flag in the sequence
        last_row = self.data.iloc[
            min(index + self.num_stacked_frames - 1, len(self.data) - 1)
        ]
        action = torch.tensor(last_row["action"], dtype=torch.long)
        done = torch.tensor(last_row["done"], dtype=torch.bool)

        return stacked_state, action, stacked_next_state, done
