import torch
import pandas as pd
from collections import deque
from PIL import Image
from torch.utils.data import Dataset


class SasDataset(Dataset):
    def __init__(
        self, csv_file, transform=None, convert_to_grayscale=False, num_stacked_frames=4
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            convert_to_grayscale (bool): Convert images to grayscale if True, else keep them as RGB.
            num_stacked_frames (int): Number of stacked frames (default=4).
        Usage:
            dataset = SasDataset(
                csv_file="../data/dataset.csv",
                transform=transforms.Compose([
                    transforms.Resize((80, 80)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ]),
                convert_to_grayscale=True,
                num_stacked_frames=4
        )
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.convert_to_grayscale = convert_to_grayscale
        self.num_stacked_frames = num_stacked_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        state_deque = deque(maxlen=self.num_stacked_frames)
        next_state_deque = deque(maxlen=self.num_stacked_frames)

        # Get initial frame
        row = self.data.iloc[index]
        state_image = Image.open(row["state"]).convert(
            "L" if self.convert_to_grayscale else "RGB"
        )
        next_state_image = Image.open(row["next_state"]).convert(
            "L" if self.convert_to_grayscale else "RGB"
        )

        # Apply transformations
        if self.transform:
            state_image = self.transform(state_image)
            next_state_image = self.transform(next_state_image)

        # Initialize deque with repeated first frame
        for _ in range(self.num_stacked_frames):
            state_deque.append(state_image)
            next_state_deque.append(next_state_image)

        # Iterate to add newer frames
        for i in range(1, self.num_stacked_frames):
            if index + i < len(self.data):
                row = self.data.iloc[index + i]
                new_state_image = Image.open(row["state"]).convert(
                    "L" if self.convert_to_grayscale else "RGB"
                )
                new_next_state_image = Image.open(row["next_state"]).convert(
                    "L" if self.convert_to_grayscale else "RGB"
                )

                if self.transform:
                    new_state_image = self.transform(new_state_image)
                    new_next_state_image = self.transform(new_next_state_image)

                state_deque.append(new_state_image)
                next_state_deque.append(new_next_state_image)

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
