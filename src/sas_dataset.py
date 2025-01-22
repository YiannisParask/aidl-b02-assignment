import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SasDataset(Dataset):
    def __init__(self, csv_file, transform=None, convert_to_grayscale=False, num_stacked_frames=1):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            convert_to_grayscale (bool): Convert images to grayscale if True, else keep them as RGB.
            num_stacked_frames (int): Number of stacked frames (default=1).
        '''
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.convert_to_grayscale = convert_to_grayscale
        self.num_stacked_frames = num_stacked_frames

    def __len__(self):
        return len(self.data) - self.num_stacked_frames + 1

    def __getitem__(self, index):
        # Get the paths of the stacked frames
        state_paths = self.data.iloc[index : index + self.num_stacked_frames]["state"].tolist()
        next_state_paths = self.data.iloc[index : index + self.num_stacked_frames]["next_state"].tolist()

        state_frames = []
        next_state_frames = []
        
        for state_path, next_state_path in zip(state_paths, next_state_paths):
            # Convert to grayscale if the flag is set
            if self.convert_to_grayscale:
                state_image = Image.open(state_path).convert("L")
                next_state_image = Image.open(next_state_path).convert("L")
            else:
                state_image = Image.open(state_path).convert("RGB")
                next_state_image = Image.open(next_state_path).convert("RGB")

            # Apply transformations, if any
            if self.transform:
                state_image = self.transform(state_image)
                next_state_image = self.transform(next_state_image)

            state_frames.append(state_image)
            next_state_frames.append(next_state_image)

        # Stack frames along the channel dimension (shape: [num_stacked_frames * channels, H, W])
        stacked_state = torch.cat(state_frames, dim=0)
        stacked_next_state = torch.cat(next_state_frames, dim=0)
            
        # Load actions and done values for the stack
        actions = self.data.iloc[index : index + self.num_stacked_frames]["action"].tolist()
        dones = self.data.iloc[index : index + self.num_stacked_frames]["done"].tolist()

        # Use the last action and done value in the stack
        last_action = torch.tensor(actions[-1], dtype=torch.long)
        last_done = torch.tensor(dones[-1], dtype=torch.bool)
        
        
        return stacked_state, last_action, stacked_next_state, last_done
