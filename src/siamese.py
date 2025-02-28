import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderModel(nn.Module):
    def __init__(self, in_channels=3, input_size=(80, 80)):
        """
        Args:
            in_channels: number of input channels (default=3).:
                * 3 for RGB
                * 1 for Grayscale
                * 4 for 4 stacked Grayscale frames
                * 12 for 4 stacked RGB frames
            input_size: size of input image (default=(80, 80)).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.output_dim = self._compute_output_dim(in_channels, input_size)
        self.fc = nn.Linear(self.output_dim, 400)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _compute_output_dim(self, in_channels, input_size):
        """
        Compute the output dimension after passing through the conv layers
        Args:
            in_channels: number of stacked frames
            input_size: size of input image
        Returns:
            output_dim: output dimension
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_size)  # Batch size = 1
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.numel()


class InverseModel(nn.Module):
    def __init__(self, encoder_dim=400, num_actions=5):
        """
        Args:
            encoder_output_dim: Dimensionality of the encoder's output (default=400).
            num_actions: Number of possible actions (default=3).
        """
        super().__init__()
        self.fc1 = nn.Linear(encoder_dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
