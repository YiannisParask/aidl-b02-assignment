import torch
import torch.nn as nn
import torch.nn.functional as F

# reference https://medium.com/@hkabhi916/mastering-convolutional-deep-q-learning-with-pytorch-a-comprehensive-guide-0114742a0a62
class QNetwork(nn.Module):
    """Actor (Policy) Model for image-based input."""

    def __init__(self, action_size, in_channels=4, input_size=(80, 80)):
        """Initialize parameters and build model."""
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.output_dim = self._compute_output_dim(in_channels, input_size)
        self.fc1 = nn.Linear(self.output_dim, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        return x.numel()
