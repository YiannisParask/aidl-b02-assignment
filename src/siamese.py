import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModel(nn.Module):
    def __init__(self, in_channels=4, input_size=(80, 80)):
        """
        Args:
            in_channels: number of stacked frames (default=4).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.output_dim = self._compute_output_dim(in_channels, input_size)
        self.fc = nn.Linear(self.output_dim, 400)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)        
        return x
    
    def _compute_output_dim(self, in_channels, input_size):
        '''
        Compute the output dimension after passing through the conv layers
        Args:
            in_channels: number of stacked frames
            input_size: size of input image
        Returns:
            output_dim: output dimension
        '''
        dummy_input = torch.zeros(1, in_channels, *input_size)  # Batch size = 1
        x = self.pool(F.relu(self.conv1(dummy_input)))  
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()
        
        
class InverseModel(nn.Module):
    def __init__(self, encoder_output_dim=400, num_actions=3):
        """
        Args:
            encoder_output_dim: Dimensionality of the encoder's output (default=400).
            num_actions: Number of possible actions (default=3).
        """
        super().__init__()
        self.fc1 = nn.Linear(encoder_output_dim*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class SiameseModel(nn.Module):
    def __init__(self):
        """
        Args:
            encoder: Encoder model.
            inverse_model: Inverse model.
        """
        super(SiameseModel, self).__init__()
        self.encoder = EncoderModel()
        self.inverse_model = InverseModel()
    
    def forward(self, S_t, S_t_next):
        # Pass both inputs through the encoder
        encoded_S_t = self.encoder(S_t)
        encoded_S_t_next = self.encoder(S_t_next)

        # Concatenate the feature vectors
        combined_features = torch.cat((encoded_S_t, encoded_S_t_next), dim=1)

        # Pass concatenated features through the inverse model
        logits = self.inverse_model(combined_features)
        return logits
    

def loss_function(logits, targets):
    '''
    Compute the cross entropy loss between the predicted logits and the target labels.
    Args:
        logits: predicted logits from the model
        targets: target labels
        Returns:
        loss: computed loss
    '''
    # Cross Entropy Loss combines LogSoftmax and NLLLoss
    # categorical cross entropy or negative log likelihood as a loss function
    return nn.CrossEntropyLoss()(logits, targets)
