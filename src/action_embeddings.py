import torch
import torch.nn as nn
import pickle
from siamese import EncoderModel as PretrainedEncoderModel


class EncoderModel(nn.Module):
    def __init__(self, encoder_path):
        super(EncoderModel, self).__init__()
        self.encoder = PretrainedEncoderModel()
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()

    def forward(self, stacked_frames):
        return self.encoder(stacked_frames)


class EmbeddingModel(nn.Module):
    def __init__(self, action_size, embedding_size):
        super(EmbeddingModel, self).__init__()
        # Define an embedding layer
        self.embedding = nn.Embedding(action_size, embedding_size)

    def forward(self, action):
        # Convert action to embedding
        return self.embedding(action)


class ForwardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardModel, self).__init__()
        # Define dense layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoded_state, encoded_action):
        # Concatenate the encoded state and action
        x = torch.cat((encoded_state, encoded_action), dim=-1)
        # Pass through dense layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.output_layer(x)


def RMSE_loss_function(predicted, actual):
    # RMSE Loss
    return torch.sqrt(torch.mean((predicted - actual) ** 2))


def store_embeddings(embedding_model, filename):
    # Save the embedding weights
    embeddings = embedding_model.embedding.weight.data.cpu().numpy()
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)
