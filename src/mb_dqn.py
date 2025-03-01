import torch
import torch.nn as nn
import pickle
from siamese import EncoderModel as PretrainedEncoderModel


class EncoderModel(nn.Module):
    def __init__(self, pretrained_path):
        super(EncoderModel, self).__init__()
        self.encoder = PretrainedEncoderModel()
        self.encoder.load_state_dict(torch.load(pretrained_path))
        self.encoder.eval()

    def forward(self, x):
        encoded_state = self.encoder(x)
        return encoded_state


class MHCAModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads):
        super(MHCAModel, self).__init__()
        self.state_projection = nn.Linear(state_dim, state_dim)
        self.action_projection = nn.Linear(action_dim, state_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, encoded_state, action_embeddings):
        # encoded_state shape: (batch_size, state_dim) => (64, 400)
        # action_embeddings shape: (num_actions, embedding_dim) => (5, 50)

        # Convert encoded_state to (batch_size, 1, state_dim)
        projected_state = self.state_projection(
            encoded_state.unsqueeze(1)
        )  # => (64, 1, 400)

        # Expand action embeddings to (batch_size, num_actions, embedding_dim)
        if action_embeddings.dim() == 2:
            action_embeddings = action_embeddings.unsqueeze(0)  # => (1, 5, 50)
        action_embeddings = action_embeddings.expand(
            projected_state.size(0), -1, -1
        )  # => (64, 5, 50)

        # Project to (batch_size, num_actions, state_dim)
        projected_actions = self.action_projection(action_embeddings)  # => (64, 5, 400)

        # Multi-head attention expects: (batch_size, seq_len, embed_dim)
        fused_features, _ = self.attention(
            projected_state, projected_actions, projected_actions
        )

        return fused_features.squeeze(1)  # => (64, 400)


class DQNModel(nn.Module):
    def __init__(
        self,
        encoder_path,
        action_embeddings,
        state_dim,
        action_dim,
        num_heads,
        num_actions,
    ):
        super(DQNModel, self).__init__()
        self.encoder = EncoderModel(encoder_path)
        self.action_embeddings = nn.Parameter(action_embeddings)
        self.mhca = MHCAModel(state_dim, action_dim, num_heads)
        self.fc = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        encoded_state = self.encoder(x)
        fused_features = self.mhca(encoded_state, self.action_embeddings)
        q_values = self.fc(fused_features)
        return q_values


def load_action_embeddings(filename):
    """
    Load action embeddings from a saved pickle file.

    Args:
        filename (str): Path to the pickle file containing action embeddings.

    Returns:
        torch.Tensor: Loaded action embeddings.
    """
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return torch.tensor(embeddings, dtype=torch.float32)
