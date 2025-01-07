import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderModel(nn.Module):
    def __init__(self, pretrained_path):
        super(EncoderModel, self).__init__()
        self.encoder = torch.load(pretrained_path)

    def forward(self, x):
        encoded_state = self.encoder(x)
        return encoded_state


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.state_projection = nn.Linear(state_dim, state_dim)
        self.action_projection = nn.Linear(action_dim, state_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, encoded_state, action_embeddings):
        # Project state and actions
        projected_state = self.state_projection(encoded_state.unsqueeze(1))
        projected_actions = self.action_projection(action_embeddings)

        # Compute attention
        fused_features, _ = self.attention(
            projected_state, projected_actions, projected_actions
        )
        return fused_features.squeeze(1)


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
        self.action_embeddings = nn.Parameter(
            action_embeddings
        )  # Learnable action embeddings
        self.mhca = MultiHeadCrossAttention(state_dim, action_dim, num_heads)
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

