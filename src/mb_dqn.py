import torch
import torch.nn as nn
import pickle


class EncoderModel(nn.Module):
    def __init__(self, pretrained_path):
        super(EncoderModel, self).__init__()
        self.encoder = torch.load(pretrained_path, weights_only=True)

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


def q_loss(q_values, rewards, next_q_values, dones, gamma=0.99):
    """
    Compute the Q-learning loss.

    Args:
        q_values (torch.Tensor): Q-values predicted by the model.
        actions (torch.Tensor): Actions taken by the agent.
        rewards (torch.Tensor): Rewards received after taking the actions.
        next_q_values (torch.Tensor): Q-values predicted by the target model.
        dones (torch.Tensor): Whether the episode has terminated.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Q-learning loss.
    """
    target_q_values = rewards + gamma * next_q_values * (1 - dones)
    return nn.HuberLoss(delta=1.0)(q_values, target_q_values)
