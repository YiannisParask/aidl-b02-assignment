from mb_dqn import DQNModel, load_action_embeddings
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn


class MHCAAgent:
    def __init__(
        self,
        encoder_path,
        action_embeddings_path,
        state_dim=400,
        action_dim=50,
        num_heads=4,
        num_actions=6,
        device="cuda",
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        tau=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        update_every=4,
    ):
        """
        MHCA-based Deep Q-Network Agent.
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_every = update_every
        self.t_step = 0

        # Load action embeddings
        action_embeddings = load_action_embeddings(action_embeddings_path).to(device)
        num_old_actions = action_embeddings.shape[0]
        action_dim = action_embeddings.shape[1]
        new_embeddings = torch.randn(num_actions - num_old_actions, action_dim).to(
            device
        )
        updated_action_embeddings = torch.cat(
            [action_embeddings, new_embeddings], dim=0
        )

        # MHCA-based Q-network
        self.q_network = DQNModel(
            encoder_path,
            updated_action_embeddings,
            state_dim,
            action_dim,
            num_heads,
            num_actions,
        ).to(device)
        self.target_network = DQNModel(
            encoder_path,
            updated_action_embeddings,
            state_dim,
            action_dim,
            num_heads,
            num_actions,
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def act(self, state, epsilon=None):
        """
        Select action using epsilon-greedy strategy.
        """
        epsilon = self.epsilon if epsilon is None else epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(
                self.q_network.fc[-1].out_features
            )  # Random action
        else:
            state = state.to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        """
        Perform a single step of learning.

        Args:
            state (np.array): Current state of the environment.
            action (int): Action taken in the current state.
            reward (float): Reward received after taking the action.
            next_state (np.array): State of the environment after taking the action.
            done (bool): Whether the episode has ended
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

        self.t_step += 1

        if (
            self.t_step % self.update_every == 0
            and len(self.replay_buffer) >= self.replay_buffer.batch_size
        ):
            self.learn()

    def learn(self):
        """
        Train using Multi-Head Cross-Attention (MHCA) and Q-loss.
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.HuberLoss()(q_values, targets)  # Using Huber Loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()

        # Update target network weights with soft update
        self.soft_update(self.q_network, self.target_network)

    def soft_update(self, source_model, target_model):
        """
        Perform a soft update of the target network parameters.
        (θ_target = τ*θ_local + (1 - τ)*θ_target)
        Args:
            target_model (torch.nn.Module): Target network.
            source_model (torch.nn.Module): Online (current) network.
            tau (float): Interpolation parameter (0 < tau <= 1).
        """
        for target_param, source_param in zip(
            target_model.parameters(), source_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
