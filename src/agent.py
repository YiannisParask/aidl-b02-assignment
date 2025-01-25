import random
import torch
import torch.optim as optim
from simple_dqn import QNetwork
from replay_buffer import ReplayBuffer
import torch.nn as nn


class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        device,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        tau=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        update_every=4,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_every = update_every
        self.t_step = 0

        # Initialize Q-network and target network
        self.q_network = QNetwork(action_size).to(device)
        self.target_network = QNetwork(action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, device)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def act(self, state, epsilon=None):
        """
        Select an action based on the current state.

        Args:
            state (np.array): Current state of the environment.
            epsilon (float): Exploration rate.
        Returns:
            action (int): Action to take.
        """
        epsilon = self.epsilon if epsilon is None else epsilon
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = state.to(self.device)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

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

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self):
        """
        Perform a single step of learning.
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).view(-1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # if delta==1, it is equivalent to SmoothL1Loss
        loss = nn.MSELoss()(q_values, targets)
        # loss = nn.HuberLoss(reduction="mean", delta=1.0)(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

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
