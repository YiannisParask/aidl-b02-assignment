import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from simple_dqn import QNetwork
from replay_buffer import ReplayBuffer


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        device,
        lr,
        buffer_size,
        batch_size,
        gamma,
        tau,
        update_every,
        input_type="vector",
    ):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            device (torch.device): device to use for computation
            lr (float): learning rate
            buffer_size (int): maximum size of replay buffer
            batch_size (int): size of each training batch
            gamma (float): discount factor
            tau (float): soft update interpolation factor
            update_every (int): how often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.input_type = input_type

        if input_type == "image":
            self.qnetwork_local = QNetwork(
                action_size=action_size, seed=seed
            ).to(device)
            self.qnetwork_target = QNetwork(
                action_size=action_size, seed=seed
            ).to(device)
        elif input_type == "vector":
            self.qnetwork_local = QNetwork(
                state_size=state_size, action_size=action_size, seed=seed
            ).to(device)
            self.qnetwork_target = QNetwork(
                state_size=state_size, action_size=action_size, seed=seed
            ).to(device)
        else:
            raise ValueError("Invalid input_type. Must be 'vector' or 'image'.")
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory and learn if enough samples are available.

        Args:
            state (array_like): current state
            action (int): action taken
            reward (float): reward received
            next_state (array_like): next state
            done (bool): whether the episode is done
        """
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)
            
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
