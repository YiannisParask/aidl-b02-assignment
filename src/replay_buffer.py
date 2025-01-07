import random
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory
        
        Returns:
            states: list of states
            actions: list of actions
            rewards: list of rewards
            next_states: list of next states
            dones: list of dones
        '''
        experiences = random.sample(self.buffer, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.int64).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)
