from collections import deque, namedtuple
import random

import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else \
    "mps" if torch.backends.mps.is_available() else \
    "cpu"
)

configs = {
    'wide': lambda n_observations, n_actions: nn.Sequential(
        nn.Linear(n_observations, 256),
        nn.ReLU(),
        nn.Linear(256, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, n_actions)),
    'conv2d': lambda n_observations, n_actions: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4,
                  kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=4, out_channels=8,
                  kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(36*8, 288),
        nn.ReLU(),
        nn.Linear(288, n_actions)
    ),
    'conv2d_deep': lambda n_observations, n_actions: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4,
                  kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=4, out_channels=8,
                  kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(36*8, 288),
        nn.ReLU(),
        nn.Linear(288, 288),
        nn.ReLU(),
        nn.Linear(288, n_actions)
        )
}

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    def __init__(self, capacity=1000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)