from actions import *

import random
import torch

EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 0.95

class EGAgent:
    def __init__(self, policy_net, discard_net, env):
        self.epsilon = EPS_START
        self.policy_net = policy_net
        self.discard_net = discard_net
        self.env = env

    def select_action(self, state):
        sample = random.random()
        if sample > 1 - EPS_START:
            action = torch.tensor(self.env.action_space.sample(), dtype=torch.long).view(1,1)
        else:
            action = torch.argmax(self.policy_net(state)).view(1,1)

        self._epsilon_update()
        return action
    
    def _epsilon_update(self):
        if self.epsilon > EPS_END:
            self.epsilon *= EPS_DECAY