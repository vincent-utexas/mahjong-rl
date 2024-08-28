import random
import math
from datetime import date
import torch
import pandas as pd
import numpy as np
from environment import MahjongEnv
from dqn import DQN, ReplayMemory, Transition

from copy import deepcopy

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = MahjongEnv()

def generate_target_net(target_for, n_inputs, n_outputs):
    net = DQN(n_inputs, n_outputs)
    net.load_state_dict(target_for.state_dict())
    return net

# invalid actions: configure in options?
# large penalty
# skip agent turn
# default: do not permit, try down the list 

n_actions = env.action_space.n

discard_net = DQN(36 + 36, 36) # my tiles, discarded tiles

policy_net = DQN(36 + 36 + 1, n_actions) # my tiles, discarded tiles, last tile placed
target_net = generate_target_net(policy_net, 36 + 36 + 1, n_actions)
policy_optim = torch.optim.Adam(policy_net.parameters(), lr=LR)

memory = ReplayMemory(10000)

class EGAgent:
    # decide between a chi, pung, or draw
    # always hu, always gan
    def __init__(self, policy_net: DQN):
        self.steps_done = 0
        self.policy_net = policy_net

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) \
            * math.exp(-1 * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).view((1, 1))
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    # Sample and preprocess states
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Don't want to include final states in the weight updates because rewards are known
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(BATCH_SIZE, 73)

    state_batch = torch.cat(batch.state).view(BATCH_SIZE, 73)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Pass states to policy network -> compute Q(s, a) values
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    # Compute Q(s', a'), to calculate loss between target and output Q values
    next_state_action_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_action_values[non_final_mask] = target_net(non_final_next_states).max(dim=1).values

    # Compute expectation E[R + gamma * Q(s', a')], this is target Q value
    target_q_values = reward_batch + GAMMA * next_state_action_values
    criterion = torch.nn.SmoothL1Loss() # Huber loss, MSE for small x, MAE for large x
    loss = criterion(state_action_values, target_q_values)
    
    policy_optim.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    policy_optim.step()

def do_sanity_check(records):
    action_defs = {
        0: env.can_draw,
        1: env.can_chi,
        2: env.can_pung,
        3: env.can_gan,
        4: env.can_hu
    }

    good_actions = 0
    named_actions = ['draw', 'chi', 'pung', 'gan', 'hu']
    counts = pd.DataFrame({'good': [0] * 5, 'bad': [0] * 5}, index=named_actions)

    for r in records:
        last_tile, action, player = r
        success = action_defs[action](player, last_tile)
        good_actions += int(success)
        if success:
            counts.loc[named_actions[action], 'good'] += 1
        else:
            counts.loc[named_actions[action], 'bad'] += 1
    
    counts['sum'] = counts.sum(axis=1)
    penalty = counts['bad'].sum() * env._invalid_penalty
    good_points = np.arange(len(counts.index.values)) @ (counts['good'].values * env._valid_reward)
    
    print(f'n records: {len(records)}')
    print(f'good actions: {good_actions/len(records)}')
    print(f'net points: {good_points + penalty - 1}')
    print(counts)
    
def train():
    agent = EGAgent(policy_net)
    adversary_net = deepcopy(policy_net)
    adversary = EGAgent(adversary_net)
    for param in adversary_net.parameters():
        param.requires_grad = False

    options = {'discard_model': discard_net, 'handle_invalid_action': 'penalty'}
    agent.policy_net.train()

    for i_episode in range(201):
        state, _ = env.reset(options=options) # Starting state with 1 discarded tile, player 1 (adversary) starts
        state = torch.from_numpy(state).float()
        record = []

        steps_done = 0
        
        while True:
            actions = [agent.select_action(state)] + [adversary.select_action(state) for _ in range(3)]
            actions = [a.item() for a in actions]
            observation, reward, terminated, truncated, _ = env._do_passing_round(actions) # Skip mechanic
            state = torch.tensor(observation, dtype=torch.float32)

            player = agent if env._current_player == 0 else adversary

            # Select an action, observe the reward and states
            action = player.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            # Agent plays against 3 copies of its past self, 
            # as it gets better the opponents become its past self
            if player == adversary:
                if not (terminated or truncated):
                    continue
                else: # Adversary won, or game ran out of tiles
                    reward = -reward
                
            steps_done += 1
            truncated = truncated or steps_done > 100

            record.append((env._last_tile, action.item(), env._player_states[0]))
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32)
            
            # Store experience in memory
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()

            # Soft update of the target network's weights (instead of periodically copying)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU \
                    + target_net_state_dict[key] * (1 - TAU)
                
            target_net.load_state_dict(target_net.state_dict())
            if done:
                break

        print()
        print(f'episode {i_episode} complete')
        print(f'ending deck size: {len(env._deck)}')
        print(f'truncated due to steps: {steps_done == 100}')
        if i_episode % 25 == 0:
            print(f'episode {i_episode}')
            do_sanity_check(record)

            adversary_net = deepcopy(policy_net)
            adversary = EGAgent(adversary_net)
            for param in adversary_net.parameters():
                param.requires_grad = False

    agent.policy_net.eval()
    torch.save(agent.policy_net.state_dict(), f"./state_dicts/{date.today()}_policy_net")

if __name__ == '__main__':
    train()
