import random
import math
from datetime import date
import torch
import pandas as pd
import numpy as np
from environment import MahjongEnv
from dqn import DQN, ReplayMemory, Transition
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-5
LOADPATH = f"./pretrained/2024-08-29_policy_net"
SAVEPATH = f"./state_dicts/2024-08-30_policy_net"

class EGAgent:
    def __init__(self, policy_net: DQN):
        self.steps_done = 0
        self.policy_net = policy_net

    def select_action(self, state):
        tileset, tileset_full, last_tile = state.values()
        tileset_in = np.concatenate((tileset, [last_tile]))
        tileset_in = torch.from_numpy(tileset_in).float()

        tileset_full_in = np.concatenate((tileset_full, [last_tile]))
        tileset_full_in = torch.from_numpy(tileset_full_in).float()

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) \
            * math.exp(-1 * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold: # Pick highest Q value
            with torch.no_grad():
                tileset_pass = self.policy_net(tileset_in)
                tileset_full_pass = self.policy_net(tileset_full_in)
                maximum = torch.maximum(tileset_pass, tileset_full_pass)
                return torch.argmax(maximum).view((1,1))
        else: # Pick randomly
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

def generate_target_net(policy_net):
    net = DQN(n_observations, n_actions)
    net.load_state_dict(policy_net.state_dict())
    return net

def optimize_model():
    global losses
    if len(memory) < BATCH_SIZE:
        return
    
    # Sample and preprocess states
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Don't want to include final states in the weight updates because rewards are known
    concatr = lambda tileset, last_tile: np.concatenate((tileset, [last_tile]))
    tensorfy = lambda concatr: torch.from_numpy(concatr).float()
    preprocessor = lambda tileset, last_tile: tensorfy(concatr(tileset, last_tile))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = torch.cat([preprocessor(s['tileset'], s['last_tile']) for s in batch.next_state if s is not None]).view(BATCH_SIZE, n_observations)

    state_batch = torch.cat([preprocessor(s['tileset'], s['last_tile']) for s in batch.state]).view(BATCH_SIZE, n_observations)
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
    losses += [loss.item()]
    
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
        if action == 0:
            success = True
            for can_action in env._can_action_defs.values():
                if can_action == env.draw:
                    continue
                if can_action(player, last_tile):
                    success = False
        else:
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

def get_adversary_net(policy_net):
    # Model an adversary net after the policy net
    # Freeze it
    adversary_net = deepcopy(policy_net)
    for param in adversary_net.parameters():
        param.requires_grad = False
    return adversary_net

def soft_update_target():
    # Soft update of the target network's weights (instead of periodically copying)
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = agent.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU \
            + target_net_state_dict[key] * (1 - TAU)

def do_skip_round(adversary, record):
    # state is discard, tileset, and last tile
    states = env._get_all_obs()
    
    actions = [agent.select_action(states[0])] + [adversary.select_action(states[i]) for i in range(1, 4)]
    actions = [a.item() for a in actions]

    resuming_player = env._current_player
    observation, reward, terminated, _, _ = env.do_passing_round(actions)

    # Here, the previous observation and the current observation could be the same
    # (nothing happened)
    # also, we only care if the agent successfully performed a skip
    if resuming_player == env._current_player or env._current_player != 1:
        return 
    
    record.append((env._last_tile, actions[0], env._player_states[0]))

    state = torch.tensor(observation, dtype=torch.float32)
    reward = torch.tensor([reward])

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32)
    
    # Store experience in memory
    memory.push(state, actions[0], next_state, reward)
    optimize_model()

    soft_update_target(agent)

def do_episode(adversary, state, record):  
    while True:
        do_skip_round(adversary, record)
        player = agent if env._current_player == 0 else adversary

        # Select an action, observe the reward and states
        action = player.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())

        if player == adversary:
            if not (terminated or truncated): # Update last tile and go to next turn
                continue
            else: # Adversary won, or game ran out of tiles
                reward = -np.abs(reward)

        record.append((env._last_tile, action.item(), env._player_states[0])) # for logging
        reward = torch.tensor([reward])
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation
        
        # Store experience in memory
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        soft_update_target()
            
        target_net.load_state_dict(target_net.state_dict())
        if done:
            break

def train():
    adversary_net = get_adversary_net(agent.policy_net)
    adversary = EGAgent(adversary_net)

    options = {'discard_model': discard_net, 'handle_invalid_action': 'penalty'}
    agent.policy_net.train()

    for i_episode in range(10):
        state, _ = env.reset(options=options) # State is player_1 (adversary) starting hand and the tile placed by player_0
        record = []

        print(state)

        do_episode(adversary, state, record)

        print()
        print(f'episode {i_episode} complete')
        print(f'ending deck size: {len(env._deck)}')

        # Adversary net improvements
        do_sanity_check(record)
        if i_episode % 10 == 0:
            print(f'episode {i_episode}')
            do_sanity_check(record)

            adversary_net = get_adversary_net(agent.policy_net)
            adversary.policy_net = adversary_net

    agent.policy_net.eval()

    sns.lineplot(x=np.arange(len(losses)), y=losses)
    plt.show()

if __name__ == '__main__':
    env = MahjongEnv()

    n_actions = env.action_space.n
    n_observations = 36 + 1 # 36 tile grid, last tile, 4 actions

    discard_net = DQN(36 + 36, 36) # my tiles, discarded tiles

    state_dict = torch.load(LOADPATH, weights_only=True)
    policy_net = DQN(37, n_actions)
    policy_net.load_state_dict(state_dict=state_dict)
    # policy_net = DQN(n_observations, n_actions)
    target_net = generate_target_net(policy_net)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=LR)
    agent = EGAgent(policy_net)
    memory = ReplayMemory(10000)

    losses = []

    try:
        train()
    finally:
        torch.save(agent.policy_net.state_dict(), SAVEPATH)
        env.close()
