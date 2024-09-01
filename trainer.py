from copy import deepcopy
import random

from dq_agent import EGAgent
from dqn import configs, ReplayMemory, Transition
from mj_env import MahjongEnv

import torch
import pandas as pd

BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
FILENAME = '2024-08-31_policy_net_conv2d_no_corrupt'
LOADPATH = f"./pretrained/{FILENAME}"
SAVEPATH = f"./state_dicts/{FILENAME}"

class Trainer:
    def train(self):
        adversary_net = self._get_adversary_net(agent.policy_net)
        adversary = EGAgent(adversary_net, discard_net, env)
        agent.policy_net.train()

        options = {
            'discard_model': agent.discard_net,
            'handle_invalid_action': 'penalty',
            'render_mode': None
            }
        
        for i_episode in range(n_episodes):
            state, _ = env.reset(options=options)
            records = []
            self._do_episode(adversary, state, records)

            print(f'episode {i_episode} complete')
            self._do_sanity_check(records)

        agent.policy_net.eval()

    def _do_episode(self, adversary, state, records):
        while True:
            if self._do_skip_round(adversary, records):
                continue

            player = agent if env._current_agent == 0 else adversary

            # Select an action, observe the reward and states
            action = player.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            if player == adversary:
                if not (terminated or truncated):
                    continue
                else:
                    reward = -abs(reward)

            reward = torch.tensor([reward])
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = observation

            # Store experience in memory
            memory.push(state, action, next_state, reward)
            records.append((deepcopy(env._last_tile), action.item(), deepcopy(env._agents[0])))
            state = next_state
            self._optimize_model()
            self._soft_update_target()

            if done:
                break

    def _do_skip_round(self, adversary, records):
        states = [env._get_obs(i) for i in range(4)]
        actions = [agent.select_action(states[0])] + [adversary.select_action(states[i]) for i in range(1, 4)]
        actions = [a.item() for a in actions]

        resuming_agent = env._current_agent
        state = env._get_obs(0) # Since we only care about the agent
        observation, reward, terminated, _, _ = env.do_skip_round(actions)
        post_action_agent = env._current_agent

        # Check if a skip event occurred
        # also, check if the agent made the skip event (post_action_agent would be 1 after the update)
        if resuming_agent == post_action_agent:
            return False
        elif post_action_agent != 1:
            return True
        
        reward = torch.tensor([reward])
        if terminated:
            next_state = None
        else:
            next_state = observation

        memory.push(state, actions[0], next_state, reward)
        records.append((env._last_tile, actions[0], env._agents[0]))
        self._optimize_model()
        self._soft_update_target()
        return True

    def _optimize_model(self):
        if len(memory) < BATCH_SIZE:
            return 
        
        # Sample and preprocesses states
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Don't want to include final state in the weight updates
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(BATCH_SIZE, 1, 4, 9)

        state_batch = torch.cat([s for s in batch.state]).view(BATCH_SIZE, 1, 4, 9)
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

    def _soft_update_target(self):
    # Soft update of the target network's weights (instead of periodically copying)
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU \
                + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

    def _get_adversary_net(self, policy_net):
        # Model an adversary net after the policy net
        # then freeze the adversary net
        net = deepcopy(policy_net)
        for param in net.parameters():
            param.requires_grad = False
        return net

    def get_target_net(self, policy_net):
        # Target net for more stable policy training
        net = configs['conv2d'](n_observations, n_actions)
        net.load_state_dict(policy_net.state_dict())
        return net

    def _do_sanity_check(self, records):
        action_defs = {
            0: env.can_draw,
            1: env.can_chi,
            2: env.can_pung,
            3: env.can_gan,
        }

        good_actions = 0
        named_actions = ['draw', 'chi', 'pung', 'gan']
        counts = pd.DataFrame({'good': [0] * n_actions, 'bad': [0] * n_actions}, index=named_actions)

        for r in records:
            last_tile, action, player = r
            if action == 0:
                success = True
                for can_action in env._can_action_defs:
                    if can_action == env.draw:
                        continue
                    if can_action(player, last_tile):
                        success = False
                        break
            else:
                success = action_defs[action](player, last_tile)
            good_actions += int(success)
            if success:
                counts.loc[named_actions[action], 'good'] += 1
            else:
                counts.loc[named_actions[action], 'bad'] += 1
            
        # Pick a random state
        record = random.choice(records)
        last_tile, action, player = record
        last_tile = last_tile.to_coords()

        # Run it through the policy net and compare
        input = torch.from_numpy(player.tileset.to_grid()).float()

        input[last_tile] += 1
        input[last_tile] *= -1
        print(input.view((4, 9)))
        input[last_tile] *= -1
        input = input.view(1, 1, 4, 9)

        with torch.no_grad():
            print(f"What I would do: {policy_net(input).argmax()}")
        print(f"What I did: {action}")
        
        print(counts)

if __name__ == '__main__':
    env = MahjongEnv()
    n_actions = env.action_space.n
    n_observations = 36
    n_episodes = 500
    memory = ReplayMemory(10000)
    trainer = Trainer()

    state_dict = torch.load(LOADPATH, weights_only=True)
    policy_net = configs['conv2d'](n_observations, n_actions)
    policy_net.load_state_dict(state_dict)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=LR)
    target_net = trainer.get_target_net(policy_net)
    discard_net = configs['wide'](72, 36)

    agent = EGAgent(policy_net, discard_net, env)
    
    try:
        trainer.train()
    finally:
        torch.save(policy_net.state_dict(), SAVEPATH)