from collections import deque

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from actions.discard import Discard
from actions.draw import Draw
from actions.chi import Chi
from actions.pung import Pung
from actions.gan import Gan
from actions.hu import Hu
from tileset import TileSet
from player import Player

# todo: turn skipping !

class MahjongEnv(gym.Env, Draw, Discard, Chi, Pung, Gan, Hu):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(MahjongEnv, self).__init__()
        
        self._action_defs = {
            0: self.draw,
            1: self.chi,
            2: self.pung,
            3: self.gan,
            4: self.hu
        }
        self._can_action_defs = {
            0: self.can_draw,
            1: self.can_chi,
            2: self.can_pung,
            3: self.can_gan,
            4: self.can_hu
        }
        self.action_space = spaces.Discrete(len(self._action_defs.keys()))

        self._current_player = 0
        self._player_states = [Player() for _ in range(4)]
        self._last_tile = None
        self._first_action = True

        self.observation_space = spaces.Dict(
            {
                "discarded": spaces.Box(low=0, high=9, shape=(4, 9)),
                "deck": spaces.Box(low=0, high=9, shape=(144,)), # hidden to the network
                "last_tile": spaces.Discrete(36),
                "player_0": spaces.Discrete(13),
                "player_1": spaces.Discrete(13),
                "player_2": spaces.Discrete(13),
                "player_3": spaces.Discrete(13)
            }
        )
    
    def reset(self, seed=0, options=None):
        assert 'discard_model' in options
        super().reset(seed=seed)

        self._handle_invalid_action = options['handle_invalid_action'] or 'try_all' # penalty, skip, try_all
        self._invalid_penalty = options.get('penalty', None) or -0.15
        self._valid_reward = options.get('reward', None) or 10

        # Start a new game
        self._current_player = 1
        self._discarded = np.zeros(shape=(4, 9)) # also counts showing tiles

        deck = np.tile(np.arange(36), 4)
        rng = np.random.RandomState(seed=seed)
        self._deck = deque(rng.permutation(deck))

        for p in self._player_states:
            deck = []
            for _ in range(13):
                deck.append(self._deck.pop())
            p.tileset = p._tileset_full = TileSet(deck)
            p.frozen = TileSet()

        # First player gets extra tile, must discard first
        self._player_states[0].tileset.add(self._deck.pop())

        for p in self._player_states:
            while p.tileset.has('h'):
                honors = p.tileset.remove('h', manner='all')
                p.points += len(honors)

                for _ in range(len(honors)):
                    p.tileset.add(self._deck.popleft())

        # Do the first step of each round: player_0 discards a tile
        self._discard_model = options['discard_model']
        self._last_tile = self.discard(self._player_states[0], self._discarded, self._discard_model)

        return self._get_obs(), self._get_info()

    def step(self, i_action):
        agent = self._player_states[self._current_player]
        action = self._action_defs[i_action]

        try: # Agents may try invalid actions
            action(agent, self._last_tile, self._deck, self._discarded)
        except AssertionError:
            truncated = len(self._deck) <= 0
            match self._handle_invalid_action:
                case "penalty":
                    return self._get_obs(), self._invalid_penalty, False, truncated, self._get_info()
                case "skip":
                    return self._get_obs(), 0, False, truncated, self._get_info()
                case "try_all":
                    pass
                case _:
                    raise ValueError('Invalid action strategies must be "penalty", "skip", or "try_all')

        self._last_tile = self.discard(agent, self._discarded, self._discard_model)

        # Check for game end conditions
        truncated = len(self._deck) <= 0
        terminated = action == 4
        if terminated or truncated:
            reward = int(terminated) or -int(truncated)
        else:
            reward = i_action * self._valid_reward
       
        observation = self._get_obs()
        info = self._get_info()

        self._do_player_update()
        return observation, reward, terminated, truncated, info

    def _do_passing_round(self, i_actions):
        """
        if all actions are illegal, skip and resume play
        return observation, reward, truncated, terminated, info
        """
        # Everyone except the agent who just went has a chance to play
        i_actions[self._current_player - 1] = 0

        action_priority = sorted(i_actions)[::-1]
        agent_priority = np.argsort(i_actions)[::-1]

        agent, action = None, None
        for i_action, i_agent in zip(action_priority, agent_priority):
            if i_action == 0 or 1: # chi and draw do not get to skip turns
                break

            can_action = self._can_action_defs[i_action]
            if can_action(self._player_states[i_agent], self._last_tile):
                action = i_action
                agent = i_agent
                break

        if action and agent: # Must be the highest priority valid action
            self._action_defs[action]
        else: # No one wants to play a valid move, go to the next turn
            return self._get_obs(), 0, False, False, self._get_info()

        self._last_tile = self.discard(self._player_states[agent], self._discarded, self._discard_model)

        # Check for game end conditions
        truncated = len(self._deck) <= 0
        terminated = action == 4
        if terminated or truncated:
            reward = int(terminated) or -int(truncated)
        else:
            reward = action * self._valid_reward

        observation = self._get_obs()
        info = self._get_info()

        self._current_player = agent
        self._do_player_update()
        return observation, reward, terminated, truncated, info

    def _do_player_update(self):
        self._current_player += 1
        if self._current_player >= 4:
            self._current_player = 0

    def _get_obs(self):
        board_state = self._discarded.ravel()
        player_state = self._player_states[self._current_player].tileset.to_grid()
        concat_state = np.append(board_state, player_state)
        return np.append(concat_state, self._last_tile.to_int())

    def _get_info(self):
        return {
            "next_player": self._current_player,
            "player_0": self._player_states[0],
            "player_1": self._player_states[1],
            "player_2": self._player_states[2],
            "player_3": self._player_states[3],
        }