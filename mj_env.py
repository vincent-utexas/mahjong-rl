from collections import deque
from dataclasses import dataclass, field

from actions import *
from tile import Tile
from tileset import TileSet

import torch
import gymnasium as gym
import gymnasium.spaces as spaces

GRID = (4,9)

@dataclass
class _Agent:
    tileset: TileSet = None
    tensor: torch.Tensor = None
    stats: dict = field(default_factory=lambda: {'n_chi': 0, 'n_pung': 0, 'n_gan': 0})
    _tileset_full: TileSet = None

class MahjongEnv(gym.Env, Draw, Discard, Chi, Pung, Gan, Hu):
    metadata = {'render_modes': ['human'], 'fps': 4}

    def __init__(self):
        super(MahjongEnv, self).__init__()

        self._action_defs = [
            self.draw,
            self.chi,
            self.pung,
            self.gan
        ]

        self._can_action_defs = [
            self.can_draw,
            self.can_chi,
            self.can_pung,
            self.can_gan
        ]
        self.action_space = spaces.Discrete(len(self._action_defs))
        self.observation_space = spaces.Dict(
            {
                'tileset': spaces.Discrete(13),
                'last_tile': spaces.Discrete(1)
            }
        )

        # State variables
        self._last_tile: Tile = None
        self._current_agent: int = 0
        self._agents: list[_Agent] = []
        self._discard: torch.Tensor = None
        self._deck: deque = None
        self._discard_model = None

    def reset(self, seed=None, options=None):
        assert 'discard_model' in options
        super().reset(seed=seed)

        self._handle_invalid_action = options.get('handle_invalid_action') or 'try_all' # penalty, skip, try_all
        self._penalty = options.get('penalty') or -0.15
        self._reward_scalar = options.get('reward') or 10
        self._discard_model = options['discard_model']

        self._current_agent = 1 # 2nd player moves next
        self._discard = torch.zeros(size=GRID, dtype=torch.float32)
        self._deck = torch.cat([torch.randperm(36) for _ in range(4)])
        self._deck = deque(self._deck)
        self._agents = [_Agent() for _ in range(4)]

        for a in self._agents:
            tiles = []
            for _ in range(13):
                tiles.append(self._deck.pop().item())
            a.tileset = TileSet(tiles)
            a._tileset_full = TileSet(tiles)
            a.tensor = torch.tensor(tiles, dtype=torch.float32)

        # First player gets extra tile but must discard first
        self._agents[0].tileset.add(self._deck.pop().item())
        self._last_tile = self.discard(self._agents[0], self._discard, self._discard_model)

        return self._get_obs(), self._get_info()
    
    def step(self, i_action):
        agent = self._agents[self._current_agent]
        action = self._action_defs[i_action]
        can_action = self._can_action_defs[i_action]

        if can_action(agent, self._last_tile):
            action(agent, self._last_tile, self._deck, self._discard)
        else:
            match self._handle_invalid_action:
                case "penalty":
                    return self._get_obs(), self._penalty, False, False, self._get_info()
                case "skip":
                    self._do_agent_update()
                    return self._get_obs(), 0, False, False, self._get_info()
                case "try_all":
                    return 
                case _:
                    raise ValueError()

        # Action successful, check if this caused a win to return early
        if self.can_hu(agent, self._last_tile):
            return self._get_obs(), 50, True, False, self._get_info()
        
        # No win, continue game by discarding a tile
        self._last_tile = self.discard(agent, self._discard, self._discard_model)

        # Checking for game end conditions
        truncated = len(self._deck) <= 0
        terminated = False
        reward = i_action * self._reward_scalar
        observation = self._get_obs()
        info = self._get_info()

        self._do_agent_update()
        return observation, reward, terminated, truncated, info

    def do_skip_round(self, i_actions):
        # Called before the main step function, after a new discard tile is placed
        # Check if someone can win from the just discarded tile
        for agent in self._agents:
            if self.can_hu(agent, self._last_tile):
                return self._get_obs(), 50, True, False, self._get_info() # todo , this is wrong
            
        # In a skip round, the person who just went cannot play again
        i_actions[self._current_agent - 1] = 0
        action_priority = sorted(i_actions)[::-1]
        agent_priority = reversed(torch.argsort(torch.tensor(i_actions)))

        agent_idx, action_idx = None, None
        for i_action, i_agent in zip(action_priority, agent_priority):
            if i_action == 0 or i_action == 1: # Chi and Draw do not get priority
                break

            can_action = self._can_action_defs[i_action]
            if can_action(self._agents[i_agent], self._last_tile):
                action_idx = i_action
                agent_idx = i_agent
                break

        if action_idx and agent_idx:
            action = self._action_defs[action_idx]
            action(self._agents[agent_idx], self._last_tile, self._deck, self._discard)
        else: # Return the current state
            return self._get_obs(), 0, False, False, self._get_info()
    
        # Update the game state
        self._last_tile = self.discard(self._agents[agent_idx], self._discard, self._discard_model)
        truncated = len(self._deck) <= 0
        terminated = False
        reward = action_idx * self._reward_scalar
        observation = self._get_obs()
        info = self._get_info()

        self._current_agent = agent_idx
        self._do_agent_update()
        return observation, reward, terminated, truncated, info

    def _do_agent_update(self):
        self._current_agent += 1
        if self._current_agent >= 4:
            self._current_agent = 0

    def _get_obs(self, i_agent=None):
        i_agent = i_agent or self._current_agent
        tileset = torch.from_numpy(self._agents[i_agent].tileset.to_grid()).float()
        tileset[self._last_tile.to_coords()] += 1
        tileset = tileset.view(1, 1, 4, 9)
        return tileset
    
    def _get_info(self):
        return None
