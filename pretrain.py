import actions.chi
from dqn import DQN
from tile import Tile, type_ranges
from tileset import TileSet
from player import Player
import actions

import random
import torch
import numpy as np

# This file is a pretrainer to help the reinforcement learning algorithm converge quicker
# This is like supervised learning to pretrain the net
# The intuition comes from a few ideas
# * If I have a winning tileset (I can hu), then I must hu
# * If I have three of a kind, and the last tile fills the suite (I can gan), then I should always gan
# * If I can't pung but I can chi, I should chi, and vice versa

def setup():
    n_tiles = np.random.randint(3, 13)
    deck = np.random.permutation(np.tile(np.arange(36), 4))
    return n_tiles, deck

def generate_chi():
    target = torch.zeros(size=(5,))
    target[1] = 1
    n_tiles, deck = setup()
    last_tile = deck[0]
    deck = np.delete(deck, (deck == last_tile))
    _tile = Tile.from_int(last_tile)
    bound = actions.chi.Chi._get_bounds(_tile)
    tiles = random.choice(bound).tolist() + [deck[i] for i in range(n_tiles - 2)]

    return TileSet(tiles), last_tile, target

def generate_pung():
    target = torch.zeros(size=(5,))
    target[2] = 1
    n_tiles, deck = setup()
    last_tile = deck[0]
    deck = np.delete(deck, (deck == last_tile))
    tiles = [last_tile] * 2 + [deck[i] for i in range(n_tiles - 2)]
    
    return TileSet(tiles), last_tile, target

def generate_gan():
    target = torch.zeros(size=(5,))
    target[3] = 1
    n_tiles, deck = setup()
    last_tile = deck[0]
    deck = np.delete(deck, (deck == last_tile))
    tiles = [last_tile] * 3 + [deck[i] for i in range(n_tiles - 3)]
    
    return TileSet(tiles), last_tile, target

def generate_hu():
    target = torch.zeros(size=(5,))
    target[4] = 1
    n_tiles, deck = setup()

    pair_exists = random.randint(0, 1)
    need_chi = random.randint(0, 1)
    need_pung = random.randint(0, 1)
    last_tile = deck[0]
    deck = deck[1:]
    if pair_exists: # Need a last set, must sum to 11 + n_gan
        if need_chi:
            n_chi = random.randint(0, 3)
            n_pung = random.randint(0, 3 - n_chi)
            n_gan = 3 - n_chi - n_pung
        elif need_pung:
            n_pung = random.randint(0, 3)
            n_chi = random.randint(0, 3 - n_pung)
            n_gan = 3 - n_chi - n_pung
        else:
            n_gan = random.randint(0, 3)
            n_chi = random.randint(0, 3 - n_gan)
            n_pung = 3 - n_chi - n_gan
    else: # Need a last pair, must sum to 12 + n_gan
        n_chi = random.randint(0, 4)
        n_pung = random.randint(0, 4 - n_chi)
        n_gan = 4 - n_chi - n_pung

    tiles = []
    for _ in range(n_chi):
        chi_tile = deck[0]
        bound = random.choice(actions.chi.Chi._get_bounds(Tile.from_int(chi_tile)))
        tiles += bound.tolist() + [chi_tile]
        deck = deck[1:]

        for n in bound:
            mask = deck == n
            deck = np.delete(deck, mask.argmax())

    for _ in range(n_pung):
        pung_tile = deck[0]
        deck = deck[1:]
        tiles += [pung_tile] * 3
        deck = np.delete(deck, deck == pung_tile)

    for _ in range(n_gan):
        gan_tile = deck[0]
        deck = deck[1:]
        tiles += [gan_tile] * 4
        deck = np.delete(deck, deck == gan_tile)

    if not pair_exists: # Must sum to 13 + n_gan
        tiles += [deck[0]]
        last_tile = deck[0]
    else:
        # Give a free pair
        pair = deck[0]
        tiles += [pair] * 2
        deck = deck[1:]
        deck = np.delete(deck, (deck == pair).argmax())

        if need_chi:
            last_tile = deck[0]
            bound = random.choice(actions.chi.Chi._get_bounds(Tile.from_int(last_tile)))
            tiles += bound.tolist()
        else: # need pung
            last_tile = deck[0]
            tiles += [last_tile] * 2

    return TileSet(tiles), last_tile, target

if __name__ == '__main__':
    from pyprind import ProgBar

    pbar = ProgBar(30000)

    n_epochs = 30000
    policy_net = DQN(36 + 1, 5) # My tiles, last tile
    optim = torch.optim.Adam(policy_net.parameters(), 1e-4)
    generators = [generate_chi, generate_pung, generate_gan, generate_hu] # todo ensure equal distribution
    losses = []

    for epoch in range(n_epochs):
        action = random.choice(generators)
        tileset, last_tile, target = action()
        tileset = torch.tensor(tileset.to_grid().ravel(), dtype=torch.float32)
        last_tile = torch.tensor([last_tile], dtype=torch.float32)

        input = torch.concat([tileset, last_tile])
        pred = policy_net(input)
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(pred, target)
        losses += [loss.item()]

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.update()

    torch.save(policy_net.state_dict(), f"./pretrained/2024-08-30_policy_net")

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # sns.lineplot(x=np.arange(n_epochs), y=losses)
    # plt.show()