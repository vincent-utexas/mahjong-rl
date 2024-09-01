from tile import Tile
from tileset import TileSet
import actions.chi

import random
import torch
import numpy as np

# This file is a pretrainer to help the reinforcement learning algorithm converge
# This is like supervised learning to pretrain the net
# The intuition comes from a few ideas
# * If I have three of a kind, and the last tile fills the suite (I can gan), then I should always gan
# * If I can't pung but I can chi, I should chi
# * If I can't do anything, I should draw

PATH = f"./pretrained/2024-08-31_policy_net_conv2d_deep"

def setup(low=6, high=15):
    n_tiles = np.random.randint(low, high)
    deck = np.random.permutation(np.tile(np.arange(36), 4))
    return n_tiles, deck

def generate_draw():
    target = torch.zeros(size=(4,))
    target[0] = 1
    n_tiles, deck = setup()

    tiles = [deck[i] for i in range(n_tiles)]
    last_tile = deck[n_tiles + 1]

    return TileSet(tiles), last_tile, target

    # sample = random.random()
    
    # if sample > 0.5: # non corrupting draw (may add some noise)
    #     tiles = [deck[i] for i in range(n_tiles)]
    #     last_tile = deck[n_tiles + 1]
    #     # mask = (deck == last_tile) | (deck == last_tile - 1) | (deck == last_tile + 1)
    #     # deck = np.delete(deck, mask) # Prevents chi, pung, gan
    #     # tiles = [deck[i] for i in range(n_tiles)]
    # else:
    #     generators = [generate_chi, generate_pung, generate_gan]
    #     generator = random.choice(generators)
    #     tileset, last_tile, _ = generator()

    #     tiles_n = tileset.tiles_n
    #     avoid = np.concatenate((tiles_n, tiles_n + 1, tiles_n - 1))

    #     # Corrupt the last tile
    #     while last_tile in avoid:
    #         last_tile = random.randint(0, 35)

    #     return tileset, last_tile, target
    
    # return TileSet(tiles), last_tile, target

def generate_chi():
    target = torch.zeros(size=(4,))
    target[1] = 1
    n_tiles, deck = setup()
    last_tile = deck[0]
    _tile = Tile.from_int(last_tile)
    bound = actions.chi.Chi._get_bounds(_tile)

    mask = (deck == last_tile) | (deck == bound[0, 0]) | (deck == bound[0, 1])
    deck = np.delete(deck, (deck == last_tile))
    tiles = random.choice(bound).tolist() + [deck[i] for i in range(n_tiles - 2)]

    return TileSet(tiles), last_tile, target

def generate_pung():
    target = torch.zeros(size=(4,))
    target[2] = 1
    n_tiles, deck = setup()
    last_tile = deck[0]
    deck = np.delete(deck, (deck == last_tile))
    tiles = [last_tile] * 2 + [deck[i] for i in range(n_tiles - 2)]
    
    return TileSet(tiles), last_tile, target

def generate_gan():
    target = torch.zeros(size=(4,))
    target[3] = 1
    n_tiles, deck = setup()
    last_tile = deck[0]
    deck = np.delete(deck, (deck == last_tile))
    tiles = [last_tile] * 3 + [deck[i] for i in range(n_tiles - 3)]
    
    return TileSet(tiles), last_tile, target

def generate_hu():
    target = torch.zeros(size=(4,))
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
    from dqn import configs

    print(generate_chi())

    BATCH_SIZE = 128
    n_epochs = 10000
    pbar = ProgBar(n_epochs)

    policy_net = configs['conv2d'](36, 4) # My tiles, last tile
    # state_dict = torch.load(PATH[:-1]+"3", weights_only=True)
    # policy_net.load_state_dict(state_dict)
    policy_net.train()
    optim = torch.optim.Adam(policy_net.parameters(), 1e-4)
    generators = [generate_draw, generate_chi, generate_pung, generate_gan]
    losses = []

    for epoch in range(n_epochs):
        batch = []
        targets = []
        for n in range(BATCH_SIZE):
            action = random.choice(generators)
            tileset, last_tile, target = action()
            tileset = torch.from_numpy(tileset.to_grid()).float()
            tileset[Tile.from_int(last_tile).to_coords()] += 1
            batch += [tileset]
            targets += [target]
            # last_tile = torch.tensor([last_tile], dtype=torch.float32)

        # input = torch.concat([tileset, last_tile])
        input = torch.cat(batch).view((BATCH_SIZE, 1, 4, 9)) # N, C, H, W
        targets = torch.cat(targets).float()
        targets = targets.view((BATCH_SIZE, 4))
        pred = policy_net(input)
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(pred, targets)
        losses += [loss.item()]

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.update()

    torch.save(policy_net.state_dict(), PATH)

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # sns.lineplot(x=np.arange(n_epochs), y=losses)
    # plt.show()