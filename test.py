from pretrain import generate_gan, generate_pung, generate_chi, generate_hu, generate_draw, PATH
from tile import Tile
import torch
from dqn import configs

LOADPATH = PATH
print(f'loading {LOADPATH}')
state_dict = torch.load(LOADPATH, weights_only=True)
policy_net = configs['conv2d_deep'](36, 4)
policy_net.load_state_dict(state_dict)
policy_net.eval()

generators = [generate_gan, generate_pung, generate_chi, generate_draw]

for gen in generators:
    res = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(5000):
        tileset, last_tile, target = gen()
        # tileset = torch.tensor(tileset.to_grid().ravel(), dtype=torch.float32)
        # last_tile = torch.tensor([last_tile], dtype=torch.float32)
        # inp = torch.concat([tileset, last_tile])
        tileset = torch.from_numpy(tileset.to_grid()).float()
        tileset[Tile.from_int(last_tile).to_coords()] += 1
        tileset = tileset.view((1, 1, 4, 9))
        res[torch.argmax(policy_net(tileset)).item()] += 1

    print(res)

