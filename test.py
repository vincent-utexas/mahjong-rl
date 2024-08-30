# from player import Player
# from actions.chi import Chi
# from actions.pung import Pung
# from actions.gan import Gan
# from actions.hu import Hu
# from tileset import TileSet
# from tile import Tile

# last_tile = Tile.from_int(10)
# tileset = TileSet([10, 10, 9])
# agent = Player(tileset, tileset, [], 0, {'n_chi': 2, 'n_pung': 2, 'n_gan': 0})

# print(Gan.can_gan(agent, last_tile))
from pretrain import generate_gan, generate_pung, generate_chi, generate_hu
from actions.chi import Chi
from player import Player
from tile import Tile
import torch
from dqn import DQN

LOADPATH = f"./pretrained/2024-08-29_policy_net"
state_dict = torch.load(LOADPATH, weights_only=True)
policy_net = DQN(37, 5)
policy_net.load_state_dict(state_dict)


res = {0: 0, 1: 0, 2: 0, 3: 0, 4:0}
# for i in range(100):
#     tileset, last_tile, target = generate_hu()
#     tileset = torch.tensor(tileset.to_grid().ravel(), dtype=torch.float32)
#     last_tile = torch.tensor([last_tile], dtype=torch.float32)
#     inp = torch.concat([tileset, last_tile])
#     res[torch.argmax(policy_net(inp)).item()] += 1

# tileset = torch.tensor([1., 1., 1., 0., 1., 1., 1., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0.,        
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0.])
tileset, last_tile, _ = generate_hu()
tileset = torch.tensor(tileset.to_grid().ravel(), dtype=torch.float32)
print(tileset.view((4,9)))
inp = torch.cat([tileset, torch.tensor([last_tile])])
print(torch.softmax(policy_net(inp), dim=0))

print(res)

