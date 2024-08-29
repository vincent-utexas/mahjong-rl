from player import Player
from actions.chi import Chi
from tileset import TileSet
from tile import Tile

last_tile = Tile('c', 3)
tileset = TileSet([1, 3])
agent = Player([], tileset, [], 0, {})

print(Chi.chi(agent, last_tile))