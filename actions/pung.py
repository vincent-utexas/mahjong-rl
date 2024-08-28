import numpy as np
from tile import Tile

class Pung:
    @staticmethod
    def can_pung(player, last_tile: Tile):
        n = last_tile.to_int()
        bincount = np.bincount(player.tileset.tiles_n, minlength=36)
        return bincount[n] >= 2

    @staticmethod
    def pung(player, last_tile, deck, discard):
        assert Pung.can_pung(player, last_tile)
        coords = last_tile.to_coords()
        discard[coords] += 3
        tiles_to_freeze = player.tileset.remove(last_tile, manner='all')
        if len(tiles_to_freeze) >= 3: # grabbed a gan, add one back
            player.tileset.add(last_tile)
        player.frozen.add(tiles_to_freeze)
        player._tileset_full.add(last_tile)
