import numpy as np
from actions.draw import Draw

class Gan:
    @staticmethod
    def can_gan(player, last_tile):
        n = last_tile.to_int()
        bincount = np.bincount(player.tileset.tiles_n, minlength=36)
        return bincount[n] >= 3

    @staticmethod
    def gan(player, last_tile, deck, discard):
        coords = last_tile.to_coords()
        discard[coords] += 4
        tiles_to_freeze = player.tileset.remove(last_tile, manner='all')
        player._tileset_full.add(last_tile)
        if len(deck) > 0:
            Draw.draw(player, last_tile, deck, discard)
