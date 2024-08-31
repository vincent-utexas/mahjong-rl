import math
from tile import Tile, type_ranges
import numpy as np

class Chi:
    @staticmethod
    def can_chi(player, last_tile: Tile):
        bounds = Chi._get_bounds(last_tile)
        nums_in_bound = np.isin(bounds, player.tileset.tiles_n)
        valid_axes = np.apply_along_axis(all, axis=1, arr=nums_in_bound)

        return valid_axes.any()

    @staticmethod
    def chi(player, last_tile, deck, discard):        
        # Rule: always take more central tiles so we (hopefully) discard edge tiles
        min, max = type_ranges[last_tile.type]
        n = last_tile.to_int()
        bounds = Chi._get_bounds(last_tile)
        if n == min or max - 1: # Figuring out which one is the central pair
            idx = 0
        else:
            idx = 1

        tiles_to_freeze = bounds[idx] # the 2 surrounding tiles as numbers
        for t in tiles_to_freeze:
            player.tileset.remove(t)
            coords = (math.floor(t / 9), t % 9 + 1) # grid coords
            discard[coords] += 1

        player._tileset_full.add(last_tile)
        discard[last_tile.to_coords()] += 1

    @staticmethod
    def _get_bounds(last_tile: Tile):
        min, max = type_ranges[last_tile.type]
        n = last_tile.to_int()

        # Check class type boundaries
        if n == min:
            bounds = [(n + 1, n + 2)]
        elif n == min + 1:
            bounds = [(n - 1, n + 1), (n + 1, n + 2)]
        elif n == max - 1: # Last tile of my class
            bounds = [(n - 2, n - 1)]
        elif n == max - 2: # Second last tile of my class
            bounds = [(n - 2, n - 1), (n - 1, n + 1)]
        else:
            bounds = [(n - 2, n - 1), (n - 1, n + 1), (n + 1, n + 2)]
        
        return np.array(bounds)