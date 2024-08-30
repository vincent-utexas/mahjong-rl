import numpy as np
from player import Player
from actions.chi import Chi
from actions.pung import Pung

class Hu:
    @staticmethod
    def can_hu(player: Player, last_tile):
        # Check for 3 sets, build 4th with last_tile
        sets = player.stats['n_chi'] + player.stats['n_pung'] + player.stats['n_gan']
        if sets < 3:
            return False
    
        can_make_fourth_set = Chi.can_chi(player, last_tile) or \
            Pung.can_pung(player, last_tile)
        
        if not can_make_fourth_set and sets < 4: # May be looking for a pair
            return False
        
        bincount = np.bincount(player._tileset_full.tiles_n, minlength=36)
        bincount[last_tile.to_int()] += 1
        pair_exists = (bincount == 2).any()
        return pair_exists

    @staticmethod
    def hu(player, last_tile, deck, discard):
        assert Hu.can_hu(player, last_tile) 