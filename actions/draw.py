from typing import Literal

class Draw:
    @staticmethod
    def can_draw(player=None, last_tile=None):
        return True

    @staticmethod
    def draw(player, last_tile, deck, discard, draw: Literal['right', 'left']='right'):
        if draw == 'right':
            tile = deck.pop()
        else:
            tile = deck.popleft()
        
        player.tileset.add(tile.item())
        player._tileset_full.add(tile.item())
        
        return tile