from typing import Literal, Iterable

import numpy as np
from tile import Tile, type_ranges

type classes = Literal['c', 'b', 'w', 'h']

class TileSet:
    def __init__(self, tiles: Iterable=[]):
        if isinstance(tiles, list):
            self.tiles_n = np.array(tiles)
        else:
            self.tiles_n = tiles
        self.tiles = np.array([Tile.from_int(n) for n in tiles])

    def has(self, cls: classes):
        min, max = type_ranges[cls]
        mask = (self.tiles_n >= min) & (self.tiles_n <= max)

        return mask.any()
    
    def _remove_as_int(self, n, manner):
        mask = self.tiles_n == n
        if not any(mask):
            return []
        
        if manner == 'first':
            mask = np.argmax(mask)

        deleted = self.tiles[mask]
        self.tiles_n = np.delete(self.tiles_n, mask)
        self.tiles = np.delete(self.tiles, mask)
        
        return deleted # Note this is either a single tile or an array of tiles
    
    def _remove_as_class(self, cls, manner):
        min, max = type_ranges[cls]
        mask = (self.tiles_n >= min) & (self.tiles_n <= max)
        if not any(mask):
            return []

        if manner == 'first':
            mask = np.argmax(mask)
            
        deleted = self.tiles[mask]
        self.tiles_n = np.delete(self.tiles_n, mask)
        self.tiles = np.delete(self.tiles, mask)

        return deleted # Note this is either a single tile or an array of tiles

    def remove(self, obj, manner: Literal['first', 'all']='first'):
        if isinstance(obj, int):
            return self._remove_as_int(obj, manner)
        elif isinstance(obj, Tile):
            n = obj.to_int()
            return self._remove_as_int(n, manner)
        elif isinstance(obj, str) and obj in ['c', 'b', 'w', 'h']:
            return self._remove_as_class(obj, manner)

    def _add_as_tile(self, tile: Tile):
        self.tiles = np.append(self.tiles, tile)
        self.tiles_n = np.append(self.tiles_n, tile.to_int())
    
    def _add_as_int(self, n: int):
        self.tiles = np.append(self.tiles, Tile.from_int(n))
        self.tiles_n = np.append(self.tiles_n, n)

    def add(self, other):
        if isinstance(other, int) or isinstance(other, np.int32):
            self._add_as_int(other)
        elif isinstance(other, Iterable) and (type(other[0]) == int or type(other[0]) == np.int32):
            for n in other:
                self._add_as_int(n)
        elif isinstance(other, Tile):
            self._add_as_tile(other)
        elif isinstance(other, Iterable) and type(other[0]) == Tile:
            for tile in other:
                self._add_as_tile(tile)

    def to_grid(self):
        grid = np.zeros(shape=(4, 9))
        for t in self.tiles:
            grid[t.to_coords()] += 1
        
        return grid

    def __repr__(self):
        return ','.join([repr(t) for t in self.tiles])

    def __len__(self):
        return len(self.tiles)
    
    def __iter__(self):
        return iter(self.tiles)