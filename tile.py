from typing import Literal

type2int = {
    'c': 0, # circles
    'b': 1, # bamboo
    'w': 2, # wan
    'h': 3  # honors 
}

int2type = {
    0: 'c',
    1: 'b',
    2: 'w',
    3: 'h'
}

type_ranges = {
    'c': (0, 9),
    'b': (9, 18),
    'w': (18, 27),
    'h': (27, 36)
}

class Tile:
    def __init__(self, type: Literal['c', 'b', 'w', 'h'], val: int):
        assert 1 <= val <= 9
        self.type = type
        self.val = val

    def to_int(self):
        return 9 * type2int[self.type] + self.val - 1
    
    def to_coords(self):
        return type2int[self.type], self.val - 1
        # coords = (math.floor(t / 9), t % 9 + 1)

    
    @staticmethod
    def from_int(n):
        type = None
        for k in type_ranges.keys():
            min, max = type_ranges[k]
            if n in range(min, max):
                type = k
                break
        
        
        val = n + 1 - 9 * type2int[type]
        return Tile(type, val)

    def __repr__(self):
        return str(self.val) + self.type

    def __len__(self):
        return 1