from dataclasses import dataclass, field
from tileset import TileSet

@dataclass
class Player:
    _tileset_full: TileSet = None # tileset + frozen
    tileset: TileSet = None
    frozen: TileSet = None
    points: int = 0
    stats: dict = field(default_factory=lambda: {'n_chi': 0, 'n_pung': 0, 'n_gan': 0})