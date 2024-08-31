import torch

class Discard:
    @staticmethod
    def discard(player, discard_pile, model):
        # Need to predict which tile to remove
        # Make tileset grid
        tileset_grid = torch.zeros(size=(4, 9), dtype=torch.float32)
        for t in player.tileset:
            tileset_grid[t.to_coords()] += 1
        
        input = torch.cat([torch.flatten(tileset_grid), torch.flatten(discard_pile)])
        tile_values = model(input) # 36 vector

        with torch.no_grad():
            # Search for the lowest value tile
            grid = torch.arange(end=36)
            # something like [0, 35, 24, ...] the integer representation of the tile
            # sorted ascending by value, the first element is the best tile to discard
            sorted_values = grid[torch.argsort(tile_values)]
            valid_tiles = torch.isin(sorted_values, torch.tensor(player.tileset.tiles_n))
            lowest_val_idx = torch.nonzero(valid_tiles).squeeze()[0].item()
            lowest_val = sorted_values[lowest_val_idx].item()

        # Discard lowest value tile, this is guaranteed to exist
        discard = player.tileset.remove(lowest_val)
        discard_pile[discard.to_coords()] += 1
        return discard
        