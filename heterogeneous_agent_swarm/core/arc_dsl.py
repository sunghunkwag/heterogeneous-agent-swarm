import numpy as np
from typing import List, Callable, Any

Grid = np.ndarray

class ARCDSL:
    """
    Domain Specific Language for ARC Tasks.
    Contains 20+ primitives for manipulating 2D grids.
    """
    
    # --- Orientations ---
    @staticmethod
    def identity(g: Grid) -> Grid:
        return g
    @staticmethod
    def rotate_cw(g: Grid) -> Grid:
        return np.rot90(g, k=-1)
    @staticmethod
    def rotate_ccw(g: Grid) -> Grid:
        return np.rot90(g, k=1)
    @staticmethod
    def rotate_180(g: Grid) -> Grid:
        return np.rot90(g, k=2)
    @staticmethod
    def flip_vertical(g: Grid) -> Grid:
        return np.flipud(g)
    @staticmethod
    def flip_horizontal(g: Grid) -> Grid:
        return np.fliplr(g)
    @staticmethod
    def transpose(g: Grid) -> Grid:
        return g.T

    # --- Cropping & Resizing ---
    @staticmethod
    def crop_to_content(g: Grid) -> Grid:
        """Returns the smallest subgrid containing all non-zero pixels."""
        rows = np.any(g != 0, axis=1)
        cols = np.any(g != 0, axis=0)
        if not np.any(rows): return g # Empty
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return g[rmin:rmax+1, cmin:cmax+1]

    @staticmethod
    def top_half(g: Grid) -> Grid:
        return g[:g.shape[0]//2, :]
    @staticmethod
    def bottom_half(g: Grid) -> Grid:
        return g[g.shape[0]//2:, :]
    @staticmethod
    def left_half(g: Grid) -> Grid:
        return g[:, :g.shape[1]//2]
    @staticmethod
    def right_half(g: Grid) -> Grid:
        return g[:, g.shape[1]//2:]
    
    @staticmethod
    def tile_2x2(g: Grid) -> Grid:
        return np.tile(g, (2, 2))
    
    # --- Movement (Rolls) ---
    @staticmethod
    def move_down(g: Grid) -> Grid:
        return np.roll(g, 1, axis=0) # Wrap around behavior
    @staticmethod
    def move_up(g: Grid) -> Grid:
        return np.roll(g, -1, axis=0)
    @staticmethod
    def move_right(g: Grid) -> Grid:
        return np.roll(g, 1, axis=1)
    @staticmethod
    def move_left(g: Grid) -> Grid:
        return np.roll(g, -1, axis=1)

    # --- Physics / Sorting ---
    @staticmethod
    def gravity_down(g: Grid) -> Grid:
        result = np.zeros_like(g)
        h, w = g.shape
        for col in range(w):
            pixels = g[:, col]
            non_zeros = pixels[pixels != 0]
            zeros = np.zeros(h - len(non_zeros))
            result[:, col] = np.concatenate([zeros, non_zeros])
        return result

    @staticmethod
    def sort_pixels_row(g: Grid) -> Grid:
        return np.sort(g, axis=1)
    @staticmethod
    def sort_pixels_col(g: Grid) -> Grid:
        return np.sort(g, axis=0)

    # --- Color Ops ---
    @staticmethod
    def color_invert(g: Grid) -> Grid:
        if np.max(g) <= 1: return 1 - g
        return g
    
    @staticmethod
    def color_shift(g: Grid) -> Grid:
        # Increment non-zero colors
        mask = g != 0
        res = g.copy()
        res[mask] = (res[mask] + 1) % 10
        # If it became 0, make it 1? No, 0 is background.
        return res

    @staticmethod
    def fill_holes(g: Grid) -> Grid:
        # Simple orthagonal fill of 0s surrounded by non-zeros?
        # Too complex for quick numpy, doing simple flood fill simulation if I had it.
        # Fallback: Replace isolated 0s.
        # Check simple kernel convolution manually
        h, w = g.shape
        res = g.copy()
        if h < 3 or w < 3: return res
        for y in range(1, h-1):
            for x in range(1, w-1):
                if res[y,x] == 0:
                    neighbors = [res[y-1,x], res[y+1,x], res[y,x-1], res[y,x+1]]
                    if all(n != 0 for n in neighbors):
                        # Fill with top neighbor
                        res[y,x] = neighbors[0]
        return res

    @staticmethod
    def outline(g: Grid) -> Grid:
        # Keep pixels that have at least one 0 neighbor
        h, w = g.shape
        res = np.zeros_like(g)
        padded = np.pad(g, 1, mode='constant', constant_values=0)
        for y in range(h):
            for x in range(w):
                if g[y,x] != 0:
                    py, px = y+1, x+1
                    # Check 4 neighbors in padded
                    neighbors = [padded[py-1,px], padded[py+1,px], padded[py,px-1], padded[py,px+1]]
                    if any(n == 0 for n in neighbors):
                        res[y,x] = g[y,x]
        return res

    @staticmethod
    def get_primitives() -> List[Callable[[Grid], Grid]]:
        return [
            ARCDSL.identity,
            ARCDSL.rotate_cw, ARCDSL.rotate_ccw, ARCDSL.rotate_180,
            ARCDSL.flip_vertical, ARCDSL.flip_horizontal, ARCDSL.transpose,
            ARCDSL.crop_to_content,
            ARCDSL.top_half, ARCDSL.bottom_half, ARCDSL.left_half, ARCDSL.right_half,
            ARCDSL.tile_2x2,
            ARCDSL.move_down, ARCDSL.move_up, ARCDSL.move_right, ARCDSL.move_left,
            ARCDSL.gravity_down,
            ARCDSL.sort_pixels_row, ARCDSL.sort_pixels_col,
            ARCDSL.color_invert, ARCDSL.color_shift,
            ARCDSL.fill_holes, ARCDSL.outline
        ]
