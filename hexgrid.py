"""
Flat-top hexagonal grid geometry, axial coordinate helpers, and zoom/pan camera.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from constants import ARENA_RADIUS, HEX_DIRECTIONS

HexCoord = Tuple[int, int]
SQRT3: float = math.sqrt(3.0)


# ============================================================================
# Axial Hex Grid Math (Flat-Top Orientation)
# ============================================================================

def is_valid_hex(q: int, r: int, radius: int = ARENA_RADIUS) -> bool:
    """Return True if axial coordinate (q, r) lies inside the hexagonal arena."""
    return max(abs(q), abs(r), abs(q + r)) <= radius


def get_all_arena_hexes(radius: int = ARENA_RADIUS) -> List[HexCoord]:
    """Return all valid (q, r) hex coordinates in a hexagonal arena of the given radius."""
    hexes: List[HexCoord] = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            hexes.append((q, r))
    return hexes


def hex_distance(a: HexCoord, b: HexCoord) -> int:
    """Compute grid distance between two axial hex coordinates."""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def hex_neighbor(coord: HexCoord, abs_dir: int) -> HexCoord:
    """Return the adjacent hex in absolute direction abs_dir (0..5)."""
    dq, dr = HEX_DIRECTIONS[abs_dir % 6]
    return (coord[0] + dq, coord[1] + dr)


def hex_direction_between(src: HexCoord, dst: HexCoord) -> Optional[int]:
    """If dst is adjacent to src, return the absolute direction 0..5 from src to dst."""
    diff = (dst[0] - src[0], dst[1] - src[1])
    for d, offset in enumerate(HEX_DIRECTIONS):
        if diff == offset:
            return d
    return None


def relative_direction(facing: int, target_abs_dir: int) -> int:
    """Compute relative direction (0..5) of target_abs_dir relative to facing."""
    return (target_abs_dir - facing) % 6


def absolute_direction(facing: int, rel_dir: int) -> int:
    """Convert a relative direction (0..5) from facing into an absolute hex direction (0..5)."""
    return (facing + rel_dir) % 6


def hex_angle_distance(facing_a: int, facing_b: int) -> int:
    """Return the minimum number of 60-degree hex steps (0..3) between two facings."""
    diff = (facing_b - facing_a) % 6
    return min(diff, 6 - diff)


# ============================================================================
# Camera with Cursor-Anchored Zoom & Panning
# ============================================================================

class HexCamera:
    """Manages zoom and pan transformations between flat-top axial hexes and screen pixels."""

    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.base_hex_radius = 68.0
        self.zoom = 1.0
        self.min_zoom = 0.50
        self.max_zoom = 2.60
        self.pan_x = viewport_w * 0.5
        self.pan_y = viewport_h * 0.5

    @property
    def hex_radius(self) -> float:
        return self.base_hex_radius * self.zoom

    def reset(self) -> None:
        self.zoom = 1.0
        self.pan_x = self.viewport_w * 0.5
        self.pan_y = self.viewport_h * 0.5

    def pan(self, dx: float, dy: float) -> None:
        self.pan_x += dx
        self.pan_y += dy

    def zoom_at(self, screen_x: float, screen_y: float, zoom_factor: float) -> None:
        """Zoom in or out keeping the world point under (screen_x, screen_y) fixed."""
        old_zoom = self.zoom
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * zoom_factor))
        if abs(new_zoom - old_zoom) < 1e-5:
            return
        scale_ratio = new_zoom / old_zoom
        self.pan_x = screen_x - (screen_x - self.pan_x) * scale_ratio
        self.pan_y = screen_y - (screen_y - self.pan_y) * scale_ratio
        self.zoom = new_zoom

    def hex_to_pixel(self, coord: HexCoord) -> Tuple[float, float]:
        """Convert axial (q, r) to screen center (x, y) for flat-top hex orientation."""
        q, r = coord
        size = self.hex_radius
        wx = size * 1.5 * q
        wy = size * SQRT3 * (r + 0.5 * q)
        return (self.pan_x + wx, self.pan_y + wy)

    def pixel_to_hex(self, screen_x: float, screen_y: float) -> HexCoord:
        """Convert screen pixel (screen_x, screen_y) to the nearest axial hex (q, r)."""
        size = self.hex_radius
        wx = screen_x - self.pan_x
        wy = screen_y - self.pan_y
        qf = (2.0 / 3.0 * wx) / size
        rf = ((-1.0 / 3.0 * wx) + (SQRT3 / 3.0 * wy)) / size
        return self._axial_round(qf, rf)

    @staticmethod
    def _axial_round(qf: float, rf: float) -> HexCoord:
        sf = -qf - rf
        rq = round(qf)
        rr = round(rf)
        rs = round(sf)

        q_diff = abs(rq - qf)
        r_diff = abs(rr - rf)
        s_diff = abs(rs - sf)

        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        return (int(rq), int(rr))

    def hex_polygon_points(self, coord: HexCoord, scale: float = 1.0) -> List[Tuple[float, float]]:
        """
        Return the 6 vertices of flat-top hex `coord` at angles 0°, 60°, 120°, 180°, 240°, 300°.
        """
        cx, cy = self.hex_to_pixel(coord)
        r = self.hex_radius * scale
        pts: List[Tuple[float, float]] = []
        for i in range(6):
            angle_rad = math.radians(60.0 * i)
            pts.append((cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad)))
        return pts

    def hex_edge_endpoints(
        self, coord: HexCoord, facing_dir: int, scale: float = 0.92
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the two vertex endpoints corresponding to `facing_dir` (0..5)."""
        pts = self.hex_polygon_points(coord, scale=scale)
        v_start = (facing_dir + 4) % 6
        v_end = (facing_dir + 5) % 6
        return pts[v_start], pts[v_end]
