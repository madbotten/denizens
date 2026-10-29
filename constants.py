"""
Global configuration, hex direction constants, and UI color palette for Denizens.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# ============================================================================
# Global Constants & Strategic Configuration
# ============================================================================

DEBUG: bool = True
ENABLE_CAMERA_ZOOM_PAN: bool = False

SCREEN_WIDTH: int = 2200
SCREEN_HEIGHT: int = 1200

RESERVE_PANEL_WIDTH: int = 340
SIDEBAR_WIDTH: int = 620
MAP_VIEWPORT_X: int = RESERVE_PANEL_WIDTH
MAP_VIEWPORT_WIDTH: int = SCREEN_WIDTH - RESERVE_PANEL_WIDTH - SIDEBAR_WIDTH
SIDEBAR_X: int = SCREEN_WIDTH - SIDEBAR_WIDTH

RESERVE_COLS: int = 2
RESERVE_ROWS: int = 8  # MAX_ARMY_SIZE // 2

ARENA_RADIUS: int = 4  # 61-hex flat-top hexagonal arena
MAX_ARMY_SIZE: int = 16
INITIAL_ARMY_SIZE: int = 6
REINFORCEMENTS_BASE: int = 4
REINFORCEMENTS_PER_ROUND: int = REINFORCEMENTS_BASE
DEFAULT_DEPLOY_LIMIT: int = MAX_ARMY_SIZE

# Wave Point Budget Constants
WAVE_BASE_POINTS: int = 4      # Round 1 budget = 4 pts (4 Rats @ 1 pt each)
WAVE_POINTS_LINEAR: int = 3    # Linear growth per round in Advanced mode
WAVE_POINTS_QUAD: int = 1      # Quadratic growth per round in Advanced mode

# Game Modes & Battle Phases
GAME_MODE_BASIC: str = "basic"
GAME_MODE_INTERMEDIATE: str = "intermediate"
GAME_MODE_ADVANCED: str = "advanced"

PHASE_SPLASH: str = "SPLASH"
PHASE_INSTRUCTIONS: str = "INSTRUCTIONS"
PHASE_SETUP: str = "SETUP"
PHASE_BATTLE: str = "BATTLE"
PHASE_EVOLUTION: str = "EVOLUTION"
PHASE_DEFEAT: str = "DEFEAT"

AI_STEP_DELAY_MS: int = 240  # Milliseconds between visible Enemy Mammal AI steps

# Teams
TEAM_PLAYER: str = "player"
TEAM_ENEMY: str = "enemy"

# Classes
CLASS_SERPENT: str = "serpent"
CLASS_LIZARD: str = "lizard"
CLASS_TURTLE: str = "turtle"
CLASS_MAMMAL: str = "mammal"

PLAYER_CLASSES: Tuple[str, ...] = (CLASS_SERPENT, CLASS_LIZARD, CLASS_TURTLE)

# ============================================================================
# Hex Facings & Relative Directions
# ============================================================================

# Absolute hex facings (0..5), clockwise starting from North (12 o'clock / 0 deg)
FACING_N: int = 0   # 0°   (Top flat edge)
FACING_NE: int = 1  # 60°  (Top-Right edge)
FACING_SE: int = 2  # 120° (Bottom-Right edge)
FACING_S: int = 3   # 180° (Bottom flat edge)
FACING_SW: int = 4  # 240° (Bottom-Left edge)
FACING_NW: int = 5  # 300° (Top-Left edge)

ABS_FACING_NAMES: Dict[int, str] = {
    0: "N (0°)",
    1: "NE (60°)",
    2: "SE (120°)",
    3: "S (180°)",
    4: "SW (240°)",
    5: "NW (300°)",
}

# Axial coordinate neighbor offsets (dq, dr) for flat-top hexes
HEX_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, -1),   # 0: N
    (1, -1),   # 1: NE
    (1, 0),    # 2: SE
    (0, 1),    # 3: S
    (-1, 1),   # 4: SW
    (-1, 0),   # 5: NW
)

# Relative directions (0..5) relative to a unit's current facing
FORWARD: int = 0
FRONT: int = 0
FORWARD_RIGHT: int = 1
FRONT_RIGHT: int = 1
FLANK_RIGHT: int = 2
REAR: int = 3
FLANK_LEFT: int = 4
FORWARD_LEFT: int = 5
FRONT_LEFT: int = 5

REL_DIR_NAMES: Dict[int, str] = {
    FORWARD: "Forward",
    FORWARD_RIGHT: "Forward-Right",
    FLANK_RIGHT: "Flank-Right",
    REAR: "Rear",
    FLANK_LEFT: "Flank-Left",
    FORWARD_LEFT: "Forward-Left",
}

# Standard direction sets
DIRS_FORWARD_ONLY: FrozenSet[int] = frozenset({FORWARD})
DIRS_FORWARD_REAR: FrozenSet[int] = frozenset({FORWARD, REAR})
DIRS_FRONT_ARC: FrozenSet[int] = frozenset({FORWARD, FORWARD_LEFT, FORWARD_RIGHT})
DIRS_FRONT_ARC_AND_REAR: FrozenSet[int] = frozenset({FORWARD, FORWARD_LEFT, FORWARD_RIGHT, REAR})
DIRS_ALL_SIX: FrozenSet[int] = frozenset(
    {FORWARD, FORWARD_RIGHT, FLANK_RIGHT, REAR, FLANK_LEFT, FORWARD_LEFT}
)
DEFAULT_VULNERABLE_ARCS: FrozenSet[int] = frozenset({FLANK_LEFT, FLANK_RIGHT, REAR})

# Turn Junctures within each Rotate-Move cycle
JUNCTURE_BEFORE_ROTATE: str = "BEFORE_ROTATE"
JUNCTURE_AFTER_ROTATE: str = "AFTER_ROTATE"
JUNCTURE_AFTER_MOVE: str = "AFTER_MOVE"
JUNCTURE_AFTER_POST_ROTATE: str = "AFTER_POST_ROTATE"

# ============================================================================
# Visual Theme & Color Palette
# ============================================================================

COLOR_BG_VOID = (14, 16, 20)
COLOR_HEX_FILL_A = (30, 35, 43)
COLOR_HEX_FILL_B = (34, 40, 49)
COLOR_HEX_CENTER = (40, 46, 56)
COLOR_HEX_DEPLOY_A = (32, 46, 42)
COLOR_HEX_DEPLOY_B = (36, 52, 47)
COLOR_HEX_DEPLOY_BORDER = (62, 118, 92)
COLOR_HEX_BORDER = (66, 76, 90)
COLOR_HEX_HOVER = (110, 130, 155)

COLOR_SELECTED_HEX = (230, 190, 80)
COLOR_LEGAL_MOVE = (55, 205, 115)
COLOR_LEGAL_FLY_OVER = (70, 195, 245)
COLOR_ATTACK_ARC = (230, 85, 75)
COLOR_VULNERABLE_EDGE = (210, 105, 55)
COLOR_FRONT_EDGE = (90, 225, 255)

COLOR_PLAYER_RING = (65, 190, 115)
COLOR_ENEMY_RING = (215, 75, 75)

COLOR_PANEL_BG = (22, 25, 31)
COLOR_PANEL_BORDER = (88, 78, 62)
COLOR_CARD_BG = (30, 34, 42)
COLOR_SLOT_EMPTY = (24, 28, 35)
COLOR_SLOT_DEPLOYED = (26, 36, 34)
COLOR_TEXT_MAIN = (232, 228, 218)
COLOR_TEXT_MUTED = (160, 166, 178)
COLOR_TEXT_GOLD = (230, 192, 98)
COLOR_TEXT_CYAN = (100, 215, 245)
COLOR_TEXT_RED = (235, 100, 95)
COLOR_TEXT_GREEN = (95, 220, 135)
