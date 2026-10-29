"""
Shared game rules, battle state models, and turn action helpers for Denizens.

Both human UI handlers and enemy AI must use the functions in this module to
query legal actions (rotations, moves, attacks) and apply state transitions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from constants import (
    ABS_FACING_NAMES,
    ARENA_RADIUS,
    CLASS_LIZARD,
    CLASS_MAMMAL,
    CLASS_SERPENT,
    CLASS_TURTLE,
     DEBUG,
    DEFAULT_DEPLOY_LIMIT,
    FACING_N,
    FACING_NE,
    FACING_NW,
    FACING_S,
    FACING_SE,
    FACING_SW,
    FLANK_LEFT,
    FLANK_RIGHT,
    FORWARD,
    FORWARD_LEFT,
    FORWARD_RIGHT,
    FRONT,
    FRONT_LEFT,
    FRONT_RIGHT,
    JUNCTURE_AFTER_MOVE,
    JUNCTURE_AFTER_ROTATE,
    JUNCTURE_BEFORE_ROTATE,
    MAX_ARMY_SIZE,
    PLAYER_CLASSES,
    REAR,
    REINFORCEMENTS_PER_ROUND,
    REL_DIR_NAMES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TEAM_ENEMY,
    TEAM_PLAYER,
)
from creatures import (
    CLASS_LINEUPS,
    CREATURE_BY_ID,
    CreatureSpec,
    evolution_battles_required,
    get_creature_spec,
    stage_for_battles_survived,
)
from hexgrid import (
    HexCoord,
    absolute_direction,
    get_all_arena_hexes,
    hex_angle_distance,
    hex_direction_between,
    hex_distance,
    hex_neighbor,
    is_valid_hex,
    relative_direction,
)


# ============================================================================
# Unit & Battle State Models
# ============================================================================

@dataclass
class Unit:
    """
    A creature instance placed on the hex battlefield.

    Turn step & juncture tracking:
      - current_step: Index of the active Rotate-Move cycle (0 .. spec.num_rotate_moves - 1).
      - juncture: Current phase within `current_step`
        (JUNCTURE_BEFORE_ROTATE, JUNCTURE_AFTER_ROTATE, or JUNCTURE_AFTER_MOVE).
      - step_start_facing: The unit's hex facing (0..5) when `current_step` began,
        used to enforce `hex_angle_distance(step_start_facing, new_facing) <= spec.max_rotate_per_move`.
      - attacks_used: Number of attacks performed so far this turn (up to `spec.total_attacks_per_move`).
      - attacked_at_current_juncture: True if the unit already attacked at the current juncture.
    """
    unit_id: str
    spec: CreatureSpec
    team: str
    hex_pos: HexCoord
    facing: int
    hp: int = 0
    battles_survived: int = 0
    current_step: int = 0
    juncture: str = JUNCTURE_BEFORE_ROTATE
    step_start_facing: int = 0
    attacks_used: int = 0
    attacked_at_current_juncture: bool = False

    def __post_init__(self) -> None:
        self.facing = self.facing % 6
        if self.hp <= 0:
            self.hp = self.spec.max_hp
        self.step_start_facing = self.facing

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def moves_completed(self) -> int:
        """Return how many moves this unit has taken or finalized so far this turn."""
        if self.juncture == JUNCTURE_AFTER_MOVE:
            return self.current_step + 1
        return self.current_step

    @property
    def moves_remaining(self) -> int:
        """Return how many move steps the unit can still take this turn."""
        return max(0, self.spec.num_rotate_moves - self.moves_completed)

    def reset_for_turn(self) -> None:
        """Reset per-turn step, rotation, move, and attack counters."""
        self.current_step = 0
        self.juncture = JUNCTURE_BEFORE_ROTATE
        self.step_start_facing = self.facing
        self.attacks_used = 0
        self.attacked_at_current_juncture = False


@dataclass
class GameState:
    """Complete state of a Denizens tactical battle."""
    arena_radius: int = ARENA_RADIUS
    units: Dict[str, Unit] = field(default_factory=dict)
    active_team: str = TEAM_PLAYER
    turn_number: int = 1
    selected_unit_id: Optional[str] = None
    max_army_size: int = MAX_ARMY_SIZE
    reinforcements_per_round: int = REINFORCEMENTS_PER_ROUND

    def get_unit(self, unit_id: Optional[str]) -> Optional[Unit]:
        if unit_id is None:
            return None
        u = self.units.get(unit_id)
        return u if (u is not None and u.is_alive) else None

    def get_unit_at_hex(
        self, coord: HexCoord, exclude_unit_id: Optional[str] = None
    ) -> Optional[Unit]:
        """Return the living unit at `coord` (optionally excluding `exclude_unit_id`)."""
        for unit in self.units.values():
            if not unit.is_alive:
                continue
            if exclude_unit_id is not None and unit.unit_id == exclude_unit_id:
                continue
            if unit.hex_pos == coord:
                return unit
        return None

    def get_units_at_hex(self, coord: HexCoord) -> List[Unit]:
        """Return all living units currently at `coord` (may be 2 while a flying unit passes over)."""
        return [u for u in self.units.values() if u.is_alive and u.hex_pos == coord]


# ============================================================================
# Shared Rules: Flight Landing Validation, Rotations, Moves & Attack Junctures
# ============================================================================

def is_unit_overlapping(state: GameState, unit_id: str) -> bool:
    """Return True if `unit_id` is currently sharing a hex with another living unit (mid-flight)."""
    unit = state.get_unit(unit_id)
    if unit is None:
        return False
    other = state.get_unit_at_hex(unit.hex_pos, exclude_unit_id=unit_id)
    return other is not None


def can_reach_empty_landing_hex(
    state: GameState,
    start_hex: HexCoord,
    start_facing: int,
    remaining_steps: int,
    spec: CreatureSpec,
    moving_unit_id: str,
) -> bool:
    """
    Return True if a flying creature at `(start_hex, start_facing)` with `remaining_steps`
    Rotate-Move steps left can land on at least one empty hex within `1..remaining_steps` moves.
    """
    if remaining_steps <= 0:
        return state.get_unit_at_hex(start_hex, exclude_unit_id=moving_unit_id) is None

    occupied: Set[HexCoord] = {
        u.hex_pos
        for u in state.units.values()
        if u.is_alive and u.unit_id != moving_unit_id
    }

    visited: Set[Tuple[HexCoord, int, int]] = set()
    queue: deque[Tuple[HexCoord, int, int]] = deque([(start_hex, start_facing % 6, remaining_steps)])
    visited.add((start_hex, start_facing % 6, remaining_steps))

    while queue:
        cur_hex, cur_facing, steps_left = queue.popleft()
        if steps_left <= 0:
            continue

        for new_facing in range(6):
            if hex_angle_distance(cur_facing, new_facing) > spec.max_rotate_per_move:
                continue
            for rel_dir in spec.legal_move_dirs:
                abs_dir = absolute_direction(new_facing, rel_dir)
                nxt_hex = hex_neighbor(cur_hex, abs_dir)
                if not is_valid_hex(nxt_hex[0], nxt_hex[1], state.arena_radius):
                    continue
                if nxt_hex not in occupied:
                    return True
                if spec.can_fly and steps_left - 1 > 0:
                    state_key = (nxt_hex, new_facing, steps_left - 1)
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append(state_key)

    return False


def _rotation_step_context(
    state: GameState, unit: Unit
) -> Optional[Tuple[int, int, bool]]:
    """
    Determine which Rotate-Move step a rotation by `unit` would apply to.

    Returns `(target_step, ref_facing, advances_to_next_step)` or `None` if rotating is illegal.
    """
    spec = unit.spec
    if spec.max_rotate_per_move <= 0 or spec.num_rotate_moves <= 0:
        return None

    if unit.juncture == JUNCTURE_BEFORE_ROTATE or (
        unit.juncture == JUNCTURE_AFTER_ROTATE and not unit.attacked_at_current_juncture
    ):
        if unit.current_step < spec.num_rotate_moves:
            return (unit.current_step, unit.step_start_facing, False)
        return None

    next_step = unit.current_step + 1
    if next_step < spec.num_rotate_moves:
        return (next_step, unit.facing, True)

    return None


def get_legal_rotations(state: GameState, unit_id: str) -> List[int]:
    """
    Return the list of legal absolute facings (0..5) that `unit_id` can rotate to right now.
    """
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != state.active_team:
        return []

    ctx = _rotation_step_context(state, unit)
    if ctx is None:
        return []

    target_step, ref_facing, _ = ctx
    spec = unit.spec
    overlapping = is_unit_overlapping(state, unit_id)
    steps_remaining_from_target = spec.num_rotate_moves - target_step

    legal_facings: List[int] = []
    for cand_facing in range(6):
        if hex_angle_distance(ref_facing, cand_facing) > spec.max_rotate_per_move:
            continue
        if overlapping:
            can_land = False
            for rel_dir in spec.legal_move_dirs:
                abs_dir = absolute_direction(cand_facing, rel_dir)
                nxt_hex = hex_neighbor(unit.hex_pos, abs_dir)
                if not is_valid_hex(nxt_hex[0], nxt_hex[1], state.arena_radius):
                    continue
                if state.get_unit_at_hex(nxt_hex, exclude_unit_id=unit_id) is None:
                    can_land = True
                    break
                if spec.can_fly and (steps_remaining_from_target - 1) > 0:
                    if can_reach_empty_landing_hex(
                        state, nxt_hex, cand_facing, steps_remaining_from_target - 1, spec, unit_id
                    ):
                        can_land = True
                        break
            if not can_land:
                continue

        legal_facings.append(cand_facing)

    return legal_facings


def rotate_unit(state: GameState, unit_id: str, new_facing: int) -> bool:
    """
    Rotate `unit_id` to `new_facing` (0..5) if legal under its CreatureSpec attributes.
    Returns True if the rotation was applied.
    """
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != state.active_team:
        return False

    new_facing = new_facing % 6
    if new_facing == unit.facing:
        return False

    legal_facings = get_legal_rotations(state, unit_id)
    if new_facing not in legal_facings:
        return False

    ctx = _rotation_step_context(state, unit)
    if ctx is None:
        return False

    target_step, ref_facing, advances_to_next_step = ctx
    if advances_to_next_step:
        unit.current_step = target_step
        unit.step_start_facing = ref_facing

    unit.facing = new_facing
    if unit.facing == unit.step_start_facing and not unit.attacked_at_current_juncture:
        unit.juncture = JUNCTURE_BEFORE_ROTATE
    else:
        if unit.juncture != JUNCTURE_AFTER_ROTATE or advances_to_next_step:
            unit.juncture = JUNCTURE_AFTER_ROTATE
            unit.attacked_at_current_juncture = False

    return True


def rotate_unit_by_step(state: GameState, unit_id: str, delta_steps: int) -> bool:
    """Rotate `unit_id` by `delta_steps` (-1 for 60° CCW, +1 for 60° CW) if legal."""
    unit = state.get_unit(unit_id)
    if unit is None:
        return False
    target_facing = (unit.facing + delta_steps) % 6
    return rotate_unit(state, unit_id, target_facing)


def _move_step_index(unit: Unit) -> Optional[int]:
    """Return the Rotate-Move step index (0 .. num_rotate_moves - 1) that a move now would use."""
    if unit.juncture in (JUNCTURE_BEFORE_ROTATE, JUNCTURE_AFTER_ROTATE):
        target_step = unit.current_step
    else:
        target_step = unit.current_step + 1

    if target_step < unit.spec.num_rotate_moves:
        return target_step
    return None


def get_legal_moves(state: GameState, unit_id: str) -> List[HexCoord]:
    """
    Return all legal destination hex coordinates that `unit_id` can move into right now,
    relative to its instantaneous facing.
    """
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != state.active_team:
        return []

    target_step = _move_step_index(unit)
    if target_step is None:
        return []

    spec = unit.spec
    steps_left_after = (spec.num_rotate_moves - 1) - target_step
    legal_hexes: List[HexCoord] = []

    for rel_dir in sorted(spec.legal_move_dirs):
        abs_dir = absolute_direction(unit.facing, rel_dir)
        dest = hex_neighbor(unit.hex_pos, abs_dir)
        if not is_valid_hex(dest[0], dest[1], state.arena_radius):
            continue

        occupant = state.get_unit_at_hex(dest, exclude_unit_id=unit_id)
        if occupant is None:
            legal_hexes.append(dest)
        elif spec.can_fly and steps_left_after >= 1:
            if can_reach_empty_landing_hex(
                state, dest, unit.facing, steps_left_after, spec, unit_id
            ):
                legal_hexes.append(dest)

    return legal_hexes


def move_unit(state: GameState, unit_id: str, target_hex: HexCoord) -> bool:
    """
    Move `unit_id` into `target_hex` if legal under its CreatureSpec attributes.
    Returns True if the move was executed.
    """
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != state.active_team:
        return False

    legal_moves = get_legal_moves(state, unit_id)
    if target_hex not in legal_moves:
        return False

    target_step = _move_step_index(unit)
    if target_step is None:
        return False

    if target_step != unit.current_step:
        unit.current_step = target_step
        unit.step_start_facing = unit.facing

    unit.hex_pos = target_hex
    unit.juncture = JUNCTURE_AFTER_MOVE
    unit.attacked_at_current_juncture = False
    return True


def can_unit_attack_now(state: GameState, unit_id: str) -> bool:
    """
    Return True if `unit_id` is eligible to take an attack at the current juncture
    (BEFORE_ROTATE, AFTER_ROTATE, or AFTER_MOVE), up to `spec.total_attacks_per_move`.
    """
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != state.active_team:
        return False
    if is_unit_overlapping(state, unit_id):
        return False
    if unit.attacks_used >= unit.spec.total_attacks_per_move:
        return False
    if unit.attacked_at_current_juncture:
        return False
    return True


def get_attack_arc_hexes(state: GameState, unit_id: str) -> List[HexCoord]:
    """Return the adjacent arena hexes covered by `unit_id`'s offensive `attack_arcs`."""
    unit = state.get_unit(unit_id)
    if unit is None:
        return []
    hexes: List[HexCoord] = []
    for rel_dir in sorted(unit.spec.attack_arcs):
        abs_dir = absolute_direction(unit.facing, rel_dir)
        target = hex_neighbor(unit.hex_pos, abs_dir)
        if is_valid_hex(target[0], target[1], state.arena_radius):
            hexes.append(target)
    return hexes


def record_juncture_attack(state: GameState, unit_id: str) -> bool:
    """
    Mark that `unit_id` has used an attack at its current juncture.
    Returns True if allowed and recorded.
    """
    if not can_unit_attack_now(state, unit_id):
        return False
    unit = state.get_unit(unit_id)
    assert unit is not None
    unit.attacks_used += 1
    unit.attacked_at_current_juncture = True
    return True


def can_end_turn(state: GameState) -> bool:
    """Return True if no unit is currently mid-flight overlapping another unit."""
    for unit in state.units.values():
        if unit.is_alive and is_unit_overlapping(state, unit.unit_id):
            return False
    return True


def end_turn(state: GameState) -> bool:
    """
    Advance the active team turn and reset turn counters for all living units.
    Returns False if a flying unit is still overlapping another unit's hex.
    """
    if not can_end_turn(state):
        return False

    if state.active_team == TEAM_PLAYER:
        state.active_team = TEAM_ENEMY
    else:
        state.active_team = TEAM_PLAYER
        state.turn_number += 1

    for unit in state.units.values():
        if unit.is_alive:
            unit.reset_for_turn()

    return True


def cycle_unit_stage(state: GameState, unit_id: str, delta: int) -> bool:
    """
    Debug helper: cycle `unit_id`'s evolution stage (1..8 for player classes, 1..13 for mammals)
    and reset its turn state so the new stage's capabilities can be tested immediately.
    """
    unit = state.get_unit(unit_id)
    if unit is None:
        return False

    lineup = CLASS_LINEUPS[unit.spec.creature_class]
    new_idx = (unit.spec.stage - 1 + delta) % len(lineup)
    new_spec = lineup[new_idx]
    unit.spec = new_spec
    unit.hp = new_spec.max_hp
    if unit.spec.creature_class in PLAYER_CLASSES:
        unit.battles_survived = evolution_battles_required(new_spec.stage)
    unit.reset_for_turn()
    return True


# ============================================================================
# Initial Battle Setup Helper
# ============================================================================

def create_demo_battle() -> GameState:
    """
    Create an initial 61-hex battle state with a player reptile squad (Serpent, Lizard, Turtle)
    in the southern half facing North and opposing Mammal enemies in the northern half facing South.
    """
    state = GameState(arena_radius=ARENA_RADIUS)

    player_placements = [
        ("p_serpent_1", get_creature_spec(CLASS_SERPENT, 1), (-2, 3), FACING_N),
        ("p_lizard_1", get_creature_spec(CLASS_LIZARD, 1), (0, 3), FACING_N),
        ("p_turtle_1", get_creature_spec(CLASS_TURTLE, 1), (2, 2), FACING_N),
        ("p_lizard_8", get_creature_spec(CLASS_LIZARD, 8), (-1, 2), FACING_N),
        ("p_serpent_8", get_creature_spec(CLASS_SERPENT, 8), (1, 1), FACING_N),
    ]

    enemy_placements = [
        ("e_mammal_1", get_creature_spec(CLASS_MAMMAL, 1), (-1, -2), FACING_S),
        ("e_mammal_2", get_creature_spec(CLASS_MAMMAL, 2), (1, -3), FACING_S),
        ("e_mammal_5", get_creature_spec(CLASS_MAMMAL, 5), (-2, -1), FACING_SE),
        ("e_mammal_8", get_creature_spec(CLASS_MAMMAL, 8), (2, -2), FACING_SW),
        ("e_mammal_10", get_creature_spec(CLASS_MAMMAL, 10), (0, -1), FACING_S),
    ]

    for uid, spec, pos, facing in player_placements:
        u = Unit(
            unit_id=uid,
            spec=spec,
            team=TEAM_PLAYER,
            hex_pos=pos,
            facing=facing,
            battles_survived=evolution_battles_required(spec.stage),
        )
        state.units[uid] = u

    for uid, spec, pos, facing in enemy_placements:
        u = Unit(
            unit_id=uid,
            spec=spec,
            team=TEAM_ENEMY,
            hex_pos=pos,
            facing=facing,
        )
        state.units[uid] = u

    state.selected_unit_id = "p_lizard_1"
    return state
