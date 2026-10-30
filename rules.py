"""
Shared game rules, battle state models, wave point purchasing, army deployment,
and turn action helpers for Denizens.

Both human UI handlers and enemy AI must use the functions in this module to
query legal actions (deployments, rotations, moves, attacks) and apply state transitions.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


def default_rng() -> random.Random:
    """
    Create a fresh `random.Random` instance seeded from `os.urandom(16)` (falling back
    to `random.getrandbits(128)`), ensuring full entropy in both desktop Python and
    Pygbag WebAssembly (WASM) environments.
    """
    try:
        return random.Random(os.urandom(16))
    except Exception:
        return random.Random(random.getrandbits(128))

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
    GAME_MODE_ADVANCED,
    GAME_MODE_BASIC,
    GAME_MODE_INTERMEDIATE,
    INITIAL_ARMY_SIZE,
    JUNCTURE_AFTER_MOVE,
    JUNCTURE_AFTER_POST_ROTATE,
    JUNCTURE_AFTER_ROTATE,
    JUNCTURE_BEFORE_ROTATE,
    MAX_ARMY_SIZE,
    PHASE_BATTLE,
    PHASE_DEFEAT,
    PHASE_EVOLUTION,
    PHASE_INSTRUCTIONS,
    PHASE_SETUP,
    PHASE_SPLASH,
    PLAYER_CLASSES,
    REAR,
    REINFORCEMENTS_BASE,
    REINFORCEMENTS_PER_ROUND,
    REL_DIR_NAMES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TEAM_ENEMY,
    TEAM_PLAYER,
    WAVE_BASE_POINTS,
    WAVE_POINTS_LINEAR,
    WAVE_POINTS_QUAD,
)
from creatures import (
    CLASS_LINEUPS,
    CREATURE_BY_ID,
    MAMMAL_LINEUP,
    CreatureSpec,
    evolution_battles_required,
    get_creature_spec,
    get_unlocked_mammals,
    is_mammal_unlocked,
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
    is_lower_half_hex,
    is_upper_half_hex,
    is_valid_hex,
    relative_direction,
)


# ============================================================================
# Unit & Battle State Models
# ============================================================================

@dataclass
class Unit:
    """
    A creature instance in the player's army or on the hex battlefield.

    When `hex_pos` is `None`, the unit is off the battlefield in the left-hand
    Army Reserve Tray (at `reserve_slot` 0 .. MAX_ARMY_SIZE - 1).
    """
    unit_id: str
    spec: CreatureSpec
    team: str
    hex_pos: Optional[HexCoord]
    facing: int
    hp: int = 0
    battles_survived: int = 0
    reserve_slot: Optional[int] = None
    current_step: int = 0
    juncture: str = JUNCTURE_BEFORE_ROTATE
    step_start_facing: int = 0
    attacks_used: int = 0
    attacked_at_current_juncture: bool = False
    poison_stacks: int = 0

    def __post_init__(self) -> None:
        self.facing = self.facing % 6
        if self.hp <= 0:
            self.hp = self.spec.max_hp
        self.step_start_facing = self.facing

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def max_hp(self) -> int:
        return self.spec.max_hp

    @property
    def is_deployed(self) -> bool:
        return self.is_alive and self.hex_pos is not None

    @property
    def in_reserve(self) -> bool:
        return self.is_alive and self.hex_pos is None

    @property
    def moves_completed(self) -> int:
        """Return how many moves this unit has taken or finalized so far this turn."""
        if self.juncture in (JUNCTURE_AFTER_MOVE, JUNCTURE_AFTER_POST_ROTATE):
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
class EvolutionRecord:
    """Tracks a surviving player creature's status and potential stage upgrade during PHASE_EVOLUTION."""
    unit_id: str
    old_spec: CreatureSpec
    new_spec: CreatureSpec
    battles_survived: int
    leveled_up: bool


@dataclass
class AttackResult:
    """Records the outcome of a directional melee/breath attack against a single target."""
    attacker_id: str
    attacker_name: str
    defender_id: str
    defender_name: str
    target_hex: HexCoord
    damage: int
    incoming_rel_dir: int
    is_vulnerable: bool
    defender_slain: bool
    retaliation_damage: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class AIActionRecord:
    """Records a single atomic Enemy Mammal AI action executed via shared `rules.py` helpers."""
    action_type: str  # "rotate", "move", "attack", or "end_turn"
    unit_id: Optional[str]
    unit_name: str = ""
    target_hex: Optional[HexCoord] = None
    new_facing: Optional[int] = None
    attack_results: List[AttackResult] = field(default_factory=list)
    description: str = ""


@dataclass
class GameState:
    """Complete state of a Denizens tactical battle and persistent army."""
    arena_radius: int = ARENA_RADIUS
    units: Dict[str, Unit] = field(default_factory=dict)
    phase: str = PHASE_BATTLE
    game_mode: str = GAME_MODE_BASIC
    round_number: int = 1
    wave_budget: int = WAVE_BASE_POINTS
    active_team: str = TEAM_PLAYER
    turn_number: int = 1
    selected_unit_id: Optional[str] = None
    max_army_size: int = MAX_ARMY_SIZE
    reinforcements_per_round: int = REINFORCEMENTS_BASE
    next_unit_seq: int = 1
    enemy_ai_enabled: bool = True
    ai_acted_unit_ids: Set[str] = field(default_factory=set)
    evolution_records: List[EvolutionRecord] = field(default_factory=list)
    last_attack_results: List[AttackResult] = field(default_factory=list)
    active_bot_id: str = "legacy_classic"
    active_bot_name: str = "Classic Bot"
    active_bot_ply: int = 1

    def get_unit(self, unit_id: Optional[str]) -> Optional[Unit]:
        if unit_id is None:
            return None
        u = self.units.get(unit_id)
        return u if (u is not None and u.is_alive) else None

    def get_unit_at_hex(
        self, coord: HexCoord, exclude_unit_id: Optional[str] = None
    ) -> Optional[Unit]:
        """Return the living deployed unit at `coord` (optionally excluding `exclude_unit_id`)."""
        for unit in self.units.values():
            if not unit.is_deployed:
                continue
            if exclude_unit_id is not None and unit.unit_id == exclude_unit_id:
                continue
            if unit.hex_pos == coord:
                return unit
        return None

    def get_units_at_hex(self, coord: HexCoord) -> List[Unit]:
        """Return all living deployed units currently at `coord`."""
        return [u for u in self.units.values() if u.is_deployed and u.hex_pos == coord]

    def get_player_army_units(self) -> List[Unit]:
        """Return all living player units (both in reserve and deployed), ordered by reserve_slot."""
        player_units = [
            u for u in self.units.values() if u.is_alive and u.team == TEAM_PLAYER
        ]
        player_units.sort(
            key=lambda u: (u.reserve_slot if u.reserve_slot is not None else 999, u.unit_id)
        )
        return player_units

    def get_unit_by_reserve_slot(self, slot_idx: int) -> Optional[Unit]:
        """Return the living player unit assigned to `slot_idx` (0 .. max_army_size - 1)."""
        for u in self.units.values():
            if u.is_alive and u.team == TEAM_PLAYER and u.reserve_slot == slot_idx:
                return u
        return None


def clone_game_state(state: GameState) -> GameState:
    """
    Create a fast, independent deep copy of `state` for X-ply lookahead search.
    Reuses immutable `CreatureSpec` references while cloning all mutable `Unit`
    and `GameState` fields.
    """
    cloned_units: Dict[str, Unit] = {}
    for uid, u in state.units.items():
        cloned_units[uid] = Unit(
            unit_id=u.unit_id,
            spec=u.spec,
            team=u.team,
            hex_pos=u.hex_pos,
            facing=u.facing,
            hp=u.hp,
            battles_survived=u.battles_survived,
            reserve_slot=u.reserve_slot,
            current_step=u.current_step,
            juncture=u.juncture,
            step_start_facing=u.step_start_facing,
            attacks_used=u.attacks_used,
            attacked_at_current_juncture=u.attacked_at_current_juncture,
            poison_stacks=u.poison_stacks,
        )
        # Preserve exact hp==0 for slain units (Unit.__post_init__ resets <=0 to max_hp)
        cloned_units[uid].hp = u.hp
        cloned_units[uid].step_start_facing = u.step_start_facing

    return GameState(
        arena_radius=state.arena_radius,
        units=cloned_units,
        phase=state.phase,
        game_mode=state.game_mode,
        round_number=state.round_number,
        wave_budget=state.wave_budget,
        active_team=state.active_team,
        turn_number=state.turn_number,
        selected_unit_id=state.selected_unit_id,
        max_army_size=state.max_army_size,
        reinforcements_per_round=state.reinforcements_per_round,
        next_unit_seq=state.next_unit_seq,
        enemy_ai_enabled=state.enemy_ai_enabled,
        ai_acted_unit_ids=set(state.ai_acted_unit_ids),
        evolution_records=list(state.evolution_records),
        last_attack_results=list(state.last_attack_results),
        active_bot_id=state.active_bot_id,
        active_bot_name=state.active_bot_name,
        active_bot_ply=state.active_bot_ply,
    )


# ============================================================================
# Army Setup & Battlefield Deployment Helpers (Player & Team-Symmetric)
# ============================================================================

def is_valid_team_deploy_hex(state: GameState, team: str, coord: HexCoord) -> bool:
    """
    Return True if `coord` is a legal deployment hex for `team` during PHASE_SETUP:
      - `TEAM_PLAYER`: lower half (`2 * r + q > 0`)
      - `TEAM_ENEMY`: upper half (`2 * r + q < 0`)
    """
    if not is_valid_hex(coord[0], coord[1], state.arena_radius):
        return False
    if team == TEAM_PLAYER:
        if not is_lower_half_hex(coord):
            return False
    elif team == TEAM_ENEMY:
        if not is_upper_half_hex(coord):
            return False
    else:
        return False

    occupant = state.get_unit_at_hex(coord)
    if occupant is not None and occupant.team != team:
        return False
    return True


def get_valid_team_deploy_hexes(state: GameState, team: str) -> List[HexCoord]:
    """Return all 28 half-arena hexes where `team` may deploy creatures."""
    return [
        c
        for c in get_all_arena_hexes(state.arena_radius)
        if is_valid_team_deploy_hex(state, team, c)
    ]


def deploy_team_unit_to_hex(
    state: GameState,
    unit_id: str,
    target_hex: HexCoord,
    facing: Optional[int] = None,
) -> bool:
    """
    Place or reposition a unit (`TEAM_PLAYER` in lower half or `TEAM_ENEMY` in upper half)
    onto `target_hex` during `PHASE_SETUP` via shared rules validation.
    """
    if state.phase != PHASE_SETUP:
        return False
    unit = state.get_unit(unit_id)
    if unit is None:
        return False
    if not is_valid_team_deploy_hex(state, unit.team, target_hex):
        return False

    old_hex = unit.hex_pos
    existing = state.get_unit_at_hex(target_hex, exclude_unit_id=unit_id)
    if existing is not None:
        if existing.team != unit.team:
            return False
        existing.hex_pos = old_hex
        existing.reset_for_turn()

    unit.hex_pos = target_hex
    if facing is not None:
        unit.facing = facing % 6
    elif old_hex is None:
        unit.facing = FACING_N if unit.team == TEAM_PLAYER else FACING_S
    unit.reset_for_turn()
    return True


def is_valid_player_deploy_hex(state: GameState, coord: HexCoord) -> bool:
    """
    Return True if `coord` is a legal hex in the lower half of the battlefield
    (`2 * r + q > 0`) for the human player to place a unit during PHASE_SETUP.
    """
    if not is_valid_hex(coord[0], coord[1], state.arena_radius):
        return False
    if not is_lower_half_hex(coord):
        return False
    occupant = state.get_unit_at_hex(coord)
    if occupant is not None and occupant.team != TEAM_PLAYER:
        return False
    return True


def get_valid_player_deploy_hexes(state: GameState) -> List[HexCoord]:
    """Return all 28 lower-half hexes where the player may deploy creatures."""
    return [
        c
        for c in get_all_arena_hexes(state.arena_radius)
        if is_valid_player_deploy_hex(state, c)
    ]


def deploy_unit_to_hex(state: GameState, unit_id: str, target_hex: HexCoord) -> bool:
    """
    Place or reposition a player creature onto `target_hex` in the lower half of the
    battlefield during PHASE_SETUP (one creature per hex; swaps if occupied by another
    player unit).
    """
    if state.phase != PHASE_SETUP:
        return False
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != TEAM_PLAYER:
        return False
    if not is_valid_player_deploy_hex(state, target_hex):
        return False

    old_hex = unit.hex_pos
    existing = state.get_unit_at_hex(target_hex, exclude_unit_id=unit_id)
    if existing is not None:
        if existing.team != TEAM_PLAYER:
            return False
        # Swap or return existing unit to reserve
        existing.hex_pos = old_hex
        existing.reset_for_turn()

    unit.hex_pos = target_hex
    if old_hex is None:
        unit.facing = FACING_N
    unit.reset_for_turn()
    return True


def undeploy_unit_to_reserve(
    state: GameState, unit_id: str, preferred_slot: Optional[int] = None
) -> bool:
    """Return a deployed player creature back to the left-hand Army Reserve Tray during PHASE_SETUP."""
    if state.phase != PHASE_SETUP:
        return False
    unit = state.get_unit(unit_id)
    if unit is None or unit.team != TEAM_PLAYER or unit.hex_pos is None:
        return False

    if preferred_slot is not None and 0 <= preferred_slot < state.max_army_size:
        occupant = state.get_unit_by_reserve_slot(preferred_slot)
        if occupant is None:
            unit.reserve_slot = preferred_slot
        elif occupant.unit_id != unit.unit_id and occupant.is_deployed:
            occupant.reserve_slot = unit.reserve_slot
            unit.reserve_slot = preferred_slot

    unit.hex_pos = None
    unit.facing = FACING_N
    unit.reset_for_turn()
    return True


def can_confirm_deployment(state: GameState) -> bool:
    """
    Return True if we are in PHASE_SETUP and ALL living player units in the army
    have been deployed onto the lower half of the battlefield (no reserves allowed).
    """
    if state.phase != PHASE_SETUP:
        return False
    player_units = state.get_player_army_units()
    if not player_units:
        return False
    return all(u.is_deployed for u in player_units)


def confirm_deployment(state: GameState) -> bool:
    """Lock in the player's deployed army and transition from PHASE_SETUP to PHASE_BATTLE."""
    if not can_confirm_deployment(state):
        return False
    state.phase = PHASE_BATTLE
    state.active_team = TEAM_PLAYER
    state.turn_number = 1
    state.ai_acted_unit_ids.clear()
    adv_ctrl = getattr(state, "_advanced_bot_controller", None)
    if adv_ctrl is not None:
        adv_ctrl.reset_for_battle()
    for unit in state.units.values():
        if unit.is_alive:
            unit.reset_for_turn()

    # Ensure selected_unit_id points to a deployed unit if possible
    sel = state.get_unit(state.selected_unit_id)
    if sel is None or not sel.is_deployed:
        deployed_players = [
            u for u in state.get_player_army_units() if u.is_deployed
        ]
        if deployed_players:
            state.selected_unit_id = deployed_players[0].unit_id
    return True


def complete_battle_to_evolution(state: GameState) -> bool:
    """
    Transition from PHASE_BATTLE to PHASE_EVOLUTION.
    Records each surviving player reptile's pre-evolution spec (`old_spec`) and
    post-battle spec (`new_spec`) based on triangular `battles_survived` progression.
    """
    if state.phase != PHASE_BATTLE:
        return False

    survivors = state.get_player_army_units()
    records: List[EvolutionRecord] = []
    for u in survivors:
        old_spec = u.spec
        u.battles_survived += 1
        if old_spec.creature_class in PLAYER_CLASSES:
            new_stage = stage_for_battles_survived(u.battles_survived)
            new_spec = get_creature_spec(old_spec.creature_class, new_stage)
            leveled_up = new_spec.stage > old_spec.stage
        else:
            new_spec = old_spec
            leveled_up = False
        u.spec = new_spec
        u.hp = new_spec.max_hp
        u.poison_stacks = 0
        u.reset_for_turn()
        records.append(
            EvolutionRecord(
                unit_id=u.unit_id,
                old_spec=old_spec,
                new_spec=new_spec,
                battles_survived=u.battles_survived,
                leveled_up=leveled_up,
            )
        )

    state.evolution_records = records
    state.phase = PHASE_EVOLUTION
    if survivors:
        state.selected_unit_id = survivors[0].unit_id
    return True


def confirm_evolution_phase(
    state: GameState, rng: Optional[random.Random] = None
) -> bool:
    """
    Confirm the Evolution Phase (`PHASE_EVOLUTION`), returning all surviving (and newly evolved)
    player units to the Left Army Reserve Tray, granting `reinforcements_per_round` (4) new
    Stage 1 reptiles, spawning the next round's Mammal wave, and returning to `PHASE_SETUP`.
    """
    if state.phase != PHASE_EVOLUTION:
        return False
    if rng is None:
        rng = default_rng()

    survivors = state.get_player_army_units()
    for idx, u in enumerate(survivors):
        u.hex_pos = None
        u.facing = FACING_N
        u.hp = u.spec.max_hp
        u.poison_stacks = 0
        u.reserve_slot = idx
        u.reset_for_turn()

    state.evolution_records.clear()
    state.round_number += 1
    state.turn_number = 1
    state.active_team = TEAM_PLAYER

    grant_base_reinforcements(
        state, count=state.reinforcements_per_round, rng=rng
    )
    spawn_enemy_wave_on_board(state, rng=rng)
    state.phase = PHASE_SETUP

    army = state.get_player_army_units()
    if army:
        state.selected_unit_id = army[0].unit_id
    return True


# ============================================================================
# Points-Based Wave Purchasing & Base Reinforcements
# ============================================================================

def calculate_player_army_points(state: GameState) -> int:
    """Return the sum of `point_cost` across all living player units (reserve + deployed)."""
    return sum(u.spec.point_cost for u in state.get_player_army_units())


def enemy_wave_budget(state: GameState) -> int:
    """
    Compute the computer opponent's Mammal wave point budget for `state.round_number`:
      - Round 1: Always `WAVE_BASE_POINTS` (4 points -> 4 Rats).
      - GAME_MODE_INTERMEDIATE & GAME_MODE_ADVANCED:
          Fixed mathematical schedule `4 + 3*(R-1) + (R-1)^2`.
      - GAME_MODE_BASIC:
          Adaptive budget scaled to the player's surviving army points
          so battles always challenge the player without overwhelming them.
    """
    r = max(1, state.round_number)
    if r == 1:
        return WAVE_BASE_POINTS

    if state.game_mode in (GAME_MODE_INTERMEDIATE, GAME_MODE_ADVANCED):
        return WAVE_BASE_POINTS + WAVE_POINTS_LINEAR * (r - 1) + WAVE_POINTS_QUAD * (r - 1) ** 2

    # GAME_MODE_BASIC: Adaptive based on player's surviving army strength
    player_pts = calculate_player_army_points(state)
    adaptive = int(round(player_pts * 0.80)) + min(r - 1, 6)
    return max(WAVE_BASE_POINTS, adaptive)


def _get_or_create_advanced_bot_controller(state: GameState):
    """
    Return the active `GeneticBotController` for `state` when playing in
    `GAME_MODE_ADVANCED`, loading the top evolved champion from `bot_stable.json`.
    """
    cached = getattr(state, "_advanced_bot_controller", None)
    if cached is not None and getattr(cached, "bot_id", None) == state.active_bot_id:
        return cached

    import bots

    stable = bots.BotStable.load()
    chosen_genome = None
    if state.active_bot_id and state.active_bot_id != "legacy_classic":
        for g in stable.bots:
            if g.bot_id == state.active_bot_id:
                chosen_genome = g
                break
    if chosen_genome is None:
        chosen_genome = stable.get_top_bot(0)

    state.active_bot_id = chosen_genome.bot_id
    state.active_bot_name = chosen_genome.name
    state.active_bot_ply = chosen_genome.lookahead.lookahead_ply
    ctrl = bots.GeneticBotController(chosen_genome)
    setattr(state, "_advanced_bot_controller", ctrl)
    return ctrl


def purchase_enemy_wave(
    round_number: int,
    budget: int,
    max_units: int = MAX_ARMY_SIZE,
    rng: Optional[random.Random] = None,
) -> List[CreatureSpec]:
    """
    Spend `budget` points to buy up to `max_units` Mammal enemy specs from the
    subset of Mammals unlocked on or before `round_number` (`get_unlocked_mammals`):
      - Round 1: Rat (1 pt)
      - Round 2: +Wolf (2 pts)
      - Round 3: +Bear (4 pts)
      - Round 4: +Lion (5 pts)
      - Round 5: +Griffon (7 pts)
      - ... up to Round 13: +Rajang (22 pts)
    """
    if rng is None:
        rng = default_rng()

    remaining = max(1, budget)
    purchased: List[CreatureSpec] = []
    unlocked_mammals = get_unlocked_mammals(round_number)

    while remaining > 0 and len(purchased) < max_units:
        slots_left = max_units - len(purchased)
        # Minimum cost needed per remaining slot so we don't run out of slots with huge unspent budget
        min_desired_cost = max(1, remaining // slots_left)

        affordable = [
            spec
            for spec in unlocked_mammals
            if spec.point_cost <= remaining
        ]
        if not affordable:
            break

        # Filter to candidates that help spend the budget within the remaining unit cap
        preferred = [s for s in affordable if s.point_cost >= min_desired_cost]
        pool = preferred if preferred else [max(affordable, key=lambda s: s.point_cost)]

        # Weight slightly toward stronger affordable beasts as rounds progress
        weights = [float(s.point_cost) ** 1.15 for s in pool]
        chosen = rng.choices(pool, weights=weights, k=1)[0]
        purchased.append(chosen)
        remaining -= chosen.point_cost

    return purchased


def spawn_enemy_wave_on_board(
    state: GameState, rng: Optional[random.Random] = None
) -> List[Unit]:
    """
    Remove any previous enemy units, calculate `state.wave_budget`, purchase the
    Mammal wave, and place the enemies in the upper half of the battlefield facing South.
    """
    if rng is None:
        rng = default_rng()

    # Clear existing enemy units
    to_remove = [uid for uid, u in state.units.items() if u.team == TEAM_ENEMY]
    for uid in to_remove:
        del state.units[uid]

    state.wave_budget = enemy_wave_budget(state)
    if state.game_mode == GAME_MODE_ADVANCED:
        ctrl = _get_or_create_advanced_bot_controller(state)
        return ctrl.purchase_and_deploy_army(
            state, TEAM_ENEMY, state.wave_budget, rng=rng
        )

    state.active_bot_id = "legacy_classic"
    state.active_bot_name = "Classic Bot"
    state.active_bot_ply = 1
    wave_specs = purchase_enemy_wave(
        state.round_number,
        state.wave_budget,
        max_units=state.max_army_size,
        rng=rng,
    )

    upper_hexes = [
        c for c in get_all_arena_hexes(state.arena_radius) if is_upper_half_hex(c)
    ]
    # Prefer front/middle northern rows (r = -3 .. -1) nicely distributed, then outer rows
    upper_hexes.sort(key=lambda c: (abs(c[1] + 2), abs(c[0])))
    candidate_pool = upper_hexes[: max(len(wave_specs) + 4, 12)]
    rng.shuffle(candidate_pool)
    chosen_hexes = candidate_pool[: len(wave_specs)]

    spawned: List[Unit] = []
    for idx, (spec, coord) in enumerate(zip(wave_specs, chosen_hexes), start=1):
        uid = f"e_r{state.round_number}_{idx}_{spec.asset_slug}"
        enemy_unit = Unit(
            unit_id=uid,
            spec=spec,
            team=TEAM_ENEMY,
            hex_pos=coord,
            facing=FACING_S,
        )
        state.units[uid] = enemy_unit
        spawned.append(enemy_unit)

    return spawned


def grant_base_reinforcements(
    state: GameState,
    count: int = REINFORCEMENTS_BASE,
    rng: Optional[random.Random] = None,
) -> List[Unit]:
    """
    Add up to `count` (default REINFORCEMENTS_BASE = 4) new random Stage 1 reptiles
    into open slots of the player's army (up to `state.max_army_size = 16`).
    """
    if rng is None:
        rng = default_rng()

    added: List[Unit] = []
    used_slots: Set[int] = {
        u.reserve_slot
        for u in state.get_player_army_units()
        if u.reserve_slot is not None
    }

    for slot in range(state.max_army_size):
        if len(added) >= count:
            break
        if len(used_slots) >= state.max_army_size:
            break
        if slot in used_slots:
            continue

        chosen_class = rng.choice(PLAYER_CLASSES)
        spec = get_creature_spec(chosen_class, 1)
        uid = f"p_{state.next_unit_seq}_{spec.asset_slug}"
        state.next_unit_seq += 1

        new_unit = Unit(
            unit_id=uid,
            spec=spec,
            team=TEAM_PLAYER,
            hex_pos=None,
            facing=FACING_N,
            reserve_slot=slot,
        )
        state.units[uid] = new_unit
        used_slots.add(slot)
        added.append(new_unit)

    return added


def create_initial_game_state(
    game_mode: str = GAME_MODE_BASIC,
    rng: Optional[random.Random] = None,
) -> GameState:
    """
    Create a new Round 1 game state in PHASE_SETUP:
      - Player starts with `INITIAL_ARMY_SIZE` (6) random Stage 1 reptiles in the
        left-hand Army Reserve Tray (2x8 grid).
      - Computer opponent spends `WAVE_BASE_POINTS` (4 pts) on 4 Rats placed in the
        upper half of the battlefield facing South.
    """
    if rng is None:
        rng = default_rng()

    state = GameState(
        arena_radius=ARENA_RADIUS,
        phase=PHASE_SETUP,
        game_mode=game_mode,
        round_number=1,
        wave_budget=WAVE_BASE_POINTS,
        active_team=TEAM_PLAYER,
        turn_number=1,
        max_army_size=MAX_ARMY_SIZE,
        reinforcements_per_round=REINFORCEMENTS_BASE,
    )

    recruits = grant_base_reinforcements(state, count=INITIAL_ARMY_SIZE, rng=rng)
    spawn_enemy_wave_on_board(state, rng=rng)

    if recruits:
        state.selected_unit_id = recruits[0].unit_id

    return state


# ============================================================================
# Shared Rules: Flight Landing Validation, Rotations, Moves & Attack Junctures
# ============================================================================

def is_unit_overlapping(state: GameState, unit_id: str) -> bool:
    """Return True if `unit_id` is currently sharing a hex with another living unit (mid-flight)."""
    unit = state.get_unit(unit_id)
    if unit is None or unit.hex_pos is None:
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
        if u.is_deployed and u.unit_id != moving_unit_id and u.hex_pos is not None
    }
    moving_unit = state.get_unit(moving_unit_id)
    takeoff_hex: Optional[HexCoord] = (
        moving_unit.hex_pos
        if (
            moving_unit is not None
            and moving_unit.hex_pos is not None
            and not is_unit_overlapping(state, moving_unit_id)
        )
        else None
    )

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
                if nxt_hex not in occupied and nxt_hex != takeoff_hex:
                    return True
                if spec.can_fly and steps_left - 1 > 0:
                    state_key = (nxt_hex, new_facing, steps_left - 1)
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append(state_key)

    return False


def _rotation_step_context(
    state: GameState, unit: Unit
) -> Optional[Tuple[int, int, bool, bool]]:
    """
    Determine which Rotate-Move step a rotation by `unit` would apply to.

    Returns `(target_step, ref_facing, advances_to_next_step, is_post_move_rotate)`
    or `None` if rotating is illegal.
    """
    spec = unit.spec
    if spec.max_rotate_per_move <= 0 or spec.num_rotate_moves <= 0:
        return None

    if unit.juncture == JUNCTURE_BEFORE_ROTATE or (
        unit.juncture == JUNCTURE_AFTER_ROTATE and not unit.attacked_at_current_juncture
    ):
        if unit.current_step < spec.num_rotate_moves:
            return (unit.current_step, unit.step_start_facing, False, False)
        return None

    next_step = unit.current_step + 1
    if next_step < spec.num_rotate_moves:
        return (next_step, unit.facing, True, False)

    if spec.can_rotate_after_move:
        if unit.juncture == JUNCTURE_AFTER_MOVE:
            return (unit.current_step, unit.facing, False, True)
        if (
            unit.juncture == JUNCTURE_AFTER_POST_ROTATE
            and not unit.attacked_at_current_juncture
        ):
            return (unit.current_step, unit.step_start_facing, False, True)

    return None


def get_legal_rotations(state: GameState, unit_id: str) -> List[int]:
    """
    Return the list of legal absolute facings (0..5) that `unit_id` can rotate to right now.
    During PHASE_SETUP, any deployed player unit may rotate freely to any of the 6 facings.
    """
    unit = state.get_unit(unit_id)
    if unit is None or not unit.is_deployed or unit.hex_pos is None:
        return []

    if state.phase == PHASE_SETUP:
        if unit.team == TEAM_PLAYER:
            return [0, 1, 2, 3, 4, 5]
        return []

    if unit.team != state.active_team:
        return []

    ctx = _rotation_step_context(state, unit)
    if ctx is None:
        return []

    target_step, ref_facing, _, _ = ctx
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
    Rotate `unit_id` to `new_facing` (0..5) if legal under its CreatureSpec attributes
    (or freely if orienting a deployed player unit during PHASE_SETUP).
    """
    unit = state.get_unit(unit_id)
    if unit is None or not unit.is_deployed:
        return False

    new_facing = new_facing % 6
    if new_facing == unit.facing:
        return False

    if state.phase == PHASE_SETUP:
        if unit.team != TEAM_PLAYER:
            return False
        unit.facing = new_facing
        unit.step_start_facing = new_facing
        return True

    if unit.team != state.active_team:
        return False

    legal_facings = get_legal_rotations(state, unit_id)
    if new_facing not in legal_facings:
        return False

    ctx = _rotation_step_context(state, unit)
    if ctx is None:
        return False

    target_step, ref_facing, advances_to_next_step, is_post_move_rotate = ctx
    if advances_to_next_step:
        unit.current_step = target_step
        unit.step_start_facing = ref_facing

    if is_post_move_rotate:
        if unit.juncture == JUNCTURE_AFTER_MOVE:
            unit.step_start_facing = ref_facing
        unit.facing = new_facing
        if unit.facing == unit.step_start_facing and not unit.attacked_at_current_juncture:
            unit.juncture = JUNCTURE_AFTER_MOVE
        else:
            if unit.juncture != JUNCTURE_AFTER_POST_ROTATE:
                unit.juncture = JUNCTURE_AFTER_POST_ROTATE
                unit.attacked_at_current_juncture = False
        return True

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


def _get_legal_moves_for_step_and_facing(
    state: GameState, unit: Unit, target_step: int, cand_facing: int
) -> List[HexCoord]:
    """
    Shared rules helper: return the legal destination hexes `unit` can move into on
    `target_step` when oriented at `cand_facing`.
    If a destination hex is occupied by another unit, moving onto it is legal IF AND ONLY IF
    `unit.spec.can_fly` is True, `steps_left_after >= 1`, and `can_reach_empty_landing_hex`
    confirms the flyer can still land on an empty hex before its moves run out.
    """
    if unit.hex_pos is None or target_step >= unit.spec.num_rotate_moves:
        return []

    spec = unit.spec
    steps_left_after = (spec.num_rotate_moves - 1) - target_step
    legal_hexes: List[HexCoord] = []

    for rel_dir in sorted(spec.legal_move_dirs):
        abs_dir = absolute_direction(cand_facing, rel_dir)
        dest = hex_neighbor(unit.hex_pos, abs_dir)
        if not is_valid_hex(dest[0], dest[1], state.arena_radius):
            continue

        occupant = state.get_unit_at_hex(dest, exclude_unit_id=unit.unit_id)
        if occupant is None:
            legal_hexes.append(dest)
        elif spec.can_fly and steps_left_after >= 1:
            if can_reach_empty_landing_hex(
                state, dest, cand_facing, steps_left_after, spec, unit.unit_id
            ):
                legal_hexes.append(dest)

    return legal_hexes


def get_legal_moves(state: GameState, unit_id: str) -> List[HexCoord]:
    """
    Return all legal destination hex coordinates that `unit_id` can move into right now,
    relative to its instantaneous facing (only active during PHASE_BATTLE).
    """
    if state.phase != PHASE_BATTLE:
        return []
    unit = state.get_unit(unit_id)
    if unit is None or not unit.is_deployed or unit.hex_pos is None or unit.team != state.active_team:
        return []

    target_step = _move_step_index(unit)
    if target_step is None:
        return []

    return _get_legal_moves_for_step_and_facing(state, unit, target_step, unit.facing)


def move_unit(state: GameState, unit_id: str, target_hex: HexCoord) -> bool:
    """
    Move `unit_id` into `target_hex` if legal under its CreatureSpec attributes.
    Returns True if the move was executed.
    """
    if state.phase != PHASE_BATTLE:
        return False
    unit = state.get_unit(unit_id)
    if unit is None or not unit.is_deployed or unit.team != state.active_team:
        return False

    legal_moves = get_legal_moves(state, unit_id)
    if target_hex not in legal_moves:
        return False

    target_step = _move_step_index(unit)
    if target_step is None:
        return False

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
    if state.phase != PHASE_BATTLE:
        return False
    unit = state.get_unit(unit_id)
    if unit is None or not unit.is_deployed or unit.team != state.active_team:
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
    if unit is None or not unit.is_deployed or unit.hex_pos is None:
        return []
    hexes: List[HexCoord] = []
    for rel_dir in sorted(unit.spec.attack_arcs):
        abs_dir = absolute_direction(unit.facing, rel_dir)
        target = hex_neighbor(unit.hex_pos, abs_dir)
        if is_valid_hex(target[0], target[1], state.arena_radius):
            hexes.append(target)
    return hexes


def get_legal_attack_targets(state: GameState, attacker_id: str) -> List[HexCoord]:
    """
    Return all adjacent hex coordinates in `attacker_id`'s `attack_arcs` that contain a living
    opposing unit and can be attacked right now at the current juncture.
    """
    if not can_unit_attack_now(state, attacker_id):
        return []
    attacker = state.get_unit(attacker_id)
    if attacker is None or attacker.hex_pos is None:
        return []

    targets: List[HexCoord] = []
    for arc_hex in get_attack_arc_hexes(state, attacker_id):
        occupant = state.get_unit_at_hex(arc_hex, exclude_unit_id=attacker_id)
        if occupant is not None and occupant.team != attacker.team:
            targets.append(arc_hex)
    return targets


def can_execute_attack(state: GameState, attacker_id: str) -> bool:
    """Return True if `attacker_id` can attack at the current juncture AND has at least 1 enemy in arc."""
    return len(get_legal_attack_targets(state, attacker_id)) > 0


# Special power sets for combat resolution
_ARMOR_PIERCE_POWERS: Set[str] = {
    "serrated_bite",
    "rending_fangs",
    "crushing_bite",
    "snapping_jaws",
    "maul",
}
_CHARGE_POWERS: Set[str] = {
    "winged_javelin",
    "pounce",
    "diving_talons",
    "predator_sprint",
    "gore_charge",
}
_CLEAVE_POWERS: Set[str] = {
    "draconic_sweep",
    "multi_headed_strike",
    "fire_breath",
    "steam_breath",
    "tempest_fury",
    "hellfire_breath",
    "twin_heads",
}


def compute_attack_damage(
    state: GameState, attacker: Unit, defender: Unit
) -> Tuple[int, int, bool, List[str]]:
    """
    Calculate directional combat damage when `attacker` strikes `defender`.
    Returns `(damage, incoming_rel_dir_on_defender, is_vulnerable, modifier_notes)`.
    """
    assert attacker.hex_pos is not None and defender.hex_pos is not None
    incoming_abs = hex_direction_between(defender.hex_pos, attacker.hex_pos)
    incoming_rel = (
        relative_direction(defender.facing, incoming_abs)
        if incoming_abs is not None
        else FORWARD
    )
    is_vulnerable = incoming_rel in defender.spec.vulnerable_arcs

    notes: List[str] = []
    atk_powers = set(attacker.spec.special_powers)

    eff_def = defender.spec.defense
    if atk_powers & _ARMOR_PIERCE_POWERS:
        eff_def = max(0, eff_def - 2)
        notes.append("Armor Pierce")

    dmg = max(1, attacker.spec.attack - eff_def)

    if is_vulnerable:
        if incoming_rel == REAR:
            dmg += 3
            notes.append("Rear Vulnerable (+3)")
        else:
            dmg += 2
            notes.append("Flank Vulnerable (+2)")

    if (atk_powers & _CHARGE_POWERS) and (
        attacker.juncture == JUNCTURE_AFTER_MOVE or attacker.current_step > 0
    ):
        dmg += 2
        notes.append("Charge (+2)")

    if "venom_fangs" in atk_powers or "lethal_venom" in atk_powers:
        dmg += 1
        notes.append("Venom (+1)")

    if "constrict" in atk_powers or "petrifying_gaze" in atk_powers:
        dmg += 2
        notes.append("Special (+2)")

    if "pack_tactics" in atk_powers or "alpha_howl" in atk_powers:
        # Bonus if another allied unit is adjacent to defender
        for d in range(6):
            adj = hex_neighbor(defender.hex_pos, d)
            ally = state.get_unit_at_hex(adj, exclude_unit_id=attacker.unit_id)
            if ally is not None and ally.team == attacker.team:
                dmg += 2
                notes.append("Pack Tactics (+2)")
                break

    return (dmg, incoming_rel, is_vulnerable, notes)


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


def execute_attack(
    state: GameState, attacker_id: str, target_hex: Optional[HexCoord] = None
) -> List[AttackResult]:
    """
    Execute an attack by `attacker_id` at its current juncture (`BEFORE_ROTATE`,
    `AFTER_ROTATE`, or `AFTER_MOVE`) against `target_hex` (or the first legal target
    in its `attack_arcs` if `target_hex` is None).
    Creatures with multi-arc cleave/breath powers strike all enemies currently in their `attack_arcs`.
    """
    legal_targets = get_legal_attack_targets(state, attacker_id)
    if not legal_targets:
        return []
    if target_hex is not None and target_hex not in legal_targets:
        return []

    attacker = state.get_unit(attacker_id)
    if attacker is None or attacker.hex_pos is None:
        return []

    atk_powers = set(attacker.spec.special_powers)
    if atk_powers & _CLEAVE_POWERS:
        # Strike all legal targets in attack_arcs, putting the clicked target_hex first
        if target_hex is not None:
            struck_hexes = [target_hex] + [h for h in legal_targets if h != target_hex]
        else:
            struck_hexes = list(legal_targets)
    else:
        struck_hexes = [target_hex if target_hex is not None else legal_targets[0]]

    if not record_juncture_attack(state, attacker_id):
        return []

    results: List[AttackResult] = []
    for t_hex in struck_hexes:
        defender = state.get_unit_at_hex(t_hex, exclude_unit_id=attacker_id)
        if defender is None or defender.team == attacker.team:
            continue

        dmg, incoming_rel, is_vuln, notes = compute_attack_damage(
            state, attacker, defender
        )
        defender.hp = max(0, defender.hp - dmg)

        if "lethal_venom" in atk_powers:
            defender.poison_stacks += 2
        elif "venom_fangs" in atk_powers:
            defender.poison_stacks += 1

        defender_slain = defender.hp <= 0
        if defender_slain:
            defender.hex_pos = None

        ret_dmg = 0
        def_powers = set(defender.spec.special_powers)
        if "molten_carapace" in def_powers:
            ret_dmg = 2
        elif "thorny_hide" in def_powers or "spiked_ridge" in def_powers:
            ret_dmg = 1

        if ret_dmg > 0 and attacker.is_alive:
            attacker.hp = max(0, attacker.hp - ret_dmg)
            notes.append(f"Thorns (-{ret_dmg} HP)")
            if attacker.hp <= 0:
                attacker.hex_pos = None

        results.append(
            AttackResult(
                attacker_id=attacker.unit_id,
                attacker_name=attacker.spec.name,
                defender_id=defender.unit_id,
                defender_name=defender.spec.name,
                target_hex=t_hex,
                damage=dmg,
                incoming_rel_dir=incoming_rel,
                is_vulnerable=is_vuln,
                defender_slain=defender_slain,
                retaliation_damage=ret_dmg,
                notes=notes,
            )
        )

    state.last_attack_results = results
    if are_all_enemies_slain(state):
        complete_battle_to_evolution(state)
    elif are_all_players_slain(state):
        state.phase = PHASE_DEFEAT
    return results


def are_all_enemies_slain(state: GameState) -> bool:
    """Return True if no living enemy Mammal units remain in the battle."""
    return not any(u.is_alive and u.team == TEAM_ENEMY for u in state.units.values())


def are_all_players_slain(state: GameState) -> bool:
    """Return True if no living human player Reptile units remain in the battle."""
    return not any(u.is_alive and u.team == TEAM_PLAYER for u in state.units.values())


def can_end_turn(state: GameState) -> bool:
    """Return True if in PHASE_BATTLE and no unit is currently mid-flight overlapping another unit."""
    if state.phase != PHASE_BATTLE:
        return False
    for unit in state.units.values():
        if unit.is_deployed and is_unit_overlapping(state, unit.unit_id):
            return False
    return True


def end_turn(state: GameState) -> bool:
    """
    Advance the combat turn and reset turn counters for all living units.
    When `state.enemy_ai_enabled` is True, alternates `active_team` between
    `TEAM_PLAYER` and `TEAM_ENEMY` (incrementing `turn_number` at the start of
    each `TEAM_PLAYER` turn).
    Returns False if a flying unit is still overlapping another unit's hex.
    """
    if not can_end_turn(state):
        return False

    if state.enemy_ai_enabled:
        if state.active_team == TEAM_PLAYER:
            state.active_team = TEAM_ENEMY
        else:
            state.active_team = TEAM_PLAYER
            state.turn_number += 1
    else:
        state.active_team = TEAM_PLAYER
        state.turn_number += 1

    state.ai_acted_unit_ids.clear()

    for unit in list(state.units.values()):
        if not unit.is_alive:
            continue
        if unit.is_deployed and unit.team == state.active_team:
            if "regeneration" in unit.spec.special_powers and unit.hp < unit.spec.max_hp:
                unit.hp = min(unit.spec.max_hp, unit.hp + 2)
            if unit.poison_stacks > 0:
                unit.hp = max(0, unit.hp - unit.poison_stacks)
                if unit.hp <= 0:
                    unit.hex_pos = None
                    continue
        unit.reset_for_turn()

    if are_all_enemies_slain(state):
        complete_battle_to_evolution(state)
    elif are_all_players_slain(state):
        state.phase = PHASE_DEFEAT
    elif state.active_team == TEAM_PLAYER:
        sel = state.get_unit(state.selected_unit_id)
        if sel is None or not sel.is_deployed or sel.team != TEAM_PLAYER:
            deployed_players = [
                u for u in state.get_player_army_units() if u.is_deployed
            ]
            if deployed_players:
                state.selected_unit_id = deployed_players[0].unit_id

    return True


# ============================================================================
# Shared-Rules Enemy Mammal AI (Selection Only; Execution via Shared Helpers)
# ============================================================================

def _best_ai_attack_right_now(
    state: GameState, unit: Unit
) -> Optional[Tuple[HexCoord, float]]:
    """
    Inspect `get_legal_attack_targets(state, unit.unit_id)` and return `(best_target_hex, score)`
    using `compute_attack_damage`, or `None` if no legal target is in arc right now.
    """
    legal_targets = get_legal_attack_targets(state, unit.unit_id)
    if not legal_targets:
        return None

    best_hex: Optional[HexCoord] = None
    best_score = -1e9
    for t_hex in legal_targets:
        defender = state.get_unit_at_hex(t_hex, exclude_unit_id=unit.unit_id)
        if defender is None:
            continue
        dmg, _, is_vuln, _ = compute_attack_damage(state, unit, defender)
        is_kill = dmg >= defender.hp
        score = (
            dmg * 30.0
            + (400.0 if is_kill else 0.0)
            + (45.0 if is_vuln else 0.0)
            + defender.spec.point_cost * 5.0
        )
        if score > best_score:
            best_score = score
            best_hex = t_hex

    if best_hex is None:
        return None
    return (best_hex, best_score)


def _potential_attack_score_at(
    state: GameState, unit: Unit, cand_hex: HexCoord, cand_facing: int
) -> float:
    """
    Return the best attack score `unit` could achieve if standing on empty `cand_hex`
    with orientation `cand_facing`.
    """
    if state.get_unit_at_hex(cand_hex, exclude_unit_id=unit.unit_id) is not None:
        return 0.0

    saved_hex = unit.hex_pos
    saved_facing = unit.facing
    best_score = 0.0
    try:
        unit.hex_pos = cand_hex
        unit.facing = cand_facing
        for rel_dir in unit.spec.attack_arcs:
            abs_dir = absolute_direction(cand_facing, rel_dir)
            t_hex = hex_neighbor(cand_hex, abs_dir)
            if not is_valid_hex(t_hex[0], t_hex[1], state.arena_radius):
                continue
            defender = state.get_unit_at_hex(t_hex, exclude_unit_id=unit.unit_id)
            if defender is None or defender.team == unit.team:
                continue
            dmg, _, is_vuln, _ = compute_attack_damage(state, unit, defender)
            is_kill = dmg >= defender.hp
            score = (
                dmg * 30.0
                + (400.0 if is_kill else 0.0)
                + (45.0 if is_vuln else 0.0)
                + defender.spec.point_cost * 5.0
            )
            if score > best_score:
                best_score = score
    finally:
        unit.hex_pos = saved_hex
        unit.facing = saved_facing

    return best_score


def _score_ai_state(
    state: GameState,
    unit: Unit,
    cand_hex: HexCoord,
    cand_facing: int,
    can_attack_after: bool,
) -> float:
    """
    Evaluate the tactical desirability of `(cand_hex, cand_facing)` for a Classic AI unit.
    Rewards closing distance to opposing units, setting up high-damage/flank/rear attacks,
    and orienting `FORWARD` toward nearby enemies to protect vulnerable flanks/rear.
    """
    opp_units = [
        u
        for u in state.units.values()
        if u.is_deployed and u.team != unit.team and u.hex_pos is not None
    ]
    if not opp_units:
        return 0.0

    # Nearest living opposing unit (tie-break by lower HP, then higher point_cost)
    nearest_opp = min(
        opp_units,
        key=lambda ou: (
            hex_distance(cand_hex, ou.hex_pos),  # type: ignore[arg-type]
            ou.hp,
            -ou.spec.point_cost,
        ),
    )
    assert nearest_opp.hex_pos is not None
    min_dist = hex_distance(cand_hex, nearest_opp.hex_pos)

    # Prefer melee adjacency (min_dist == 1)
    score = -abs(min_dist - 1) * 38.0 - min_dist * 4.0

    # If at distance > 1, reward stepping onto a hex that has an unblocked closer neighbor
    # so units behind an allied blocker step around to an open approach lane
    if min_dist > 1:
        has_open_step = any(
            is_valid_hex(hex_neighbor(cand_hex, d)[0], hex_neighbor(cand_hex, d)[1], state.arena_radius)
            and hex_distance(hex_neighbor(cand_hex, d), nearest_opp.hex_pos) < min_dist
            and state.get_unit_at_hex(hex_neighbor(cand_hex, d), exclude_unit_id=unit.unit_id) is None
            for d in range(6)
        )
        if has_open_step:
            score += 15.0

    # Slight tie-break preference for empty hexes over occupied fly-over hexes at equal distance,
    # while still allowing flyers to cross occupied blockers to reduce distance to opposing units.
    if state.get_unit_at_hex(cand_hex, exclude_unit_id=unit.unit_id) is not None:
        score -= 6.0

    # Reward immediate attack opportunity from `(cand_hex, cand_facing)`
    if can_attack_after:
        atk_score = _potential_attack_score_at(state, unit, cand_hex, cand_facing)
        if atk_score > 0.0:
            score += 500.0 + atk_score

    # Orient facing toward the nearest opposing unit
    best_dir_to_nearest = min(
        range(6),
        key=lambda d: hex_distance(
            hex_neighbor(cand_hex, d), nearest_opp.hex_pos  # type: ignore[arg-type]
        ),
    )
    score -= hex_angle_distance(cand_facing, best_dir_to_nearest) * 10.0

    # Flanking & defensive orientation around adjacent opposing units
    for ou in opp_units:
        if ou.hex_pos is None:
            continue
        if hex_distance(cand_hex, ou.hex_pos) == 1:
            # Incoming direction on `ou` when we attack `ou` (from `ou.hex_pos` to `cand_hex`)
            inc_on_opp = hex_direction_between(ou.hex_pos, cand_hex)
            if inc_on_opp is not None:
                rel_on_opp = relative_direction(ou.facing, inc_on_opp)
                if rel_on_opp in ou.spec.vulnerable_arcs:
                    score += 14.0
            # Incoming direction on us when `ou` attacks us (from `cand_hex` to `ou.hex_pos`)
            inc_on_us = hex_direction_between(cand_hex, ou.hex_pos)
            if inc_on_us is not None:
                rel_on_us = relative_direction(cand_facing, inc_on_us)
                if rel_on_us in unit.spec.vulnerable_arcs:
                    score -= 10.0

    return score


def execute_next_ai_action(
    state: GameState, allow_any_team: bool = False
) -> Optional[AIActionRecord]:
    """
    Execute a single atomic Classic AI action (`rotate_unit`, `move_unit`,
    `execute_attack`, or `end_turn`) using ONLY shared `rules.py` helpers.

    When `allow_any_team` is False (default), only acts when `state.active_team == TEAM_ENEMY`.
    When `allow_any_team` is True, acts for `state.active_team` (`TEAM_ENEMY` or `TEAM_PLAYER`),
    allowing the Classic AI to play either side in headless arena benchmarks.
    """
    if state.phase != PHASE_BATTLE:
        return None
    if not allow_any_team and state.active_team != TEAM_ENEMY:
        return None

    # In Play Advanced mode, the enemy Mammal wave is commanded by the top evolved
    # bot from `bot_stable.json` using X-ply lookahead and evolved tactical genes.
    if (
        not allow_any_team
        and state.game_mode == GAME_MODE_ADVANCED
        and state.active_team == TEAM_ENEMY
    ):
        ctrl = _get_or_create_advanced_bot_controller(state)
        return ctrl.execute_next_action(state)

    active_team = state.active_team
    opp_units = [
        u
        for u in state.units.values()
        if u.is_deployed and u.team != active_team and u.hex_pos is not None
    ]
    my_units = [
        u
        for u in state.units.values()
        if u.is_deployed
        and u.team == active_team
        and u.hex_pos is not None
        and u.unit_id not in state.ai_acted_unit_ids
    ]

    # Order active team's units front-to-back (closest to opposing army first)
    if opp_units:
        my_units.sort(
            key=lambda mu: (
                min(
                    hex_distance(mu.hex_pos, ou.hex_pos)  # type: ignore[arg-type]
                    for ou in opp_units
                ),
                mu.unit_id,
            )
        )
    else:
        my_units.sort(key=lambda mu: mu.unit_id)

    for unit in my_units:
        assert unit.hex_pos is not None
        state.selected_unit_id = unit.unit_id
        attacks_left = unit.spec.total_attacks_per_move - unit.attacks_used

        # 1. Check if `unit` can attack right now at its current juncture
        now_atk = _best_ai_attack_right_now(state, unit)
        if now_atk is not None:
            target_hex, now_atk_score = now_atk
            should_attack_now = False
            if attacks_left >= 2 or unit.moves_remaining == 0:
                should_attack_now = True
            else:
                # Check if rotating/moving on this step can yield a strictly better attack
                step_idx = _move_step_index(unit)
                best_alt_atk = 0.0
                if step_idx is not None:
                    cand_rots = (
                        [unit.facing]
                        if unit.juncture == JUNCTURE_AFTER_ROTATE
                        else [unit.facing]
                        + [
                            f
                            for f in get_legal_rotations(state, unit.unit_id)
                            if f != unit.facing
                        ]
                    )
                    for cf in cand_rots:
                        best_alt_atk = max(
                            best_alt_atk,
                            _potential_attack_score_at(state, unit, unit.hex_pos, cf),
                        )
                        for dest in _get_legal_moves_for_step_and_facing(
                            state, unit, step_idx, cf
                        ):
                            best_alt_atk = max(
                                best_alt_atk,
                                _potential_attack_score_at(state, unit, dest, cf),
                            )
                if now_atk_score >= best_alt_atk - 5.0:
                    should_attack_now = True

            if should_attack_now:
                results = execute_attack(state, unit.unit_id, target_hex)
                if results:
                    first = results[0]
                    desc = (
                        f"Enemy {unit.spec.name} attacked {first.defender_name} "
                        f"for {first.damage} dmg"
                        + (" (SLAIN!)" if first.defender_slain else "")
                    )
                    return AIActionRecord(
                        action_type="attack",
                        unit_id=unit.unit_id,
                        unit_name=unit.spec.name,
                        target_hex=target_hex,
                        attack_results=results,
                        description=desc,
                    )

        # 2. If at JUNCTURE_AFTER_ROTATE, `unit` has already rotated for `current_step`,
        #    so only consider moving on `current_step` (never re-rotating in the same step).
        overlapping = is_unit_overlapping(state, unit.unit_id)
        if unit.juncture == JUNCTURE_AFTER_ROTATE:
            legal_mvs = get_legal_moves(state, unit.unit_id)
            can_atk_after_move = (unit.attacks_used < unit.spec.total_attacks_per_move)
            best_dest: Optional[HexCoord] = None
            best_score = (
                -1e9
                if overlapping
                else _score_ai_state(
                    state, unit, unit.hex_pos, unit.facing, can_atk_after_move
                )
            )
            for dest in legal_mvs:
                sc = _score_ai_state(state, unit, dest, unit.facing, can_atk_after_move)
                if sc > best_score + 0.5:
                    best_score = sc
                    best_dest = dest

            if best_dest is not None and move_unit(state, unit.unit_id, best_dest):
                return AIActionRecord(
                    action_type="move",
                    unit_id=unit.unit_id,
                    unit_name=unit.spec.name,
                    target_hex=best_dest,
                    description=f"Enemy {unit.spec.name} moved to {best_dest}.",
                )

            # Finished with current_step (skipped moving after rotating)
            state.ai_acted_unit_ids.add(unit.unit_id)
            continue

        # 3. If at JUNCTURE_AFTER_POST_ROTATE, no moves or rotations remain
        if unit.juncture == JUNCTURE_AFTER_POST_ROTATE:
            state.ai_acted_unit_ids.add(unit.unit_id)
            continue

        # 4. At JUNCTURE_BEFORE_ROTATE or JUNCTURE_AFTER_MOVE:
        #    Evaluate `(cand_facing, cand_dest_hex)` for the upcoming Rotate-Move step
        step_idx = _move_step_index(unit)
        if step_idx is None:
            state.ai_acted_unit_ids.add(unit.unit_id)
            continue

        legal_rots = get_legal_rotations(state, unit.unit_id)
        cand_facings = [unit.facing] + [f for f in legal_rots if f != unit.facing]
        can_atk_after = unit.attacks_used < unit.spec.total_attacks_per_move

        best_cf = unit.facing
        best_dest = unit.hex_pos
        best_score = (
            -1e9
            if overlapping
            else _score_ai_state(
                state, unit, unit.hex_pos, unit.facing, can_atk_after
            )
        )

        for cf in cand_facings:
            if not overlapping:
                stay_sc = _score_ai_state(state, unit, unit.hex_pos, cf, can_atk_after)
                if stay_sc > best_score + 0.5:
                    best_score = stay_sc
                    best_cf = cf
                    best_dest = unit.hex_pos

            for dest in _get_legal_moves_for_step_and_facing(state, unit, step_idx, cf):
                mv_sc = _score_ai_state(state, unit, dest, cf, can_atk_after)
                if mv_sc > best_score + 0.5:
                    best_score = mv_sc
                    best_cf = cf
                    best_dest = dest

        if best_cf != unit.facing and rotate_unit(state, unit.unit_id, best_cf):
            return AIActionRecord(
                action_type="rotate",
                unit_id=unit.unit_id,
                unit_name=unit.spec.name,
                new_facing=best_cf,
                description=f"Enemy {unit.spec.name} rotated to {ABS_FACING_NAMES[best_cf]}.",
            )

        if best_dest != unit.hex_pos and move_unit(state, unit.unit_id, best_dest):
            return AIActionRecord(
                action_type="move",
                unit_id=unit.unit_id,
                unit_name=unit.spec.name,
                target_hex=best_dest,
                description=f"Enemy {unit.spec.name} moved to {best_dest}.",
            )

        state.ai_acted_unit_ids.add(unit.unit_id)

    # All living enemy units have completed their actions -> end the enemy turn
    end_turn(state)
    return AIActionRecord(
        action_type="end_turn",
        unit_id=None,
        description=f"Enemy turn complete — Turn {state.turn_number} (YOUR TURN).",
    )


def run_enemy_turn(
    state: GameState, max_actions: int = 250, allow_any_team: bool = False
) -> List[AIActionRecord]:
    """
    Execute the entire Classic AI turn synchronously until control passes to
    the opposing team or the battle concludes (`PHASE_EVOLUTION` or `PHASE_DEFEAT`).
    """
    starting_team = state.active_team
    if not allow_any_team and starting_team != TEAM_ENEMY:
        return []
    records: List[AIActionRecord] = []
    for _ in range(max_actions):
        if state.phase != PHASE_BATTLE or state.active_team != starting_team:
            break
        rec = execute_next_ai_action(state, allow_any_team=allow_any_team)
        if rec is None:
            break
        records.append(rec)
        if rec.action_type == "end_turn":
            break
    return records


def cycle_unit_stage(state: GameState, unit_id: str, delta: int) -> bool:
    """
    Debug helper: cycle `unit_id`'s evolution stage (1..8 for player classes, 1..13 for mammals)
    and reset its turn state so the new stage's capabilities can be tested immediately.
    Works for units both in reserve and on the battlefield.
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


def create_splash_game_state() -> GameState:
    """Create the initial Pre-Game Splash & Title Screen state (`PHASE_SPLASH`)."""
    return GameState(
        arena_radius=ARENA_RADIUS,
        phase=PHASE_SPLASH,
        game_mode=GAME_MODE_BASIC,
        round_number=1,
        wave_budget=WAVE_BASE_POINTS,
        active_team=TEAM_PLAYER,
        turn_number=1,
        max_army_size=MAX_ARMY_SIZE,
        reinforcements_per_round=REINFORCEMENTS_BASE,
    )


def open_instructions(state: GameState) -> bool:
    """Transition from `PHASE_SPLASH` to `PHASE_INSTRUCTIONS` (`View Instructions`)."""
    if state.phase != PHASE_SPLASH:
        return False
    state.phase = PHASE_INSTRUCTIONS
    return True


def close_instructions(state: GameState) -> bool:
    """Return from `PHASE_INSTRUCTIONS` to `PHASE_SPLASH`."""
    if state.phase != PHASE_INSTRUCTIONS:
        return False
    state.phase = PHASE_SPLASH
    return True


def return_to_splash(state: GameState) -> bool:
    """Return any active game state to the Pre-Game Splash Screen (`PHASE_SPLASH`)."""
    state.units.clear()
    state.evolution_records.clear()
    state.last_attack_results.clear()
    state.ai_acted_unit_ids.clear()
    state.selected_unit_id = None
    state.round_number = 1
    state.turn_number = 1
    state.active_team = TEAM_PLAYER
    state.phase = PHASE_SPLASH
    return True


def start_campaign(
    state: GameState,
    game_mode: str = GAME_MODE_BASIC,
    rng: Optional[random.Random] = None,
) -> bool:
    """
    Start a new Round 1 campaign in `PHASE_SETUP` with the chosen `game_mode`:
      - `GAME_MODE_BASIC`        : Classic Bot + adaptive wave budget
      - `GAME_MODE_INTERMEDIATE` : Classic Bot + fixed mathematical wave schedule
      - `GAME_MODE_ADVANCED`     : Evolved Bot (`bot_stable.json`) + fixed wave schedule
    Mutates `state` in place so callers holding a reference to `state` see the new campaign.
    """
    if game_mode not in (
        GAME_MODE_BASIC,
        GAME_MODE_INTERMEDIATE,
        GAME_MODE_ADVANCED,
    ):
        return False

    fresh = create_initial_game_state(game_mode=game_mode, rng=rng)
    state.arena_radius = fresh.arena_radius
    state.phase = fresh.phase
    state.game_mode = fresh.game_mode
    state.round_number = fresh.round_number
    state.wave_budget = fresh.wave_budget
    state.active_team = fresh.active_team
    state.turn_number = fresh.turn_number
    state.max_army_size = fresh.max_army_size
    state.reinforcements_per_round = fresh.reinforcements_per_round
    state.next_unit_seq = fresh.next_unit_seq
    state.units = fresh.units
    state.selected_unit_id = fresh.selected_unit_id
    state.evolution_records = fresh.evolution_records
    state.last_attack_results = fresh.last_attack_results
    state.enemy_ai_enabled = fresh.enemy_ai_enabled
    state.ai_acted_unit_ids = fresh.ai_acted_unit_ids
    state.active_bot_id = fresh.active_bot_id
    state.active_bot_name = fresh.active_bot_name
    state.active_bot_ply = fresh.active_bot_ply
    setattr(
        state,
        "_advanced_bot_controller",
        getattr(fresh, "_advanced_bot_controller", None),
    )
    return True


def create_demo_battle() -> GameState:
    """Create the default initial Pre-Game Splash Screen state (`PHASE_SPLASH`)."""
    return create_splash_game_state()
