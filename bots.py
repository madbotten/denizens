"""
Genetic AI Bot Controllers, Genome Definitions, X-Ply Lookahead Search,
and Decision Telemetry for Denizens.

All bot setup placements, rotations, movements, juncture attacks, and turn transitions
execute strictly through shared functions in `rules.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from constants import (
    ABS_FACING_NAMES,
    CLASS_LIZARD,
    CLASS_MAMMAL,
    CLASS_SERPENT,
    CLASS_TURTLE,
    FACING_N,
    FACING_NE,
    FACING_NW,
    FACING_S,
    FACING_SE,
    FACING_SW,
    FORWARD,
    JUNCTURE_AFTER_MOVE,
    JUNCTURE_AFTER_POST_ROTATE,
    JUNCTURE_AFTER_ROTATE,
    JUNCTURE_BEFORE_ROTATE,
    MAX_ARMY_SIZE,
    PHASE_BATTLE,
    PHASE_DEFEAT,
    PHASE_EVOLUTION,
    PHASE_SETUP,
    PLAYER_CLASSES,
    REAR,
    REL_DIR_NAMES,
    TEAM_ENEMY,
    TEAM_PLAYER,
)
from creatures import (
    CREATURE_BY_ID,
    MAMMAL_LINEUP,
    CreatureSpec,
    get_creature_spec,
    get_unlocked_mammals,
    is_mammal_unlocked,
)
from hexgrid import (
    HexCoord,
    absolute_direction,
    hex_angle_distance,
    hex_direction_between,
    hex_distance,
    hex_neighbor,
    is_valid_hex,
    relative_direction,
)
import rules


# ============================================================================
# Genetic Bot Genome Dataclasses
# ============================================================================

@dataclass
class SetupGenes:
    """
    Genes governing how a bot spends its wave point budget and deploys its army
    onto its half of the 61-hex arena during PHASE_SETUP.
    """
    mammal_purchase_weights: Dict[str, float] = field(default_factory=dict)
    elite_vs_swarm_bias: float = 0.15       # -1.0 = swarm cheap units, +1.0 = save for elites
    frontline_aggression: float = 0.75      # 0.0 = deploy deep back, 1.0 = deploy on center line
    tank_frontline_weight: float = 0.85     # Place high-HP/DEF tanks in front of agile strikers
    spread_vs_cluster: float = -0.20        # -1.0 = tight center pack, +1.0 = wide flank wings
    wing_inward_facing_bias: float = 0.60   # Angle outer flank units inward toward center

    def __post_init__(self) -> None:
        for spec in MAMMAL_LINEUP:
            if spec.creature_id not in self.mammal_purchase_weights:
                self.mammal_purchase_weights[spec.creature_id] = 1.0


@dataclass
class MammalTacticalGenes:
    """
    Species-specific movement, positioning, and attack-timing genes for a single
    Mammal form (`mammal_01_rat` .. `mammal_13_hell_hound`).
    """
    aggression_weight: float = 38.0          # Reward for closing distance to enemies
    flanking_weight: float = 20.0            # Reward for threatening enemy vulnerable_arcs
    self_defense_weight: float = 18.0        # Penalty for exposing own vulnerable_arcs
    pack_cohesion_weight: float = 12.0       # Reward for allied adjacency (pack_tactics/alpha_howl)
    cleave_positioning_weight: float = 24.0  # Reward for multi-target arc alignment
    hit_and_run_weight: float = 16.0         # Reward for stepping out of threat arcs after attacking
    fly_over_readiness: float = 12.0         # Willingness of flyers to cross blockers


@dataclass
class TargetingGenes:
    """
    Genes governing target selection: whether to focus strong enemies first,
    clear weak enemies first, or combine both (confirming kills on weakened
    units while concentrating primary damage on high-threat targets).
    """
    strong_vs_weak_bias: float = 0.20          # -1.0 = weak first, +1.0 = strong first
    hybrid_kill_confirm_weight: float = 1.35   # "Both" strategy: high priority on lethal blows
    vulnerable_arc_bonus_weight: float = 1.15  # Multiplier on flank/rear vulnerability attacks
    thorns_caution_weight: float = 1.10        # Caution against thorny_hide / molten_carapace
    focus_fire_weight: float = 1.25            # Bonus for attacking already-damaged enemies


@dataclass
class LookaheadGenes:
    """
    Genes governing X-ply lookahead search depth and opponent counter-response weighting.
      - lookahead_ply = 1: Evaluates own unit's turn only (ignores enemy counter-turn).
      - lookahead_ply = 2: Completes own team turn + evaluates enemy counter-turn response.
      - lookahead_ply = 3: Own turn -> Enemy response -> Own follow-up (ignores 2nd enemy response).
      - lookahead_ply = 4: Evaluates 2 full action-reaction rounds (own -> enemy -> own -> enemy).
    """
    lookahead_ply: int = 2
    beam_width: int = 3
    enemy_response_weight: float = 0.78
    followup_discount: float = 0.65

    def __post_init__(self) -> None:
        self.lookahead_ply = max(1, min(4, int(self.lookahead_ply)))
        self.beam_width = max(2, min(5, int(self.beam_width)))


@dataclass
class GlobalCoordinationGenes:
    """Army-wide coordination and survival threshold genes."""
    activation_order_bias: float = 0.15       # -1.0 = frontmost first, +1.0 = mobile strikers first
    wounded_retreat_threshold: float = 0.28   # HP fraction below which unit prioritizes safety


@dataclass
class BotGenome:
    """
    Complete genetic specification for an evolved Denizens AI bot controller.
    Fully serializable to/from JSON dictionaries for `bot_stable.json`.
    """
    bot_id: str = "bot_default"
    name: str = "Ironfang Prime"
    generation: int = 0
    fitness: float = 0.0
    elo_rating: float = 1200.0
    wins_vs_classic: int = 0
    games_vs_classic: int = 0
    championship_streak: int = 0
    setup: SetupGenes = field(default_factory=SetupGenes)
    mammal_genes: Dict[str, MammalTacticalGenes] = field(default_factory=dict)
    targeting: TargetingGenes = field(default_factory=TargetingGenes)
    lookahead: LookaheadGenes = field(default_factory=LookaheadGenes)
    coordination: GlobalCoordinationGenes = field(default_factory=GlobalCoordinationGenes)

    def __post_init__(self) -> None:
        for spec in MAMMAL_LINEUP:
            if spec.creature_id not in self.mammal_genes:
                self.mammal_genes[spec.creature_id] = _default_mammal_genes_for_spec(spec)

    def get_mammal_genes(self, creature_id: str) -> MammalTacticalGenes:
        if creature_id in self.mammal_genes:
            return self.mammal_genes[creature_id]
        return MammalTacticalGenes()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BotGenome:
        setup_data = data.get("setup", {})
        mammal_data = data.get("mammal_genes", {})
        targeting_data = data.get("targeting", {})
        lookahead_data = data.get("lookahead", {})
        coord_data = data.get("coordination", {})

        mammal_genes_map: Dict[str, MammalTacticalGenes] = {}
        for cid, mdict in mammal_data.items():
            if isinstance(mdict, dict):
                mammal_genes_map[cid] = MammalTacticalGenes(**mdict)

        return cls(
            bot_id=str(data.get("bot_id", "bot_loaded")),
            name=str(data.get("name", "Loaded Bot")),
            generation=int(data.get("generation", 0)),
            fitness=float(data.get("fitness", 0.0)),
            elo_rating=float(data.get("elo_rating", 1200.0)),
            wins_vs_classic=int(data.get("wins_vs_classic", 0)),
            games_vs_classic=int(data.get("games_vs_classic", 0)),
            championship_streak=int(data.get("championship_streak", 0)),
            setup=SetupGenes(**setup_data) if isinstance(setup_data, dict) else SetupGenes(),
            mammal_genes=mammal_genes_map,
            targeting=(
                TargetingGenes(**targeting_data)
                if isinstance(targeting_data, dict)
                else TargetingGenes()
            ),
            lookahead=(
                LookaheadGenes(**lookahead_data)
                if isinstance(lookahead_data, dict)
                else LookaheadGenes()
            ),
            coordination=(
                GlobalCoordinationGenes(**coord_data)
                if isinstance(coord_data, dict)
                else GlobalCoordinationGenes()
            ),
        )


def _default_mammal_genes_for_spec(spec: CreatureSpec) -> MammalTacticalGenes:
    """Return sensible baseline tactical genes tailored to a Mammal species' traits."""
    powers = set(spec.special_powers)
    is_pack = bool(powers & {"pack_tactics", "alpha_howl"})
    is_cleave = len(spec.attack_arcs) > 1 or bool(powers & {"hellfire_breath", "tentacle_lash"})
    is_tank = spec.max_hp >= 18 and spec.defense >= 4
    return MammalTacticalGenes(
        aggression_weight=38.0 if not is_tank else 34.0,
        flanking_weight=24.0 if spec.num_rotate_moves >= 2 else 14.0,
        self_defense_weight=22.0 if is_tank else 16.0,
        pack_cohesion_weight=26.0 if is_pack else 10.0,
        cleave_positioning_weight=32.0 if is_cleave else 14.0,
        hit_and_run_weight=22.0 if spec.num_rotate_moves >= 3 else 10.0,
        fly_over_readiness=18.0 if spec.can_fly else 0.0,
    )


def create_random_genome(
    rng: Optional[random.Random] = None,
    bot_id: str = "bot_rand",
    name: str = "Wild Strain",
    generation: int = 0,
    allowed_plies: Tuple[int, ...] = (1, 2, 3, 4),
) -> BotGenome:
    """Generate a randomized `BotGenome` for initial population seeding or diversity injection."""
    if rng is None:
        rng = random.Random()

    purchase_weights = {
        spec.creature_id: round(rng.uniform(0.25, 2.75), 3) for spec in MAMMAL_LINEUP
    }
    setup = SetupGenes(
        mammal_purchase_weights=purchase_weights,
        elite_vs_swarm_bias=round(rng.uniform(-0.9, 0.9), 3),
        frontline_aggression=round(rng.uniform(0.2, 1.0), 3),
        tank_frontline_weight=round(rng.uniform(-0.3, 1.8), 3),
        spread_vs_cluster=round(rng.uniform(-0.9, 0.9), 3),
        wing_inward_facing_bias=round(rng.uniform(0.0, 1.0), 3),
    )

    mammal_genes: Dict[str, MammalTacticalGenes] = {}
    for spec in MAMMAL_LINEUP:
        mammal_genes[spec.creature_id] = MammalTacticalGenes(
            aggression_weight=round(rng.uniform(16.0, 68.0), 2),
            flanking_weight=round(rng.uniform(4.0, 48.0), 2),
            self_defense_weight=round(rng.uniform(4.0, 44.0), 2),
            pack_cohesion_weight=round(rng.uniform(0.0, 40.0), 2),
            cleave_positioning_weight=round(rng.uniform(4.0, 52.0), 2),
            hit_and_run_weight=round(rng.uniform(0.0, 42.0), 2),
            fly_over_readiness=round(rng.uniform(-10.0, 35.0), 2),
        )

    targeting = TargetingGenes(
        strong_vs_weak_bias=round(rng.uniform(-1.0, 1.0), 3),
        hybrid_kill_confirm_weight=round(rng.uniform(0.2, 2.4), 3),
        vulnerable_arc_bonus_weight=round(rng.uniform(0.3, 2.2), 3),
        thorns_caution_weight=round(rng.uniform(0.2, 2.2), 3),
        focus_fire_weight=round(rng.uniform(0.2, 2.4), 3),
    )

    lookahead = LookaheadGenes(
        lookahead_ply=rng.choice(allowed_plies),
        beam_width=rng.choice((2, 3, 4)),
        enemy_response_weight=round(rng.uniform(0.65, 1.45), 3),
        followup_discount=round(rng.uniform(0.45, 0.85), 3),
    )

    coordination = GlobalCoordinationGenes(
        activation_order_bias=round(rng.uniform(-1.0, 1.0), 3),
        wounded_retreat_threshold=round(rng.uniform(0.10, 0.45), 3),
    )

    return BotGenome(
        bot_id=bot_id,
        name=name,
        generation=generation,
        setup=setup,
        mammal_genes=mammal_genes,
        targeting=targeting,
        lookahead=lookahead,
        coordination=coordination,
    )


# ============================================================================
# Lookahead Telemetry & Decision Trace Models
# ============================================================================

@dataclass
class AtomicBotStep:
    """A single rule-verified action (`rotate`, `move`, or `attack`) in a unit's planned turn."""
    action_type: str  # "rotate", "move", "attack"
    new_facing: Optional[int] = None
    target_hex: Optional[HexCoord] = None


@dataclass
class CandidatePlanTrace:
    """Diagnostic breakdown for one candidate single-unit turn plan considered during search."""
    plan_summary: str
    end_hex: HexCoord
    end_facing: int
    immediate_attack_score: float
    positional_score: float
    ply1_score: float
    enemy_response_penalty: float = 0.0
    followup_net_score: float = 0.0
    total_score: float = 0.0


@dataclass
class DecisionTrace:
    """
    Full lookahead telemetry recorded when a `GeneticBotController` selects a unit's
    turn plan. Allows auditing why an X-ply bot chose one move over another.
    """
    turn_number: int
    team: str
    unit_id: str
    unit_name: str
    lookahead_ply: int
    chosen_plan_summary: str
    chosen_1ply_score: float
    chosen_enemy_response_penalty: float
    chosen_total_score: float
    top_candidates: List[CandidatePlanTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Classic (Legacy) Bot Controller Wrapper (Benchmark)
# ============================================================================

class LegacyBotController:
    """
    Wraps the original Classic AI in `rules.py` (`purchase_enemy_wave`,
    `spawn_enemy_wave_on_board`, `execute_next_ai_action`, `run_enemy_turn`)
    so it can serve as a fixed benchmark in the headless arena and in Basic/Intermediate modes.
    """

    def __init__(self, name: str = "Classic Legacy Bot") -> None:
        self.bot_id = "legacy_classic"
        self.name = name
        self.lookahead_ply = 1
        self.decision_traces: List[DecisionTrace] = []

    def purchase_and_deploy_army(
        self,
        state: rules.GameState,
        team: str,
        budget: int,
        rng: Optional[random.Random] = None,
    ) -> List[rules.Unit]:
        """Purchase and deploy a Mammal wave for `team` using the Classic algorithm."""
        if rng is None:
            rng = random.Random()

        to_remove = [uid for uid, u in state.units.items() if u.team == team]
        for uid in to_remove:
            del state.units[uid]

        wave_specs = rules.purchase_enemy_wave(
            state.round_number, budget, max_units=state.max_army_size, rng=rng
        )
        valid_hexes = rules.get_valid_team_deploy_hexes(state, team)
        target_r = -2 if team == TEAM_ENEMY else 2
        valid_hexes.sort(key=lambda c: (abs(c[1] - target_r), abs(c[0])))
        candidate_pool = valid_hexes[: max(len(wave_specs) + 4, 12)]
        rng.shuffle(candidate_pool)
        chosen_hexes = candidate_pool[: len(wave_specs)]

        default_facing = FACING_S if team == TEAM_ENEMY else FACING_N
        prefix = "e" if team == TEAM_ENEMY else "p"
        saved_phase = state.phase
        state.phase = PHASE_SETUP
        spawned: List[rules.Unit] = []
        try:
            for idx, (spec, coord) in enumerate(zip(wave_specs, chosen_hexes), start=1):
                uid = f"{prefix}_r{state.round_number}_{idx}_{spec.asset_slug}"
                unit = rules.Unit(
                    unit_id=uid,
                    spec=spec,
                    team=team,
                    hex_pos=None,
                    facing=default_facing,
                )
                state.units[uid] = unit
                rules.deploy_team_unit_to_hex(state, uid, coord, facing=default_facing)
                spawned.append(unit)
        finally:
            state.phase = saved_phase

        return spawned

    def execute_next_action(self, state: rules.GameState) -> Optional[rules.AIActionRecord]:
        return rules.execute_next_ai_action(state, allow_any_team=True)

    def run_turn(
        self, state: rules.GameState, max_actions: int = 250
    ) -> List[rules.AIActionRecord]:
        return rules.run_enemy_turn(state, max_actions=max_actions, allow_any_team=True)


# ============================================================================
# Genetic Bot Controller (Setup Genes + Per-Mammal Genes + Targeting + X-Ply)
# ============================================================================

@dataclass
class _SimulatedPlan:
    """Internal representation of a legal single-unit turn plan discovered via `rules.py`."""
    steps: List[AtomicBotStep]
    final_state: rules.GameState
    end_hex: HexCoord
    end_facing: int
    immediate_attack_score: float
    positional_score: float
    ply1_score: float
    summary: str


class GeneticBotController:
    """
    Executes army purchasing/deployment and X-ply tactical combat decisions
    governed by a `BotGenome`, strictly through shared `rules.py` helpers.
    """

    def __init__(
        self, genome: BotGenome, record_telemetry: bool = False
    ) -> None:
        self.genome = genome
        self.bot_id = genome.bot_id
        self.name = genome.name
        self.record_telemetry = record_telemetry
        self.decision_traces: List[DecisionTrace] = []
        self._pending_unit_id: Optional[str] = None
        self._pending_steps: List[AtomicBotStep] = []

    @property
    def lookahead_ply(self) -> int:
        return self.genome.lookahead.lookahead_ply

    def reset_for_battle(self) -> None:
        """Clear any pending step queue and decision traces at the start of a new battle."""
        self._pending_unit_id = None
        self._pending_steps.clear()
        self.decision_traces.clear()

    # ------------------------------------------------------------------------
    # 1. Army Purchasing & Setup Deployment (governed by `SetupGenes`)
    # ------------------------------------------------------------------------

    def purchase_wave_specs(
        self,
        round_number: int,
        budget: int,
        max_units: int = MAX_ARMY_SIZE,
        rng: Optional[random.Random] = None,
    ) -> List[CreatureSpec]:
        """
        Spend `budget` points to purchase up to `max_units` Mammal specs from the
        subset of Mammals unlocked on or before `round_number` (`get_unlocked_mammals`),
        using `genome.setup.mammal_purchase_weights` and `genome.setup.elite_vs_swarm_bias`.
        """
        if rng is None:
            rng = random.Random()

        setup = self.genome.setup
        remaining = max(1, budget)
        purchased: List[CreatureSpec] = []
        picked_counts: Dict[str, int] = {}
        unlocked_mammals = get_unlocked_mammals(round_number)
        max_unlocked_cost = max((s.point_cost for s in unlocked_mammals), default=1)

        while remaining > 0 and len(purchased) < max_units:
            slots_left = max_units - len(purchased)
            min_desired_cost = max(1, remaining // slots_left)

            affordable = [
                spec
                for spec in unlocked_mammals
                if spec.point_cost <= remaining
            ]
            if not affordable:
                break

            # Avoid filling early picks with 1-pt fodder when stronger mammals are affordable,
            # unless this bot's genome explicitly favors extreme swarm tactics
            tier_floor = (
                max(min_desired_cost, max_unlocked_cost // 3)
                if setup.elite_vs_swarm_bias >= -0.45
                else min_desired_cost
            )
            preferred = [s for s in affordable if s.point_cost >= tier_floor]
            pool = preferred if preferred else [max(affordable, key=lambda s: s.point_cost)]

            # Deterministic gene-driven selection (with gentle diminishing returns on duplicates)
            cost_exp = 1.15 + 0.70 * setup.elite_vs_swarm_bias

            def _score_candidate(spec: CreatureSpec) -> Tuple[float, int]:
                species_w = max(0.05, setup.mammal_purchase_weights.get(spec.creature_id, 1.0))
                already = picked_counts.get(spec.creature_id, 0)
                repeat_factor = 0.78 ** already
                val = species_w * (float(spec.point_cost) ** cost_exp) * repeat_factor
                return (round(val, 6), spec.point_cost)

            chosen = max(pool, key=_score_candidate)
            purchased.append(chosen)
            picked_counts[chosen.creature_id] = picked_counts.get(chosen.creature_id, 0) + 1
            remaining -= chosen.point_cost

        return purchased

    def purchase_and_deploy_army(
        self,
        state: rules.GameState,
        team: str,
        budget: int,
        rng: Optional[random.Random] = None,
    ) -> List[rules.Unit]:
        """
        Purchase a Mammal wave for `team` within `budget` and deploy each creature onto
        `team`'s legal half of the battlefield using `genome.setup` formation genes
        and `rules.deploy_team_unit_to_hex`.
        """
        if rng is None:
            rng = random.Random()

        to_remove = [uid for uid, u in state.units.items() if u.team == team]
        for uid in to_remove:
            del state.units[uid]

        wave_specs = self.purchase_wave_specs(
            state.round_number, budget, max_units=state.max_army_size, rng=rng
        )
        # Sort units so tanks and key pieces pick their preferred formation hexes first
        wave_specs.sort(
            key=lambda s: (s.max_hp + s.defense * 3, s.point_cost), reverse=True
        )

        valid_hexes = rules.get_valid_team_deploy_hexes(state, team)
        used_hexes: Set[HexCoord] = set()
        placed_coords: List[HexCoord] = []
        setup = self.genome.setup

        saved_phase = state.phase
        state.phase = PHASE_SETUP
        spawned: List[rules.Unit] = []
        prefix = "e" if team == TEAM_ENEMY else "p"

        try:
            for idx, spec in enumerate(wave_specs, start=1):
                is_tank = (spec.max_hp >= 18 and spec.defense >= 4) or spec.stage in (3, 6, 7)
                # Desired distance from the center dividing line |2*r + q| (range 1..8)
                base_line_dist = 1.0 + (1.0 - setup.frontline_aggression) * 5.0
                if is_tank:
                    desired_line_dist = max(1.0, base_line_dist - setup.tank_frontline_weight * 1.6)
                elif spec.can_fly or spec.num_rotate_moves >= 3:
                    desired_line_dist = min(7.0, base_line_dist + setup.tank_frontline_weight * 0.8)
                else:
                    desired_line_dist = base_line_dist

                best_hex: Optional[HexCoord] = None
                best_hex_score = -1e9
                for coord in valid_hexes:
                    if coord in used_hexes:
                        continue
                    q, r = coord
                    line_dist = abs(2 * r + q)
                    lateral = abs(q)

                    # Score proximity to desired frontline depth
                    h_score = -abs(line_dist - desired_line_dist) * 4.0

                    # Spread vs. cluster around center column and already-placed allies
                    if setup.spread_vs_cluster >= 0.0:
                        # Prefer spreading across wings (|q| around 1..3)
                        h_score += setup.spread_vs_cluster * min(lateral, 3) * 2.5
                        if placed_coords:
                            min_ally_d = min(hex_distance(coord, pc) for pc in placed_coords)
                            if min_ally_d == 1:
                                h_score -= setup.spread_vs_cluster * 1.5
                    else:
                        # Prefer tight pack cluster
                        cluster_strength = -setup.spread_vs_cluster
                        h_score -= cluster_strength * lateral * 2.2
                        if placed_coords:
                            adj_allies = sum(
                                1 for pc in placed_coords if hex_distance(coord, pc) == 1
                            )
                            h_score += cluster_strength * adj_allies * 3.2

                    # Deterministic center-out tie-breaker so setup genes produce reproducible formations
                    h_score -= abs(q) * 0.02 + abs(r) * 0.01
                    if h_score > best_hex_score:
                        best_hex_score = h_score
                        best_hex = coord

                if best_hex is None:
                    break

                # Determine initial facing (straight vs. angled inward from wings)
                q_pos = best_hex[0]
                if team == TEAM_ENEMY:
                    if setup.wing_inward_facing_bias > 0.5 and q_pos <= -2:
                        facing = FACING_SE
                    elif setup.wing_inward_facing_bias > 0.5 and q_pos >= 2:
                        facing = FACING_SW
                    else:
                        facing = FACING_S
                else:
                    if setup.wing_inward_facing_bias > 0.5 and q_pos <= -2:
                        facing = FACING_NE
                    elif setup.wing_inward_facing_bias > 0.5 and q_pos >= 2:
                        facing = FACING_NW
                    else:
                        facing = FACING_N

                uid = f"{prefix}_r{state.round_number}_{idx}_{spec.asset_slug}"
                unit = rules.Unit(
                    unit_id=uid,
                    spec=spec,
                    team=team,
                    hex_pos=None,
                    facing=facing,
                )
                state.units[uid] = unit
                assert rules.deploy_team_unit_to_hex(state, uid, best_hex, facing=facing)
                used_hexes.add(best_hex)
                placed_coords.append(best_hex)
                spawned.append(unit)
        finally:
            state.phase = saved_phase

        return spawned

    # ------------------------------------------------------------------------
    # 2. Target & Positional Evaluation (TargetingGenes + MammalTacticalGenes)
    # ------------------------------------------------------------------------

    def score_attack_on_target(
        self, state: rules.GameState, attacker: rules.Unit, target_hex: HexCoord
    ) -> float:
        """
        Score an immediate attack by `attacker` onto `target_hex` using
        `genome.targeting` and `genome.get_mammal_genes(attacker.spec.creature_id)`.
        """
        defender = state.get_unit_at_hex(target_hex, exclude_unit_id=attacker.unit_id)
        if defender is None or defender.team == attacker.team:
            return -1e9

        t_genes = self.genome.targeting
        m_genes = self.genome.get_mammal_genes(attacker.spec.creature_id)

        # Account for multi-arc cleave powers hitting additional adjacent enemies
        atk_powers = set(attacker.spec.special_powers)
        legal_targets = rules.get_legal_attack_targets(state, attacker.unit_id)
        if atk_powers & rules._CLEAVE_POWERS and target_hex in legal_targets:
            struck_hexes = [target_hex] + [h for h in legal_targets if h != target_hex]
        else:
            struck_hexes = [target_hex]

        total_score = 0.0
        for idx, th in enumerate(struck_hexes):
            dfd = state.get_unit_at_hex(th, exclude_unit_id=attacker.unit_id)
            if dfd is None or dfd.team == attacker.team:
                continue

            dmg, _, is_vuln, _ = rules.compute_attack_damage(state, attacker, dfd)
            effective_dmg = min(dfd.hp, dmg)
            overkill_dmg = max(0, dmg - effective_dmg)
            is_kill = dmg >= dfd.hp

            # Base damage reward (penalize wasting massive overkill on a 1-HP target)
            sub_score = effective_dmg * 30.0 - overkill_dmg * 3.5

            # Strong vs. weak target preference
            threat_metric = dfd.spec.point_cost * 3.5 + dfd.spec.attack * 2.0
            weakness_metric = max(0.0, 30.0 - dfd.hp) * 1.8
            if t_genes.strong_vs_weak_bias >= 0.0:
                sub_score += t_genes.strong_vs_weak_bias * threat_metric * 3.0
                sub_score += (1.0 - t_genes.strong_vs_weak_bias) * weakness_metric
            else:
                weak_bias = -t_genes.strong_vs_weak_bias
                sub_score += weak_bias * weakness_metric * 3.0
                sub_score += (1.0 - weak_bias) * threat_metric

            # "Both" hybrid strategy: reward confirming a kill scaled by the target's point value & threat
            if is_kill:
                kill_bonus = (110.0 + dfd.spec.point_cost * 32.0 + dfd.spec.attack * 12.0)
                sub_score += kill_bonus * max(0.25, t_genes.hybrid_kill_confirm_weight)

            # Focus-fire bonus on already-wounded targets
            if dfd.hp < dfd.spec.max_hp:
                missing_frac = (dfd.spec.max_hp - dfd.hp) / max(1.0, float(dfd.spec.max_hp))
                sub_score += missing_frac * 55.0 * t_genes.focus_fire_weight

            # Vulnerable arc bonus
            if is_vuln:
                sub_score += 42.0 * t_genes.vulnerable_arc_bonus_weight

            # Retaliation / thorns caution
            def_powers = set(dfd.spec.special_powers)
            ret_dmg = 0
            if "molten_carapace" in def_powers:
                ret_dmg = 2
            elif "thorny_hide" in def_powers or "spiked_ridge" in def_powers:
                ret_dmg = 1
            if ret_dmg > 0:
                if attacker.hp <= ret_dmg:
                    sub_score -= 320.0 * t_genes.thorns_caution_weight
                else:
                    sub_score -= ret_dmg * 24.0 * t_genes.thorns_caution_weight

            if idx > 0:
                # Cleave secondary target bonus
                sub_score += m_genes.cleave_positioning_weight * 2.5

            total_score += sub_score

        return total_score

    def score_unit_position(
        self,
        state: rules.GameState,
        unit: rules.Unit,
        cand_hex: HexCoord,
        cand_facing: int,
        attacked_this_turn: bool,
    ) -> float:
        """
        Evaluate the end-of-turn tactical quality of `(cand_hex, cand_facing)` for `unit`
        using its species-specific `MammalTacticalGenes` and `GlobalCoordinationGenes`.
        """
        opp_units = [
            u
            for u in state.units.values()
            if u.is_deployed and u.team != unit.team and u.hex_pos is not None
        ]
        if not opp_units:
            return 1000.0

        ally_units = [
            u
            for u in state.units.values()
            if u.is_deployed
            and u.team == unit.team
            and u.unit_id != unit.unit_id
            and u.hex_pos is not None
        ]

        m_genes = self.genome.get_mammal_genes(unit.spec.creature_id)
        c_genes = self.genome.coordination

        # 1. Distance to primary target (weighted by strong_vs_weak_bias + focus_fire)
        def _target_priority(ou: rules.Unit) -> float:
            assert ou.hex_pos is not None
            d = hex_distance(cand_hex, ou.hex_pos)
            wounded_bonus = (ou.spec.max_hp - ou.hp) * 1.5 * self.genome.targeting.focus_fire_weight
            bias_term = self.genome.targeting.strong_vs_weak_bias * ou.spec.point_cost * 1.5
            return d * 10.0 - wounded_bonus - bias_term

        primary_opp = min(opp_units, key=_target_priority)
        assert primary_opp.hex_pos is not None
        dist_primary = hex_distance(cand_hex, primary_opp.hex_pos)

        wounded_ratio = unit.hp / max(1.0, float(unit.spec.max_hp))
        wants_retreat = (
            wounded_ratio <= c_genes.wounded_retreat_threshold and attacked_this_turn
        )

        if wants_retreat:
            # Prefer stepping back to distance 2..3 after attacking when critically wounded
            score = -abs(dist_primary - 2) * (m_genes.hit_and_run_weight + 15.0)
        else:
            score = -abs(dist_primary - 1) * m_genes.aggression_weight - dist_primary * 5.0
            # Reward stepping around allied blockers onto an open approach lane
            if dist_primary > 1:
                has_open_lane = any(
                    is_valid_hex(hex_neighbor(cand_hex, d)[0], hex_neighbor(cand_hex, d)[1], state.arena_radius)
                    and hex_distance(hex_neighbor(cand_hex, d), primary_opp.hex_pos) < dist_primary
                    and state.get_unit_at_hex(hex_neighbor(cand_hex, d), exclude_unit_id=unit.unit_id) is None
                    for d in range(6)
                )
                if has_open_lane:
                    score += 16.0

        # 2. Orient facing toward the nearest opposing unit
        nearest_opp = min(
            opp_units,
            key=lambda ou: hex_distance(cand_hex, ou.hex_pos),  # type: ignore[arg-type]
        )
        assert nearest_opp.hex_pos is not None
        best_dir = min(
            range(6),
            key=lambda d: hex_distance(
                hex_neighbor(cand_hex, d), nearest_opp.hex_pos  # type: ignore[arg-type]
            ),
        )
        score -= hex_angle_distance(cand_facing, best_dir) * (8.0 + 0.45 * m_genes.self_defense_weight)

        # 3. Flanking, Cleave alignment, Hit-and-Run safety & Self-Defense around nearby enemies
        enemies_in_my_arc = 0
        for ou in opp_units:
            if ou.hex_pos is None:
                continue
            d = hex_distance(cand_hex, ou.hex_pos)
            if d == 1:
                # Incoming direction on `ou` when we attack `ou` (from `ou.hex_pos` to `cand_hex`)
                inc_on_ou = hex_direction_between(ou.hex_pos, cand_hex)
                if inc_on_ou is not None:
                    rel_on_ou = relative_direction(ou.facing, inc_on_ou)
                    if rel_on_ou in ou.spec.vulnerable_arcs:
                        score += m_genes.flanking_weight

                    # Is `cand_hex` directly inside `ou`'s current attack_arcs?
                    if rel_on_ou in ou.spec.attack_arcs:
                        if attacked_this_turn:
                            score -= m_genes.hit_and_run_weight
                        else:
                            score -= m_genes.self_defense_weight * 0.45

                # Incoming direction on us when `ou` attacks us (from `cand_hex` to `ou.hex_pos`)
                dir_to_ou = hex_direction_between(cand_hex, ou.hex_pos)
                if dir_to_ou is not None:
                    rel_to_ou = relative_direction(cand_facing, dir_to_ou)
                    if rel_to_ou in unit.spec.attack_arcs:
                        enemies_in_my_arc += 1
                        score += 14.0

                    # Are we exposing our own vulnerable_arcs to `ou`?
                    if rel_to_ou in unit.spec.vulnerable_arcs:
                        score -= m_genes.self_defense_weight * 1.15

        if enemies_in_my_arc >= 2:
            score += (enemies_in_my_arc - 1) * m_genes.cleave_positioning_weight

        # 4. Pack Cohesion (staying adjacent to allies / double-teaming enemies)
        if ally_units:
            adj_allies = sum(
                1
                for au in ally_units
                if au.hex_pos is not None and hex_distance(cand_hex, au.hex_pos) == 1
            )
            score += min(adj_allies, 2) * (0.55 * m_genes.pack_cohesion_weight)
            # Pack Tactics / Alpha Howl synergy when adjacent to an enemy that an ally also threatens
            if "pack_tactics" in unit.spec.special_powers or "alpha_howl" in unit.spec.special_powers:
                for ou in opp_units:
                    if ou.hex_pos is not None and hex_distance(cand_hex, ou.hex_pos) == 1:
                        ally_adj_to_ou = any(
                            au.hex_pos is not None and hex_distance(au.hex_pos, ou.hex_pos) == 1
                            for au in ally_units
                        )
                        if ally_adj_to_ou:
                            score += m_genes.pack_cohesion_weight * 1.4

        return score

    # ------------------------------------------------------------------------
    # 3. Single-Unit Candidate Turn Plan Generator (Strictly via `rules.py`)
    # ------------------------------------------------------------------------

    def _generate_unit_candidate_plans(
        self, state: rules.GameState, unit_id: str, max_plans: int = 16
    ) -> List[_SimulatedPlan]:
        """
        Explore legal single-unit turn action sequences `(rotate / move / attack)` for
        `unit_id` on cloned `GameState` instances using shared `rules.py` helpers.
        """
        unit = state.get_unit(unit_id)
        if unit is None or not unit.is_deployed or unit.hex_pos is None:
            return []

        plans: List[_SimulatedPlan] = []
        visited_signatures: Set[Tuple[HexCoord, int, int, int]] = set()

        def _record_terminal_plan(
            sim_state: rules.GameState,
            steps: List[AtomicBotStep],
            atk_score_accum: float,
            desc_parts: List[str],
        ) -> None:
            u_now = sim_state.get_unit(unit_id)
            if u_now is None or not u_now.is_deployed or u_now.hex_pos is None:
                # Unit died to thorns during its own attack; still record if it got a kill
                end_h = unit.hex_pos if unit.hex_pos is not None else (0, 0)
                end_f = unit.facing
                pos_sc = -250.0
            else:
                if rules.is_unit_overlapping(sim_state, unit_id):
                    # Illegal to end turn overlapping another unit!
                    return
                end_h = u_now.hex_pos
                end_f = u_now.facing
                pos_sc = self.score_unit_position(
                    sim_state,
                    u_now,
                    end_h,
                    end_f,
                    attacked_this_turn=(u_now.attacks_used > 0),
                )

            sig = (end_h, end_f, int(round(atk_score_accum)), len(steps))
            if sig in visited_signatures:
                return
            visited_signatures.add(sig)

            ply1_sc = atk_score_accum + pos_sc
            summary = " -> ".join(desc_parts) if desc_parts else "hold"
            plans.append(
                _SimulatedPlan(
                    steps=list(steps),
                    final_state=sim_state,
                    end_hex=end_h,
                    end_facing=end_f,
                    immediate_attack_score=atk_score_accum,
                    positional_score=pos_sc,
                    ply1_score=ply1_sc,
                    summary=summary,
                )
            )

        def _maybe_attack_at_juncture(
            sim_state: rules.GameState,
            steps: List[AtomicBotStep],
            atk_score_accum: float,
            desc_parts: List[str],
        ) -> Tuple[rules.GameState, List[AtomicBotStep], float, List[str]]:
            """If `unit_id` has a legal attack right now, greedily pick the highest-scoring target."""
            targets = rules.get_legal_attack_targets(sim_state, unit_id)
            if not targets:
                return (sim_state, steps, atk_score_accum, desc_parts)
            u_cur = sim_state.get_unit(unit_id)
            if u_cur is None:
                return (sim_state, steps, atk_score_accum, desc_parts)

            best_t = max(targets, key=lambda th: self.score_attack_on_target(sim_state, u_cur, th))
            best_sc = self.score_attack_on_target(sim_state, u_cur, best_t)
            if best_sc <= -1e8:
                return (sim_state, steps, atk_score_accum, desc_parts)

            s_atk = rules.clone_game_state(sim_state)
            res = rules.execute_attack(s_atk, unit_id, best_t)
            if not res:
                return (sim_state, steps, atk_score_accum, desc_parts)

            new_steps = list(steps) + [AtomicBotStep(action_type="attack", target_hex=best_t)]
            new_desc = list(desc_parts) + [f"atk{best_t}"]
            return (s_atk, new_steps, atk_score_accum + best_sc, new_desc)

        # Root state (optionally attacking at BEFORE_ROTATE)
        base_clone = rules.clone_game_state(state)
        _record_terminal_plan(base_clone, [], 0.0, [])

        s_before_atk, steps_b, sc_b, desc_b = _maybe_attack_at_juncture(
            base_clone, [], 0.0, []
        )
        if steps_b:
            _record_terminal_plan(s_before_atk, steps_b, sc_b, desc_b)
            roots = [(s_before_atk, steps_b, sc_b, desc_b), (base_clone, [], 0.0, [])]
        else:
            roots = [(base_clone, [], 0.0, [])]

        # Expand up to `unit.spec.num_rotate_moves` steps using beam pruning per step
        beam = roots
        max_steps = unit.spec.num_rotate_moves
        for _step_i in range(max_steps):
            next_beam: List[Tuple[rules.GameState, List[AtomicBotStep], float, List[str]]] = []
            for cur_st, cur_steps, cur_sc, cur_desc in beam:
                u_cur = cur_st.get_unit(unit_id)
                if u_cur is None or not u_cur.is_deployed or cur_st.phase != PHASE_BATTLE:
                    continue

                # Candidate rotations (including staying at current facing)
                legal_rots = rules.get_legal_rotations(cur_st, unit_id)
                cand_facings = [u_cur.facing] + [f for f in legal_rots if f != u_cur.facing]

                for cf in cand_facings:
                    if cf == u_cur.facing:
                        st_rot = cur_st
                        steps_rot = cur_steps
                        sc_rot = cur_sc
                        desc_rot = cur_desc
                    else:
                        st_rot = rules.clone_game_state(cur_st)
                        if not rules.rotate_unit(st_rot, unit_id, cf):
                            continue
                        steps_rot = list(cur_steps) + [
                            AtomicBotStep(action_type="rotate", new_facing=cf)
                        ]
                        sc_rot = cur_sc
                        desc_rot = list(cur_desc) + [f"rot({ABS_FACING_NAMES[cf].split()[0]})"]

                        # Check AFTER_ROTATE attack opportunity
                        st_rot_atk, steps_ra, sc_ra, desc_ra = _maybe_attack_at_juncture(
                            st_rot, steps_rot, sc_rot, desc_rot
                        )
                        if len(steps_ra) > len(steps_rot):
                            _record_terminal_plan(st_rot_atk, steps_ra, sc_ra, desc_ra)
                            st_rot, steps_rot, sc_rot, desc_rot = (
                                st_rot_atk,
                                steps_ra,
                                sc_ra,
                                desc_ra,
                            )
                        else:
                            _record_terminal_plan(st_rot, steps_rot, sc_rot, desc_rot)

                    # Candidate moves from `st_rot`
                    legal_mvs = rules.get_legal_moves(st_rot, unit_id)
                    for dest in legal_mvs:
                        st_mv = rules.clone_game_state(st_rot)
                        if not rules.move_unit(st_mv, unit_id, dest):
                            continue
                        steps_mv = list(steps_rot) + [
                            AtomicBotStep(action_type="move", target_hex=dest)
                        ]
                        desc_mv = list(desc_rot) + [f"mv{dest}"]
                        sc_mv = sc_rot

                        # Check AFTER_MOVE attack opportunity
                        st_mv_atk, steps_ma, sc_ma, desc_ma = _maybe_attack_at_juncture(
                            st_mv, steps_mv, sc_mv, desc_mv
                        )
                        if len(steps_ma) > len(steps_mv):
                            _record_terminal_plan(st_mv_atk, steps_ma, sc_ma, desc_ma)
                            next_beam.append((st_mv_atk, steps_ma, sc_ma, desc_ma))
                        else:
                            _record_terminal_plan(st_mv, steps_mv, sc_mv, desc_mv)
                            next_beam.append((st_mv, steps_mv, sc_mv, desc_mv))

            if not next_beam:
                break
            # Keep top 8 intermediate branches per Rotate-Move cycle
            next_beam.sort(
                key=lambda item: (
                    item[2]
                    + (
                        self.score_unit_position(
                            item[0],
                            item[0].get_unit(unit_id),  # type: ignore[arg-type]
                            item[0].get_unit(unit_id).hex_pos,  # type: ignore[union-attr,arg-type]
                            item[0].get_unit(unit_id).facing,  # type: ignore[union-attr]
                            attacked_this_turn=(item[2] > 0),
                        )
                        if item[0].get_unit(unit_id) is not None
                        and item[0].get_unit(unit_id).hex_pos is not None  # type: ignore[union-attr]
                        else -500.0
                    )
                ),
                reverse=True,
            )
            beam = next_beam[:8]

        plans.sort(key=lambda p: p.ply1_score, reverse=True)
        return plans[:max_plans]

    # ------------------------------------------------------------------------
    # 4. X-Ply Lookahead Evaluation (1, 2, 3, and 4 Ply)
    # ------------------------------------------------------------------------

    def _simulate_greedy_unit_action(
        self, st: rules.GameState, u_live: rules.Unit
    ) -> Tuple[float, rules.GameState]:
        """
        Execute the best fast 1-step greedy action (Attack, Rotate->Move->Attack, or
        Rotate->Move approach) for `u_live` on `st` during lookahead rollout.
        """
        if u_live.hex_pos is None:
            return (0.0, st)

        best_impact = 0.0
        best_state: Optional[rules.GameState] = None

        # 1. Check immediate BEFORE_ROTATE attack
        now_targets = rules.get_legal_attack_targets(st, u_live.unit_id)
        if now_targets:
            t_hex = max(
                now_targets,
                key=lambda th: self.score_attack_on_target(st, u_live, th),
            )
            imp = max(0.0, self.score_attack_on_target(st, u_live, t_hex))
            s_try = rules.clone_game_state(st)
            if rules.execute_attack(s_try, u_live.unit_id, t_hex):
                best_impact = imp
                best_state = s_try

        # 2. Check Rotate -> Move -> Attack across up to `num_rotate_moves` steps
        cur_sim = best_state if best_state is not None else rules.clone_game_state(st)
        accum_impact = best_impact
        for _ in range(u_live.spec.num_rotate_moves):
            u_cur = cur_sim.get_unit(u_live.unit_id)
            if u_cur is None or not u_cur.is_deployed or u_cur.hex_pos is None:
                break
            step_idx = rules._move_step_index(u_cur)
            if step_idx is None:
                break

            cand_rots = [u_cur.facing] + [
                f
                for f in rules.get_legal_rotations(cur_sim, u_cur.unit_id)
                if f != u_cur.facing
            ]
            best_step_sc = -1e9
            best_step_cf = u_cur.facing
            best_step_dest = u_cur.hex_pos
            can_atk = u_cur.attacks_used < u_cur.spec.total_attacks_per_move

            for cf in cand_rots:
                if not rules.is_unit_overlapping(cur_sim, u_cur.unit_id):
                    stay_sc = rules._score_ai_state(
                        cur_sim, u_cur, u_cur.hex_pos, cf, can_atk
                    )
                    if stay_sc > best_step_sc:
                        best_step_sc = stay_sc
                        best_step_cf = cf
                        best_step_dest = u_cur.hex_pos
                for dest in rules._get_legal_moves_for_step_and_facing(
                    cur_sim, u_cur, step_idx, cf
                ):
                    mv_sc = rules._score_ai_state(cur_sim, u_cur, dest, cf, can_atk)
                    if mv_sc > best_step_sc:
                        best_step_sc = mv_sc
                        best_step_cf = cf
                        best_step_dest = dest

            if best_step_cf != u_cur.facing:
                rules.rotate_unit(cur_sim, u_cur.unit_id, best_step_cf)
            if best_step_dest != u_cur.hex_pos:
                rules.move_unit(cur_sim, u_cur.unit_id, best_step_dest)

            u_after = cur_sim.get_unit(u_live.unit_id)
            if u_after is not None:
                t_list = rules.get_legal_attack_targets(cur_sim, u_after.unit_id)
                if t_list:
                    th = max(
                        t_list,
                        key=lambda h: self.score_attack_on_target(cur_sim, u_after, h),
                    )
                    imp = max(0.0, self.score_attack_on_target(cur_sim, u_after, th))
                    if rules.execute_attack(cur_sim, u_after.unit_id, th):
                        accum_impact += imp

        return (accum_impact, cur_sim)

    def _simulate_remaining_allies_in_turn(
        self, sim_state: rules.GameState, acting_unit_id: str
    ) -> Tuple[float, rules.GameState]:
        """
        Complete the current team's turn by simulating the remaining unacted allied
        units (so coordinated kills happen before the opponent's counter-turn).
        """
        st = rules.clone_game_state(sim_state)
        if st.phase != PHASE_BATTLE:
            return (0.0, st)

        my_team = st.active_team
        acted = set(st.ai_acted_unit_ids) | {acting_unit_id}
        remaining_allies = [
            u
            for u in st.units.values()
            if u.is_deployed
            and u.team == my_team
            and u.hex_pos is not None
            and u.unit_id not in acted
        ]
        opp_units = [
            u
            for u in st.units.values()
            if u.is_deployed and u.team != my_team and u.hex_pos is not None
        ]
        if not remaining_allies or not opp_units:
            return (0.0, st)

        remaining_allies.sort(
            key=lambda u: min(
                hex_distance(u.hex_pos, ou.hex_pos)  # type: ignore[arg-type]
                for ou in opp_units
                if ou.hex_pos is not None
            )
        )

        ally_impact = 0.0
        for ally in remaining_allies:
            if st.phase != PHASE_BATTLE:
                break
            a_live = st.get_unit(ally.unit_id)
            if a_live is None or not a_live.is_deployed:
                continue
            imp, st = self._simulate_greedy_unit_action(st, a_live)
            ally_impact += imp

        return (ally_impact, st)

    def _estimate_team_counter_turn_impact(
        self, sim_state: rules.GameState, acting_team: str
    ) -> Tuple[float, rules.GameState]:
        """
        Simulate `acting_team`'s full counter-turn on `sim_state` (using shared
        `rules.py` helpers) and return `(counter_turn_damage_and_kill_score, resulting_state)`.
        Used at Ply 2 and Ply 4 to evaluate the opponent's response, and at Ply 3 for
        our own follow-up turn.
        """
        st = rules.clone_game_state(sim_state)
        if st.phase != PHASE_BATTLE:
            # One side has already been eliminated
            any_acting_alive = any(
                u.is_alive and u.team == acting_team for u in st.units.values()
            )
            return ((600.0 if any_acting_alive else 0.0), st)

        st.active_team = acting_team
        st.ai_acted_unit_ids.clear()
        for u in st.units.values():
            if u.is_alive:
                u.reset_for_turn()

        units_to_act = [
            u
            for u in st.units.values()
            if u.is_deployed and u.team == acting_team and u.hex_pos is not None
        ]
        opp_units = [
            u
            for u in st.units.values()
            if u.is_deployed and u.team != acting_team and u.hex_pos is not None
        ]
        if not units_to_act or not opp_units:
            return (0.0, st)

        units_to_act.sort(
            key=lambda u: min(
                hex_distance(u.hex_pos, ou.hex_pos)  # type: ignore[arg-type]
                for ou in opp_units
                if ou.hex_pos is not None
            )
        )

        total_impact = 0.0
        for u in units_to_act:
            if st.phase != PHASE_BATTLE:
                break
            u_live = st.get_unit(u.unit_id)
            if u_live is None or not u_live.is_deployed:
                continue
            imp, st = self._simulate_greedy_unit_action(st, u_live)
            total_impact += imp

        return (total_impact, st)

    def select_best_unit_plan(
        self, state: rules.GameState, unit_id: str
    ) -> Optional[_SimulatedPlan]:
        """
        Evaluate candidate turn plans for `unit_id` using `genome.lookahead.lookahead_ply`
        (1, 2, 3, or 4 plies) and record a `DecisionTrace` when telemetry is enabled.
        """
        unit = state.get_unit(unit_id)
        if unit is None or not unit.is_deployed:
            return None

        l_genes = self.genome.lookahead
        ply = l_genes.lookahead_ply
        beam_w = l_genes.beam_width

        candidates = self._generate_unit_candidate_plans(
            state, unit_id, max_plans=max(10, beam_w * 3)
        )
        if not candidates:
            return None

        my_team = unit.team
        opp_team = TEAM_PLAYER if my_team == TEAM_ENEMY else TEAM_ENEMY

        traces: List[CandidatePlanTrace] = []
        best_plan = candidates[0]
        best_total_score = -1e9

        eval_pool = candidates if ply == 1 else candidates[:beam_w]
        for plan in eval_pool:
            enemy_penalty = 0.0
            followup_net = 0.0

            if ply >= 2:
                # Complete our team's remaining allies for this turn so coordinated kills
                # are reflected before the opponent's counter-turn
                ally_rest_sc, st_end_of_our_turn = self._simulate_remaining_allies_in_turn(
                    plan.final_state, unit_id
                )
                # Ply 2: Simulate opponent's full counter-turn response
                opp_impact, st_after_opp = self._estimate_team_counter_turn_impact(
                    st_end_of_our_turn, opp_team
                )
                enemy_penalty = opp_impact * l_genes.enemy_response_weight - ally_rest_sc * 0.35

                if ply == 2:
                    # Evaluate surviving counter-strike readiness after the enemy's reply so
                    # supported pack advances aren't paralyzed while solo suicidal lunges are punished
                    surviving_counter, _ = self._estimate_team_counter_turn_impact(
                        st_after_opp, my_team
                    )
                    followup_net = surviving_counter * (0.48 * l_genes.followup_discount)
                elif ply >= 3:
                    # Ply 3: Simulate our own follow-up turn after opponent's response
                    my_followup, st_after_my_followup = self._estimate_team_counter_turn_impact(
                        st_after_opp, my_team
                    )
                    followup_net = my_followup * l_genes.followup_discount

                    if ply >= 4:
                        # Ply 4: Also simulate the opponent's 2nd counter-turn response!
                        opp_second_impact, st_after_opp2 = self._estimate_team_counter_turn_impact(
                            st_after_my_followup, opp_team
                        )
                        surviving_r2, _ = self._estimate_team_counter_turn_impact(
                            st_after_opp2, my_team
                        )
                        followup_net -= (
                            opp_second_impact
                            * l_genes.enemy_response_weight
                            * l_genes.followup_discount
                        )
                        followup_net += surviving_r2 * (0.30 * l_genes.followup_discount)

            total_sc = plan.ply1_score - enemy_penalty + followup_net
            traces.append(
                CandidatePlanTrace(
                    plan_summary=plan.summary,
                    end_hex=plan.end_hex,
                    end_facing=plan.end_facing,
                    immediate_attack_score=round(plan.immediate_attack_score, 2),
                    positional_score=round(plan.positional_score, 2),
                    ply1_score=round(plan.ply1_score, 2),
                    enemy_response_penalty=round(enemy_penalty, 2),
                    followup_net_score=round(followup_net, 2),
                    total_score=round(total_sc, 2),
                )
            )
            if total_sc > best_total_score:
                best_total_score = total_sc
                best_plan = plan

        if self.record_telemetry and traces:
            traces.sort(key=lambda t: t.total_score, reverse=True)
            chosen_tr = traces[0]
            self.decision_traces.append(
                DecisionTrace(
                    turn_number=state.turn_number,
                    team=my_team,
                    unit_id=unit.unit_id,
                    unit_name=unit.spec.name,
                    lookahead_ply=ply,
                    chosen_plan_summary=chosen_tr.plan_summary,
                    chosen_1ply_score=chosen_tr.ply1_score,
                    chosen_enemy_response_penalty=chosen_tr.enemy_response_penalty,
                    chosen_total_score=chosen_tr.total_score,
                    top_candidates=traces[:beam_w],
                )
            )

        return best_plan

    # ------------------------------------------------------------------------
    # 5. Live & Headless Turn Execution via Shared `rules.py` Helpers
    # ------------------------------------------------------------------------

    def _ordered_unacted_units(self, state: rules.GameState) -> List[rules.Unit]:
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
        if not opp_units:
            my_units.sort(key=lambda u: u.unit_id)
            return my_units

        bias = self.genome.coordination.activation_order_bias
        my_units.sort(
            key=lambda u: (
                min(
                    hex_distance(u.hex_pos, ou.hex_pos)  # type: ignore[arg-type]
                    for ou in opp_units
                    if ou.hex_pos is not None
                )
                - bias * (u.spec.num_rotate_moves * 0.8 + u.spec.attack * 0.15),
                u.unit_id,
            )
        )
        return my_units

    def execute_next_action(
        self, state: rules.GameState
    ) -> Optional[rules.AIActionRecord]:
        """
        Execute the next atomic action (`rotate_unit`, `move_unit`, `execute_attack`,
        or `end_turn`) for `state.active_team` using ONLY shared `rules.py` helpers.
        """
        if state.phase != PHASE_BATTLE:
            self._pending_unit_id = None
            self._pending_steps.clear()
            return None

        # 1. If we have queued steps for the currently active unit, execute the next step
        while self._pending_unit_id is not None and self._pending_steps:
            u = state.get_unit(self._pending_unit_id)
            if (
                u is None
                or not u.is_deployed
                or u.team != state.active_team
                or u.unit_id in state.ai_acted_unit_ids
            ):
                self._pending_unit_id = None
                self._pending_steps.clear()
                break

            step = self._pending_steps.pop(0)
            if not self._pending_steps:
                # Mark unit finished once its last planned step is popped
                state.ai_acted_unit_ids.add(u.unit_id)

            if step.action_type == "rotate" and step.new_facing is not None:
                if rules.rotate_unit(state, u.unit_id, step.new_facing):
                    return rules.AIActionRecord(
                        action_type="rotate",
                        unit_id=u.unit_id,
                        unit_name=u.spec.name,
                        new_facing=step.new_facing,
                        description=(
                            f"{u.spec.name} rotated to {ABS_FACING_NAMES[step.new_facing]}."
                        ),
                    )
            elif step.action_type == "move" and step.target_hex is not None:
                if rules.move_unit(state, u.unit_id, step.target_hex):
                    return rules.AIActionRecord(
                        action_type="move",
                        unit_id=u.unit_id,
                        unit_name=u.spec.name,
                        target_hex=step.target_hex,
                        description=f"{u.spec.name} moved to {step.target_hex}.",
                    )
            elif step.action_type == "attack" and step.target_hex is not None:
                results = rules.execute_attack(state, u.unit_id, step.target_hex)
                if results:
                    first = results[0]
                    desc = (
                        f"{u.spec.name} attacked {first.defender_name} "
                        f"for {first.damage} dmg"
                        + (" (SLAIN!)" if first.defender_slain else "")
                    )
                    return rules.AIActionRecord(
                        action_type="attack",
                        unit_id=u.unit_id,
                        unit_name=u.spec.name,
                        target_hex=step.target_hex,
                        attack_results=results,
                        description=desc,
                    )

        # 2. Otherwise, pick the next unacted unit on `state.active_team` and plan its turn
        for unit in self._ordered_unacted_units(state):
            state.selected_unit_id = unit.unit_id
            plan = self.select_best_unit_plan(state, unit.unit_id)
            if plan is None or not plan.steps:
                state.ai_acted_unit_ids.add(unit.unit_id)
                continue

            self._pending_unit_id = unit.unit_id
            self._pending_steps = list(plan.steps)
            return self.execute_next_action(state)

        # 3. Safety check: if any flying unit is still overlapping, force it onto a legal empty landing hex
        for unit in list(state.units.values()):
            if (
                unit.is_deployed
                and unit.team == state.active_team
                and rules.is_unit_overlapping(state, unit.unit_id)
            ):
                legal_mvs = rules.get_legal_moves(state, unit.unit_id)
                empty_mvs = [
                    h
                    for h in legal_mvs
                    if state.get_unit_at_hex(h, exclude_unit_id=unit.unit_id) is None
                ]
                target_list = empty_mvs if empty_mvs else legal_mvs
                if target_list and rules.move_unit(state, unit.unit_id, target_list[0]):
                    return rules.AIActionRecord(
                        action_type="move",
                        unit_id=unit.unit_id,
                        unit_name=unit.spec.name,
                        target_hex=target_list[0],
                        description=f"{unit.spec.name} landed on {target_list[0]}.",
                    )

        # 4. All units on `state.active_team` have completed their actions -> end turn
        self._pending_unit_id = None
        self._pending_steps.clear()
        rules.end_turn(state)
        return rules.AIActionRecord(
            action_type="end_turn",
            unit_id=None,
            description=f"{self.name} turn complete — Turn {state.turn_number}.",
        )

    def run_turn(
        self, state: rules.GameState, max_actions: int = 250
    ) -> List[rules.AIActionRecord]:
        """Execute a complete turn for `state.active_team` synchronously."""
        starting_team = state.active_team
        records: List[rules.AIActionRecord] = []
        for _ in range(max_actions):
            if state.phase != PHASE_BATTLE or state.active_team != starting_team:
                break
            rec = self.execute_next_action(state)
            if rec is None:
                break
            records.append(rec)
            if rec.action_type == "end_turn":
                break
        return records


BotControllerType = Union[GeneticBotController, LegacyBotController]


# ============================================================================
# Game Recording & Diagnostic Instrumentation (`BattleReplayLog`)
# ============================================================================

@dataclass
class BattleReplayLog:
    """
    Complete recorded transcript of a headless battle, including initial army
    setups, turn-by-turn atomic actions, lookahead `DecisionTrace` telemetry,
    outcome metrics, and upset diagnostics.
    """
    match_id: str
    match_type: str  # "mammal_vs_mammal" or "mammal_vs_reptiles"
    round_number: int
    wave_budget: int
    north_bot_id: str
    north_bot_name: str
    north_ply: int
    south_bot_id: str
    south_bot_name: str
    south_ply: int
    winner_team: Optional[str]
    winner_bot_id: Optional[str]
    turns_played: int
    north_remaining_hp: int
    south_remaining_hp: int
    north_remaining_pts: int
    south_remaining_pts: int
    is_upset: bool = False
    upset_reason: str = ""
    initial_units: List[Dict[str, Any]] = field(default_factory=list)
    turn_logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BattleReplayLog:
        return cls(**data)

    def save_json(self, filepath: str) -> None:
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> BattleReplayLog:
        with open(filepath, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def format_human_readable(self, max_turns: int = 50) -> str:
        """Render a clear, human-readable diagnostic report of the recorded battle."""
        lines: List[str] = [
            "=" * 86,
            f"BATTLE REPLAY LOG: {self.match_id}  ({self.match_type})",
            f"Round {self.round_number}  •  Wave Budget: {self.wave_budget} pts  •  Turns Played: {self.turns_played}",
            f"  NORTH (TEAM_ENEMY) : {self.north_bot_name} [{self.north_bot_id}] (Ply {self.north_ply})",
            f"  SOUTH (TEAM_PLAYER): {self.south_bot_name} [{self.south_bot_id}] (Ply {self.south_ply})",
            f"  WINNER             : {self.winner_bot_id or 'DRAW'} (Team: {self.winner_team or 'NONE'})",
            f"  REMAINING HP / PTS : North {self.north_remaining_hp} HP ({self.north_remaining_pts} pts) vs "
            f"South {self.south_remaining_hp} HP ({self.south_remaining_pts} pts)",
        ]
        if self.is_upset:
            lines.append(f"  [!] UPSET FLAGGED  : {self.upset_reason}")
        lines.append("-" * 86)
        lines.append("INITIAL DEPLOYMENT:")
        for u in self.initial_units:
            lines.append(
                f"  [{u['team'].upper():6s}] {u['unit_id']:22s} ({u['name']:15s}, {u['point_cost']:2d}pt, "
                f"HP {u['max_hp']:2d}) @ {tuple(u['hex_pos'])} facing {ABS_FACING_NAMES[u['facing']]}"
            )
        lines.append("-" * 86)

        for idx, tlog in enumerate(self.turn_logs[:max_turns]):
            lines.append(
                f"Turn {tlog['turn_number']} — Active Team: {tlog['active_team'].upper()} "
                f"({tlog['bot_name']} [Ply {tlog['bot_ply']}])"
            )
            for dt in tlog.get("decision_traces", []):
                lines.append(
                    f"  [LOOKAHEAD Ply {dt['lookahead_ply']}] {dt['unit_name']} ({dt['unit_id']}) "
                    f"chose '{dt['chosen_plan_summary']}' | 1-Ply={dt['chosen_1ply_score']:.1f}, "
                    f"EnemyRespPenalty={dt['chosen_enemy_response_penalty']:.1f}, "
                    f"Total={dt['chosen_total_score']:.1f}"
                )
            for act in tlog.get("actions", []):
                if act["action_type"] == "end_turn":
                    continue
                lines.append(f"    -> {act['description']}")
        lines.append("=" * 86)
        return "\n".join(lines)


# ============================================================================
# UI-Free Headless Battle Arena (Symmetric Skirmishes & Reptile Benchmarks)
# ============================================================================

def _snapshot_initial_units(state: rules.GameState) -> List[Dict[str, Any]]:
    snaps: List[Dict[str, Any]] = []
    for uid in sorted(state.units.keys()):
        u = state.units[uid]
        if not u.is_deployed or u.hex_pos is None:
            continue
        snaps.append(
            {
                "unit_id": u.unit_id,
                "name": u.spec.name,
                "creature_id": u.spec.creature_id,
                "team": u.team,
                "point_cost": u.spec.point_cost,
                "max_hp": u.spec.max_hp,
                "hex_pos": list(u.hex_pos),
                "facing": u.facing,
            }
        )
    return snaps


def _execute_headless_battle_loop(
    state: rules.GameState,
    north_bot: BotControllerType,
    south_bot: BotControllerType,
    match_id: str,
    match_type: str,
    max_turns: int = 36,
    record_log: bool = True,
) -> BattleReplayLog:
    """
    Run a headless battle to completion between `north_bot` (`TEAM_ENEMY`, upper half)
    and `south_bot` (`TEAM_PLAYER`, lower half) strictly through `rules.py`.
    """
    initial_units = _snapshot_initial_units(state) if record_log else []
    turn_logs: List[Dict[str, Any]] = []

    if isinstance(north_bot, GeneticBotController):
        north_bot.record_telemetry = record_log
        north_bot.decision_traces.clear()
    if isinstance(south_bot, GeneticBotController):
        south_bot.record_telemetry = record_log
        south_bot.decision_traces.clear()

    half_turns = 0
    max_half_turns = max_turns * 2

    while state.phase == PHASE_BATTLE and half_turns < max_half_turns:
        half_turns += 1
        active_team = state.active_team
        active_bot = north_bot if active_team == TEAM_ENEMY else south_bot
        turn_num = state.turn_number

        trace_start_idx = len(active_bot.decision_traces)
        records = active_bot.run_turn(state)
        new_traces = active_bot.decision_traces[trace_start_idx:]

        if record_log:
            act_dicts: List[Dict[str, Any]] = []
            for r in records:
                act_dicts.append(
                    {
                        "action_type": r.action_type,
                        "unit_id": r.unit_id,
                        "unit_name": r.unit_name,
                        "target_hex": list(r.target_hex) if r.target_hex else None,
                        "new_facing": r.new_facing,
                        "description": r.description,
                    }
                )
            turn_logs.append(
                {
                    "turn_number": turn_num,
                    "active_team": active_team,
                    "bot_id": active_bot.bot_id,
                    "bot_name": active_bot.name,
                    "bot_ply": active_bot.lookahead_ply,
                    "decision_traces": [dt.to_dict() for dt in new_traces],
                    "actions": act_dicts,
                }
            )

    north_hp = sum(
        u.hp for u in state.units.values() if u.is_alive and u.team == TEAM_ENEMY
    )
    south_hp = sum(
        u.hp for u in state.units.values() if u.is_alive and u.team == TEAM_PLAYER
    )
    north_pts = sum(
        u.spec.point_cost
        for u in state.units.values()
        if u.is_alive and u.team == TEAM_ENEMY
    )
    south_pts = sum(
        u.spec.point_cost
        for u in state.units.values()
        if u.is_alive and u.team == TEAM_PLAYER
    )

    if state.phase == PHASE_EVOLUTION or (north_hp == 0 and south_hp > 0):
        winner_team: Optional[str] = TEAM_PLAYER
        winner_bot_id: Optional[str] = south_bot.bot_id
    elif state.phase == PHASE_DEFEAT or (south_hp == 0 and north_hp > 0):
        winner_team = TEAM_ENEMY
        winner_bot_id = north_bot.bot_id
    else:
        # Turn limit reached: adjudicate by remaining army point value + HP
        north_tie_score = north_pts * 12 + north_hp
        south_tie_score = south_pts * 12 + south_hp
        if north_tie_score > south_tie_score + 6:
            winner_team = TEAM_ENEMY
            winner_bot_id = north_bot.bot_id
        elif south_tie_score > north_tie_score + 6:
            winner_team = TEAM_PLAYER
            winner_bot_id = south_bot.bot_id
        else:
            winner_team = None
            winner_bot_id = None

    # Check if this result is an upset worth flagging for diagnostic inspection
    is_upset = False
    upset_reason = ""
    if winner_team == TEAM_ENEMY:
        w_ply, l_ply = north_bot.lookahead_ply, south_bot.lookahead_ply
        w_id, l_id = north_bot.bot_id, south_bot.bot_id
    elif winner_team == TEAM_PLAYER:
        w_ply, l_ply = south_bot.lookahead_ply, north_bot.lookahead_ply
        w_id, l_id = south_bot.bot_id, north_bot.bot_id
    else:
        w_ply, l_ply = 0, 0
        w_id, l_id = "", ""

    if winner_team is not None:
        if w_ply in (1, 3) and l_ply in (2, 4) and l_id != "legacy_classic":
            is_upset = True
            upset_reason = (
                f"Odd-ply bot ({w_id}, Ply {w_ply}) defeated Even-ply bot ({l_id}, Ply {l_ply})"
            )
        elif w_id == "legacy_classic" and l_ply in (2, 4):
            is_upset = True
            upset_reason = (
                f"Classic Legacy Bot defeated Even-ply genetic bot ({l_id}, Ply {l_ply})"
            )

    return BattleReplayLog(
        match_id=match_id,
        match_type=match_type,
        round_number=state.round_number,
        wave_budget=state.wave_budget,
        north_bot_id=north_bot.bot_id,
        north_bot_name=north_bot.name,
        north_ply=north_bot.lookahead_ply,
        south_bot_id=south_bot.bot_id,
        south_bot_name=south_bot.name,
        south_ply=south_bot.lookahead_ply,
        winner_team=winner_team,
        winner_bot_id=winner_bot_id,
        turns_played=state.turn_number,
        north_remaining_hp=north_hp,
        south_remaining_hp=south_hp,
        north_remaining_pts=north_pts,
        south_remaining_pts=south_pts,
        is_upset=is_upset,
        upset_reason=upset_reason,
        initial_units=initial_units,
        turn_logs=turn_logs,
    )


def run_headless_skirmish(
    north_bot: BotControllerType,
    south_bot: BotControllerType,
    round_number: int = 3,
    budget: int = 14,
    max_turns: int = 32,
    seed: int = 42,
    record_log: bool = True,
    match_id: Optional[str] = None,
) -> BattleReplayLog:
    """
    Run a single symmetric Mammal-vs-Mammal skirmish where `north_bot` (`TEAM_ENEMY`)
    and `south_bot` (`TEAM_PLAYER`) each spend `budget` points to buy and deploy a
    Mammal army on their half of the arena and battle to the finish.
    """
    state = rules.GameState(
        arena_radius=4,
        phase=PHASE_SETUP,
        round_number=round_number,
        wave_budget=budget,
        active_team=TEAM_PLAYER,
        turn_number=1,
        enemy_ai_enabled=True,
    )
    rng_north = random.Random(seed)
    rng_south = random.Random(seed + 10007)

    north_bot.purchase_and_deploy_army(state, TEAM_ENEMY, budget, rng=rng_north)
    south_bot.purchase_and_deploy_army(state, TEAM_PLAYER, budget, rng=rng_south)

    state.phase = PHASE_BATTLE
    state.active_team = TEAM_PLAYER
    state.turn_number = 1
    state.ai_acted_unit_ids.clear()

    mid = match_id or f"skirmish_r{round_number}_b{budget}_{north_bot.bot_id}_vs_{south_bot.bot_id}_s{seed}"
    return _execute_headless_battle_loop(
        state,
        north_bot,
        south_bot,
        match_id=mid,
        match_type="mammal_vs_mammal",
        max_turns=max_turns,
        record_log=record_log,
    )


def run_paired_skirmish(
    bot_a: BotControllerType,
    bot_b: BotControllerType,
    round_number: int = 3,
    budget: int = 14,
    max_turns: int = 32,
    seed: int = 42,
    record_log: bool = True,
) -> Tuple[BattleReplayLog, BattleReplayLog]:
    """
    Run two symmetric Mammal-vs-Mammal games swapping North and South sides so
    first-mover advantage cancels out completely.
    """
    log1 = run_headless_skirmish(
        north_bot=bot_a,
        south_bot=bot_b,
        round_number=round_number,
        budget=budget,
        max_turns=max_turns,
        seed=seed,
        record_log=record_log,
        match_id=f"pair1_r{round_number}_b{budget}_{bot_a.bot_id}_N_vs_{bot_b.bot_id}_S",
    )
    log2 = run_headless_skirmish(
        north_bot=bot_b,
        south_bot=bot_a,
        round_number=round_number,
        budget=budget,
        max_turns=max_turns,
        seed=seed,
        record_log=record_log,
        match_id=f"pair2_r{round_number}_b{budget}_{bot_b.bot_id}_N_vs_{bot_a.bot_id}_S",
    )
    return (log1, log2)


def run_benchmark_vs_reptiles(
    mammal_bot: BotControllerType,
    round_number: int = 3,
    budget: int = 14,
    max_turns: int = 32,
    seed: int = 101,
    record_log: bool = False,
) -> BattleReplayLog:
    """
    Run an asymmetric benchmark battle where `mammal_bot` commands a Mammal wave
    (`TEAM_ENEMY`, upper half) against a standard evolved Reptile army (`TEAM_PLAYER`,
    lower half) piloted by `LegacyBotController`.
    """
    rng = random.Random(seed)
    state = rules.GameState(
        arena_radius=4,
        phase=PHASE_SETUP,
        round_number=round_number,
        wave_budget=budget,
        active_team=TEAM_PLAYER,
        turn_number=1,
        enemy_ai_enabled=True,
    )

    # Deploy mammal_bot's wave in the upper half
    mammal_bot.purchase_and_deploy_army(state, TEAM_ENEMY, budget, rng=rng)

    # Build a deterministic Reptile defender roster scaled to `budget`
    reptile_specs: List[CreatureSpec] = []
    rem = budget
    stage_cap = min(8, max(2, round_number + 1))
    idx_counter = 0
    while rem > 0 and len(reptile_specs) < MAX_ARMY_SIZE:
        cls_name = PLAYER_CLASSES[idx_counter % len(PLAYER_CLASSES)]
        idx_counter += 1
        # Find highest stage <= stage_cap that fits within `rem`
        chosen_spec = get_creature_spec(cls_name, 1)
        for stg in range(stage_cap, 0, -1):
            cand = get_creature_spec(cls_name, stg)
            if cand.point_cost <= rem:
                chosen_spec = cand
                break
        reptile_specs.append(chosen_spec)
        rem -= chosen_spec.point_cost

    lower_hexes = rules.get_valid_team_deploy_hexes(state, TEAM_PLAYER)
    lower_hexes.sort(key=lambda c: (abs(2 * c[1] + c[0] - 3), abs(c[0])))
    for idx, (spec, coord) in enumerate(zip(reptile_specs, lower_hexes), start=1):
        uid = f"rep_{idx}_{spec.asset_slug}"
        u = rules.Unit(uid, spec, TEAM_PLAYER, None, FACING_N)
        state.units[uid] = u
        rules.deploy_team_unit_to_hex(state, uid, coord, facing=FACING_N)

    state.phase = PHASE_BATTLE
    state.active_team = TEAM_PLAYER
    state.turn_number = 1
    state.ai_acted_unit_ids.clear()

    reptile_pilot = LegacyBotController(name="Reptile Garrison AI")
    mid = f"bench_reptiles_r{round_number}_b{budget}_{mammal_bot.bot_id}_s{seed}"
    return _execute_headless_battle_loop(
        state,
        north_bot=mammal_bot,
        south_bot=reptile_pilot,
        match_id=mid,
        match_type="mammal_vs_reptiles",
        max_turns=max_turns,
        record_log=record_log,
    )


# ============================================================================
# Genetic Operators (Crossover & Mutation)
# ============================================================================

_BOT_FIRST_NAMES = (
    "Iron", "Shadow", "Crimson", "Frost", "Storm", "Grim", "Apex", "Dread",
    "Obsidian", "Feral", "Blood", "Venom", "Thunder", "Ashen", "Vanguard", "Night",
)
_BOT_LAST_NAMES = (
    "Fang", "Claw", "Maw", "Stalker", "Howler", "Ravager", "Behemoth", "Pouncer",
    "Talon", "Warden", "Predator", "Warlord", "Alpha", "pack", "Hunter", "Reaver",
)


def _generate_bot_name(rng: random.Random, ply: int, gen: int) -> str:
    first = rng.choice(_BOT_FIRST_NAMES)
    last = rng.choice(_BOT_LAST_NAMES).title()
    return f"{first}{last}-G{gen}P{ply}"


def _blend_float(
    a: float,
    b: float,
    rng: random.Random,
    low: float,
    high: float,
    mutation_rate: float = 0.25,
    mutation_scale: float = 0.15,
) -> float:
    weight = rng.uniform(0.2, 0.8) if rng.random() < 0.7 else (1.0 if rng.random() < 0.5 else 0.0)
    val = a * weight + b * (1.0 - weight)
    if rng.random() < mutation_rate:
        span = high - low
        val += rng.gauss(0.0, span * mutation_scale)
    return round(max(low, min(high, val)), 3)


def crossover_and_mutate(
    parent_a: BotGenome,
    parent_b: BotGenome,
    rng: random.Random,
    child_id: str,
    generation: int,
    mutation_rate: float = 0.25,
    allowed_plies: Tuple[int, ...] = (1, 2, 3, 4),
) -> BotGenome:
    """
    Create an offspring `BotGenome` by recombining `parent_a` and `parent_b`
    and applying Gaussian/discrete mutations across setup, mammal, targeting,
    and lookahead genes.
    """
    # 1. Setup genes
    child_weights: Dict[str, float] = {}
    for spec in MAMMAL_LINEUP:
        cid = spec.creature_id
        wa = parent_a.setup.mammal_purchase_weights.get(cid, 1.0)
        wb = parent_b.setup.mammal_purchase_weights.get(cid, 1.0)
        child_weights[cid] = _blend_float(wa, wb, rng, 0.15, 3.2, mutation_rate)

    setup = SetupGenes(
        mammal_purchase_weights=child_weights,
        elite_vs_swarm_bias=_blend_float(
            parent_a.setup.elite_vs_swarm_bias,
            parent_b.setup.elite_vs_swarm_bias,
            rng,
            -1.0,
            1.0,
            mutation_rate,
        ),
        frontline_aggression=_blend_float(
            parent_a.setup.frontline_aggression,
            parent_b.setup.frontline_aggression,
            rng,
            0.1,
            1.0,
            mutation_rate,
        ),
        tank_frontline_weight=_blend_float(
            parent_a.setup.tank_frontline_weight,
            parent_b.setup.tank_frontline_weight,
            rng,
            -0.5,
            2.0,
            mutation_rate,
        ),
        spread_vs_cluster=_blend_float(
            parent_a.setup.spread_vs_cluster,
            parent_b.setup.spread_vs_cluster,
            rng,
            -1.0,
            1.0,
            mutation_rate,
        ),
        wing_inward_facing_bias=_blend_float(
            parent_a.setup.wing_inward_facing_bias,
            parent_b.setup.wing_inward_facing_bias,
            rng,
            0.0,
            1.0,
            mutation_rate,
        ),
    )

    # 2. Per-Mammal tactical genes
    mammal_genes: Dict[str, MammalTacticalGenes] = {}
    for spec in MAMMAL_LINEUP:
        cid = spec.creature_id
        ma = parent_a.get_mammal_genes(cid)
        mb = parent_b.get_mammal_genes(cid)
        mammal_genes[cid] = MammalTacticalGenes(
            aggression_weight=_blend_float(
                ma.aggression_weight, mb.aggression_weight, rng, 12.0, 75.0, mutation_rate
            ),
            flanking_weight=_blend_float(
                ma.flanking_weight, mb.flanking_weight, rng, 0.0, 55.0, mutation_rate
            ),
            self_defense_weight=_blend_float(
                ma.self_defense_weight, mb.self_defense_weight, rng, 2.0, 50.0, mutation_rate
            ),
            pack_cohesion_weight=_blend_float(
                ma.pack_cohesion_weight, mb.pack_cohesion_weight, rng, 0.0, 45.0, mutation_rate
            ),
            cleave_positioning_weight=_blend_float(
                ma.cleave_positioning_weight,
                mb.cleave_positioning_weight,
                rng,
                0.0,
                60.0,
                mutation_rate,
            ),
            hit_and_run_weight=_blend_float(
                ma.hit_and_run_weight, mb.hit_and_run_weight, rng, 0.0, 50.0, mutation_rate
            ),
            fly_over_readiness=_blend_float(
                ma.fly_over_readiness, mb.fly_over_readiness, rng, -15.0, 40.0, mutation_rate
            ),
        )

    # 3. Targeting genes
    targeting = TargetingGenes(
        strong_vs_weak_bias=_blend_float(
            parent_a.targeting.strong_vs_weak_bias,
            parent_b.targeting.strong_vs_weak_bias,
            rng,
            -1.0,
            1.0,
            mutation_rate,
        ),
        hybrid_kill_confirm_weight=_blend_float(
            parent_a.targeting.hybrid_kill_confirm_weight,
            parent_b.targeting.hybrid_kill_confirm_weight,
            rng,
            0.2,
            2.5,
            mutation_rate,
        ),
        vulnerable_arc_bonus_weight=_blend_float(
            parent_a.targeting.vulnerable_arc_bonus_weight,
            parent_b.targeting.vulnerable_arc_bonus_weight,
            rng,
            0.2,
            2.5,
            mutation_rate,
        ),
        thorns_caution_weight=_blend_float(
            parent_a.targeting.thorns_caution_weight,
            parent_b.targeting.thorns_caution_weight,
            rng,
            0.1,
            2.5,
            mutation_rate,
        ),
        focus_fire_weight=_blend_float(
            parent_a.targeting.focus_fire_weight,
            parent_b.targeting.focus_fire_weight,
            rng,
            0.2,
            2.5,
            mutation_rate,
        ),
    )

    # 4. Lookahead genes
    chosen_ply = (
        parent_a.lookahead.lookahead_ply
        if rng.random() < 0.55
        else parent_b.lookahead.lookahead_ply
    )
    if rng.random() < mutation_rate * 0.65:
        chosen_ply = rng.choice(allowed_plies)

    chosen_beam = (
        parent_a.lookahead.beam_width
        if rng.random() < 0.5
        else parent_b.lookahead.beam_width
    )
    if rng.random() < mutation_rate * 0.5:
        chosen_beam = rng.choice((2, 3, 4))

    lookahead = LookaheadGenes(
        lookahead_ply=chosen_ply,
        beam_width=chosen_beam,
        enemy_response_weight=_blend_float(
            parent_a.lookahead.enemy_response_weight,
            parent_b.lookahead.enemy_response_weight,
            rng,
            0.45,
            1.35,
            mutation_rate,
        ),
        followup_discount=_blend_float(
            parent_a.lookahead.followup_discount,
            parent_b.lookahead.followup_discount,
            rng,
            0.35,
            0.90,
            mutation_rate,
        ),
    )

    # 5. Global coordination genes
    coordination = GlobalCoordinationGenes(
        activation_order_bias=_blend_float(
            parent_a.coordination.activation_order_bias,
            parent_b.coordination.activation_order_bias,
            rng,
            -1.0,
            1.0,
            mutation_rate,
        ),
        wounded_retreat_threshold=_blend_float(
            parent_a.coordination.wounded_retreat_threshold,
            parent_b.coordination.wounded_retreat_threshold,
            rng,
            0.05,
            0.48,
            mutation_rate,
        ),
    )

    name = _generate_bot_name(rng, lookahead.lookahead_ply, generation)
    return BotGenome(
        bot_id=child_id,
        name=name,
        generation=generation,
        setup=setup,
        mammal_genes=mammal_genes,
        targeting=targeting,
        lookahead=lookahead,
        coordination=coordination,
    )


def mutate_champion(
    champion: BotGenome,
    rng: random.Random,
    child_id: str,
    generation: int,
    mutation_rate: float = 0.20,
) -> BotGenome:
    """
    Create a direct single-parent mutated offspring of an existing `champion`
    genome (inheriting 100% of the champion's unmutated genes and applying
    fine-tuning Gaussian perturbations to ~20% of genes).
    """
    mutated = crossover_and_mutate(
        parent_a=champion,
        parent_b=champion,
        rng=rng,
        child_id=child_id,
        generation=generation,
        mutation_rate=mutation_rate,
        allowed_plies=(champion.lookahead.lookahead_ply, 2, 4),
    )
    base_prefix = champion.name.split("-")[0]
    mutated.name = f"{base_prefix}-M{generation}P{mutated.lookahead.lookahead_ply}"
    return mutated


# ============================================================================
# Persisted Battle Bot Stable (`BotStable` & `bot_stable.json`)
# ============================================================================

DEFAULT_STABLE_FILENAME = "bot_stable.json"


def create_default_champion_genomes() -> List[BotGenome]:
    """
    Return a set of strong, well-calibrated 2-ply and 4-ply champion genomes
    used to seed `bot_stable.json` before or alongside evolutionary runs.
    """
    champ_2ply = BotGenome(
        bot_id="champ_ironfang_p2",
        name="IronFang-G1P2",
        generation=1,
        fitness=185.0,
        elo_rating=1320.0,
        setup=SetupGenes(
            elite_vs_swarm_bias=0.25,
            frontline_aggression=0.80,
            tank_frontline_weight=0.95,
            spread_vs_cluster=-0.25,
            wing_inward_facing_bias=0.65,
        ),
        targeting=TargetingGenes(
            strong_vs_weak_bias=0.25,
            hybrid_kill_confirm_weight=1.55,
            vulnerable_arc_bonus_weight=1.25,
            thorns_caution_weight=1.15,
            focus_fire_weight=1.40,
        ),
        lookahead=LookaheadGenes(
            lookahead_ply=2,
            beam_width=3,
            enemy_response_weight=0.78,
            followup_discount=0.65,
        ),
        coordination=GlobalCoordinationGenes(
            activation_order_bias=0.20,
            wounded_retreat_threshold=0.25,
        ),
    )

    champ_4ply = BotGenome(
        bot_id="champ_dreadmaw_p4",
        name="DreadMaw-G1P4",
        generation=1,
        fitness=192.0,
        elo_rating=1340.0,
        setup=SetupGenes(
            elite_vs_swarm_bias=0.35,
            frontline_aggression=0.75,
            tank_frontline_weight=1.05,
            spread_vs_cluster=-0.15,
            wing_inward_facing_bias=0.70,
        ),
        targeting=TargetingGenes(
            strong_vs_weak_bias=0.30,
            hybrid_kill_confirm_weight=1.65,
            vulnerable_arc_bonus_weight=1.30,
            thorns_caution_weight=1.20,
            focus_fire_weight=1.45,
        ),
        lookahead=LookaheadGenes(
            lookahead_ply=4,
            beam_width=3,
            enemy_response_weight=0.80,
            followup_discount=0.62,
        ),
        coordination=GlobalCoordinationGenes(
            activation_order_bias=0.25,
            wounded_retreat_threshold=0.28,
        ),
    )

    champ_pack_2ply = BotGenome(
        bot_id="champ_stormpack_p2",
        name="StormPack-G1P2",
        generation=1,
        fitness=178.0,
        elo_rating=1305.0,
        setup=SetupGenes(
            elite_vs_swarm_bias=-0.15,
            frontline_aggression=0.85,
            tank_frontline_weight=0.80,
            spread_vs_cluster=-0.45,
            wing_inward_facing_bias=0.55,
        ),
        targeting=TargetingGenes(
            strong_vs_weak_bias=-0.20,
            hybrid_kill_confirm_weight=1.70,
            vulnerable_arc_bonus_weight=1.35,
            thorns_caution_weight=1.05,
            focus_fire_weight=1.50,
        ),
        lookahead=LookaheadGenes(
            lookahead_ply=2,
            beam_width=3,
            enemy_response_weight=0.75,
            followup_discount=0.65,
        ),
        coordination=GlobalCoordinationGenes(
            activation_order_bias=0.10,
            wounded_retreat_threshold=0.22,
        ),
    )
    return [champ_4ply, champ_2ply, champ_pack_2ply]


@dataclass
class BotStable:
    """
    Persisted Hall-of-Fame stable of the strongest evolved `BotGenome` controllers.
    Allows evolutionary runs to resume from previous champions and supplies
    elite bots for `GAME_MODE_ADVANCED`.
    """
    max_size: int = 12
    total_generations_run: int = 0
    bots: List[BotGenome] = field(default_factory=list)

    def upsert_bot(self, genome: BotGenome) -> None:
        """Insert or update `genome` in the stable and keep top `max_size` by fitness."""
        existing_idx = next(
            (i for i, b in enumerate(self.bots) if b.bot_id == genome.bot_id), None
        )
        copy_g = BotGenome.from_dict(genome.to_dict())
        if existing_idx is not None:
            self.bots[existing_idx] = copy_g
        else:
            self.bots.append(copy_g)
        self.bots.sort(key=lambda b: (b.fitness, b.elo_rating), reverse=True)
        if len(self.bots) > self.max_size:
            self.bots = self.bots[: self.max_size]

    def get_top_bot(self, rank: int = 0) -> BotGenome:
        if not self.bots:
            self.bots = create_default_champion_genomes()
        idx = max(0, min(len(self.bots) - 1, rank))
        return self.bots[idx]

    def pick_advanced_bot(
        self, rng: Optional[random.Random] = None, top_k: int = 3
    ) -> BotGenome:
        """Pick one of the top `top_k` evolved bots in the stable for Advanced mode."""
        if not self.bots:
            self.bots = create_default_champion_genomes()
        pool = self.bots[: max(1, min(top_k, len(self.bots)))]
        if rng is None:
            return pool[0]
        return rng.choice(pool)

    def save(self, filepath: str = DEFAULT_STABLE_FILENAME) -> None:
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "max_size": self.max_size,
            "total_generations_run": self.total_generations_run,
            "bots": [b.to_dict() for b in self.bots],
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, filepath: str = DEFAULT_STABLE_FILENAME) -> BotStable:
        if not os.path.exists(filepath):
            return cls(max_size=12, total_generations_run=0, bots=create_default_champion_genomes())
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        loaded_bots = [
            BotGenome.from_dict(bdict) for bdict in data.get("bots", [])
        ]
        if not loaded_bots:
            loaded_bots = create_default_champion_genomes()
        return cls(
            max_size=int(data.get("max_size", 12)),
            total_generations_run=int(data.get("total_generations_run", 0)),
            bots=loaded_bots,
        )


# ============================================================================
# Evolutionary Tournament Runner & Ply Telemetry Matrix
# ============================================================================

@dataclass
class PlyTelemetryMatrix:
    """Tracks head-to-head win/loss/draw counts by lookahead ply (`1..4` and `0` for Classic)."""
    wins: Dict[Tuple[int, int], int] = field(default_factory=dict)
    games: Dict[Tuple[int, int], int] = field(default_factory=dict)
    decisive_games: int = 0
    turn_limit_games: int = 0
    upset_logs: List[BattleReplayLog] = field(default_factory=list)

    def record_log(self, log: BattleReplayLog, north_is_classic: bool, south_is_classic: bool) -> None:
        if log.north_remaining_hp == 0 or log.south_remaining_hp == 0:
            self.decisive_games += 1
        else:
            self.turn_limit_games += 1

        p_n = 0 if north_is_classic else log.north_ply
        p_s = 0 if south_is_classic else log.south_ply
        if p_n == p_s:
            return
        pair_ns = (p_n, p_s)
        pair_sn = (p_s, p_n)
        self.games[pair_ns] = self.games.get(pair_ns, 0) + 1
        self.games[pair_sn] = self.games.get(pair_sn, 0) + 1
        if log.winner_team == TEAM_ENEMY:
            self.wins[pair_ns] = self.wins.get(pair_ns, 0) + 1
        elif log.winner_team == TEAM_PLAYER:
            self.wins[pair_sn] = self.wins.get(pair_sn, 0) + 1

        if log.is_upset:
            self.upset_logs.append(log)

    def format_summary(self) -> str:
        labels = {0: "Classic", 1: "1-Ply", 2: "2-Ply", 3: "3-Ply", 4: "4-Ply"}
        tot_g = max(1, self.decisive_games + self.turn_limit_games)
        dec_pct = 100.0 * self.decisive_games / tot_g
        lines = [
            f"PLY-VS-PLY WIN RATE TELEMETRY  "
            f"(Decisive Wipeouts: {self.decisive_games}/{tot_g} [{dec_pct:.0f}%], "
            f"Turn-Limit: {self.turn_limit_games}):"
        ]
        for p_a in (1, 2, 3, 4):
            row_parts: List[str] = []
            for p_b in (0, 1, 2, 3, 4):
                if p_a == p_b:
                    continue
                g = self.games.get((p_a, p_b), 0)
                w = self.wins.get((p_a, p_b), 0)
                if g > 0:
                    row_parts.append(f"vs {labels[p_b]}: {w}/{g} ({100.0 * w / g:.0f}%)")
            if row_parts:
                lines.append(f"  {labels[p_a]:7s} ->  " + "  |  ".join(row_parts))
        return "\n".join(lines)


def format_genome_dump(genome: BotGenome) -> str:
    """
    Format a comprehensive human-readable gene dump for a `BotGenome`, including
    its lookahead parameters, army setup & formation bias, top preferred Mammal
    purchases, sample deterministic wave rosters, targeting philosophy, and
    species-specific tactical weights.
    """
    setup = genome.setup
    t_genes = genome.targeting
    l_genes = genome.lookahead
    c_genes = genome.coordination
    ctrl = GeneticBotController(genome)

    wr_pct = (
        100.0 * genome.wins_vs_classic / max(1, genome.games_vs_classic)
        if genome.games_vs_classic > 0
        else 0.0
    )

    if setup.elite_vs_swarm_bias > 0.25:
        roster_style = "Elite Heavy Hitters"
    elif setup.elite_vs_swarm_bias < -0.25:
        roster_style = "Swarm / Pack Numbers"
    else:
        roster_style = "Balanced Combined Arms"

    if t_genes.strong_vs_weak_bias > 0.25:
        target_style = "Focus Strong Threats First (+Kill-Confirm)"
    elif t_genes.strong_vs_weak_bias < -0.25:
        target_style = "Clear Weak Units First (+Kill-Confirm)"
    else:
        target_style = "Hybrid Opportunistic (Strong + Weak Kill-Confirm)"

    # Rank mammals by effective purchase affinity
    cost_exp = 1.15 + 0.70 * setup.elite_vs_swarm_bias
    ranked_mammals = sorted(
        MAMMAL_LINEUP,
        key=lambda s: setup.mammal_purchase_weights.get(s.creature_id, 1.0),
        reverse=True,
    )
    top_pref_strs = [
        f"{s.name} ({s.point_cost}pt, w={setup.mammal_purchase_weights.get(s.creature_id, 1.0):.2f})"
        for s in ranked_mammals[:5]
    ]

    wave_r2 = [s.name for s in ctrl.purchase_wave_specs(round_number=2, budget=8)]
    wave_r3 = [s.name for s in ctrl.purchase_wave_specs(round_number=3, budget=14)]
    wave_r6 = [s.name for s in ctrl.purchase_wave_specs(round_number=6, budget=22)]

    lines = [
        "=" * 86,
        f"CHAMPION BOT GENE DUMP: {genome.name} [{genome.bot_id}]",
        "-" * 86,
        f"  Generation         : {genome.generation}  |  Championship Streak: {genome.championship_streak} gen(s)",
        f"  Tournament Fitness : {genome.fitness:.1f}  |  Elo Rating: {genome.elo_rating:.1f}  |  "
        f"vs Classic: {genome.wins_vs_classic}/{genome.games_vs_classic} ({wr_pct:.0f}%)",
        "-" * 86,
        "  1. X-PLY LOOKAHEAD GENES:",
        f"     • Lookahead Depth       : {l_genes.lookahead_ply}-Ply  (Beam Width: {l_genes.beam_width})",
        f"     • Enemy Response Weight : {l_genes.enemy_response_weight:.3f}  |  "
        f"Follow-Up Discount: {l_genes.followup_discount:.3f}",
        "  2. ARMY SETUP & FORMATION GENES:",
        f"     • Roster Philosophy     : {roster_style} (elite_vs_swarm_bias = {setup.elite_vs_swarm_bias:+.3f})",
        f"     • Frontline Aggression  : {setup.frontline_aggression:.3f}  |  "
        f"Tank Frontline Weight: {setup.tank_frontline_weight:+.3f}",
        f"     • Spread vs. Cluster    : {setup.spread_vs_cluster:+.3f} "
        f"({'Wide Flank Wings' if setup.spread_vs_cluster >= 0 else 'Tight Pack Cluster'})  |  "
        f"Wing Inward Facing: {setup.wing_inward_facing_bias:.3f}",
        f"     • Top Preferred Mammals : {', '.join(top_pref_strs)}",
        f"     • Deterministic Wave @  8 pts (R2): {', '.join(wave_r2)}",
        f"     • Deterministic Wave @ 14 pts (R3): {', '.join(wave_r3)}",
        f"     • Deterministic Wave @ 22 pts (R6): {', '.join(wave_r6)}",
        "  3. TARGETING & COORDINATION GENES:",
        f"     • Target Priority Style : {target_style} (strong_vs_weak_bias = {t_genes.strong_vs_weak_bias:+.3f})",
        f"     • Hybrid Kill-Confirm   : {t_genes.hybrid_kill_confirm_weight:.3f}  |  "
        f"Focus-Fire Weight: {t_genes.focus_fire_weight:.3f}",
        f"     • Vulnerable Arc Bonus  : {t_genes.vulnerable_arc_bonus_weight:.3f}  |  "
        f"Thorns Caution: {t_genes.thorns_caution_weight:.3f}",
        f"     • Activation Order Bias : {c_genes.activation_order_bias:+.3f}  |  "
        f"Wounded Retreat Threshold: {c_genes.wounded_retreat_threshold:.0%}",
        "  4. SPECIES TACTICAL GENES (Top 6 Preferred Mammals):",
        "     Species           | Aggr  | Flank | SelfDef | Pack  | Cleave | Hit&Run | FlyOver",
        "     ------------------+-------+-------+---------+-------+--------+---------+--------",
    ]
    for s in ranked_mammals[:6]:
        mg = genome.get_mammal_genes(s.creature_id)
        lines.append(
            f"     {s.name:17s} | {mg.aggression_weight:5.1f} | {mg.flanking_weight:5.1f} | "
            f"{mg.self_defense_weight:7.1f} | {mg.pack_cohesion_weight:5.1f} | "
            f"{mg.cleave_positioning_weight:6.1f} | {mg.hit_and_run_weight:7.1f} | {mg.fly_over_readiness:6.1f}"
        )
    lines.append("=" * 86)
    return "\n".join(lines)


def _score_log_for_bot(log: BattleReplayLog, bot_id: str) -> float:
    """Compute tournament fitness points earned by `bot_id` in `log`."""
    is_north = log.north_bot_id == bot_id
    my_hp = log.north_remaining_hp if is_north else log.south_remaining_hp
    opp_hp = log.south_remaining_hp if is_north else log.north_remaining_hp
    my_pts = log.north_remaining_pts if is_north else log.south_remaining_pts
    opp_pts = log.south_remaining_pts if is_north else log.north_remaining_pts

    score = (my_pts - opp_pts) * 2.5 + (my_hp - opp_hp) * 0.45
    if log.winner_bot_id == bot_id:
        score += 35.0 + max(0, 36 - log.turns_played) * 0.5
    elif log.winner_bot_id is None:
        score += 10.0
    return score


def run_evolution(
    generations: int = 3,
    population_size: int = 8,
    random_injection_count: int = 2,
    stable_path: str = DEFAULT_STABLE_FILENAME,
    log_dir: Optional[str] = "battle_logs",
    record_upsets: bool = True,
    record_all_games: bool = False,
    max_turns: int = 28,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[BotStable, PlyTelemetryMatrix]:
    """
    Run `generations` rounds of UI-free evolutionary tournament combat:
      1. Loads existing champions from `stable_path` so evolution automatically
         continues improving the current bots in the stable across invocations.
      2. Each generation evaluates all bots on an identical, apples-to-apples
         gauntlet (shared seeds & opponents) so true champions retain their crown
         across multiple rounds until genuinely dethroned:
         - Paired North/South-swapped head-to-head matches vs. the Classic Legacy Bot.
         - Asymmetric Reptile Army benchmark battles.
         - Paired North/South-swapped head-to-head matches against the reigning
           Top 3 Benchmark Seeds in the population.
      3. Logs ply-vs-ply performance and saves any upset/anomaly games to `log_dir`.
      4. Updates and saves `BotStable` (`stable_path`), selects top elites, breeds
         mutated offspring, and injects `random_injection_count` fresh random-gene bots.
    """
    has_existing_stable = os.path.exists(stable_path)
    stable = BotStable.load(stable_path)
    effective_seed = (
        seed
        if seed is not None
        else random.SystemRandom().randint(100_000, 999_999_999)
    )
    rng = random.Random(effective_seed)
    telemetry = PlyTelemetryMatrix()
    classic_bot = LegacyBotController()

    population_size = max(4, population_size)
    inject_cnt = max(0, min(random_injection_count, population_size - 2))
    start_gen = stable.total_generations_run
    seq = start_gen * 100 + 1

    def _breed_population_from_parents(
        source_bots: List[BotGenome], target_gen: int
    ) -> Tuple[List[BotGenome], int, int, int]:
        """
        Build a generation of `population_size` bots comprising:
          1. Unmodified Top Elites (preserving reigning champions)
          2. Direct Single-Parent Mutations of Existing Champions (`mutate_champion`)
          3. Two-Parent Crossover Offspring (`crossover_and_mutate`)
          4. Fresh Random-Gene Injections (`create_random_genome`)
        """
        nonlocal seq
        elite_cnt = max(2, min(len(source_bots), (population_size - inject_cnt) // 2))
        elites = [BotGenome.from_dict(b.to_dict()) for b in source_bots[:elite_cnt]]
        new_pop: List[BotGenome] = list(elites)

        breed_target = population_size - inject_cnt
        slots_for_offspring = max(0, breed_target - len(new_pop))
        # At least half of bred slots are direct single-parent mutations of existing champions
        mut_target = max(1, (slots_for_offspring + 1) // 2) if slots_for_offspring > 0 else 0

        mut_created = 0
        while mut_created < mut_target and len(new_pop) < breed_target:
            champ_parent = source_bots[mut_created % min(len(source_bots), max(1, elite_cnt))]
            mut_id = f"g{target_gen}_mut_{seq}"
            seq += 1
            new_pop.append(
                mutate_champion(champ_parent, rng=rng, child_id=mut_id, generation=target_gen)
            )
            mut_created += 1

        xover_created = 0
        while len(new_pop) < breed_target:
            pa = rng.choice(source_bots[: max(2, len(source_bots) // 2)])
            pb_pool = [
                b for b in source_bots[: max(3, (len(source_bots) * 2) // 3)] if b.bot_id != pa.bot_id
            ]
            pb = rng.choice(pb_pool) if pb_pool else pa
            child_id = f"g{target_gen}_xover_{seq}"
            seq += 1
            new_pop.append(
                crossover_and_mutate(pa, pb, rng=rng, child_id=child_id, generation=target_gen)
            )
            xover_created += 1

        rand_created = 0
        while len(new_pop) < population_size:
            r_ply = ((seq - 1) % 4) + 1
            rand_id = f"g{target_gen}_rand_{seq}_p{r_ply}"
            seq += 1
            new_pop.append(
                create_random_genome(
                    rng=rng,
                    bot_id=rand_id,
                    name=_generate_bot_name(rng, r_ply, target_gen),
                    generation=target_gen,
                    allowed_plies=(r_ply,),
                )
            )
            rand_created += 1

        return (new_pop, len(elites), mut_created + xover_created, rand_created)

    population, n_elites, n_bred, n_rand = _breed_population_from_parents(
        stable.bots, start_gen + 1
    )

    if verbose:
        print("=" * 86)
        if has_existing_stable and start_gen > 0:
            print(
                f"RESUMING EVOLUTION FROM '{stable_path}' "
                f"(Generations {start_gen + 1}..{start_gen + generations} | "
                f"{n_elites} Champs + {n_bred} Mutated/Xover + {n_rand} Random | Seed {effective_seed})"
            )
        else:
            print(
                f"STARTING NEW EVOLUTION RUN -> '{stable_path}' "
                f"(Generations 1..{generations} | "
                f"{n_elites} Seed Champs + {n_bred} Mutated/Xover + {n_rand} Random | Seed {effective_seed})"
            )

    saved_log_count = 0
    reigning_champ_id: Optional[str] = population[0].bot_id if population else None

    for gen_offset in range(1, generations + 1):
        gen_num = stable.total_generations_run + 1
        stable.total_generations_run = gen_num
        fitness_map: Dict[str, float] = {b.bot_id: 0.0 for b in population}

        # Scenarios: early/mid/late budgets with IDENTICAL benchmark seeds for all bots
        budgets = [(3, 14), (7, 22)]

        for genome in population:
            ctrl = GeneticBotController(genome, record_telemetry=True)

            # A. Paired head-to-head matches vs. Classic Legacy Bot (shared seed across all bots!)
            for r_num, b_pts in budgets:
                match_seed = effective_seed + gen_num * 1000 + b_pts
                log1, log2 = run_paired_skirmish(
                    ctrl,
                    classic_bot,
                    round_number=r_num,
                    budget=b_pts,
                    max_turns=max_turns,
                    seed=match_seed,
                    record_log=(record_upsets or record_all_games),
                )
                for lg, n_cls, s_cls in ((log1, False, True), (log2, True, False)):
                    fitness_map[genome.bot_id] += _score_log_for_bot(lg, genome.bot_id) * 1.25
                    genome.games_vs_classic += 1
                    if lg.winner_bot_id == genome.bot_id:
                        genome.wins_vs_classic += 1
                    telemetry.record_log(lg, north_is_classic=n_cls, south_is_classic=s_cls)
                    if log_dir and (record_upsets and lg.is_upset or record_all_games):
                        if saved_log_count < 40:
                            lg.save_json(os.path.join(log_dir, f"gen{gen_num}_{lg.match_id}.json"))
                            saved_log_count += 1

            # B. Asymmetric Benchmark vs. Reptile Army (shared seed across all bots!)
            rep_log = run_benchmark_vs_reptiles(
                ctrl,
                round_number=5,
                budget=18,
                max_turns=max_turns,
                seed=effective_seed + gen_num * 500,
                record_log=record_all_games,
            )
            fitness_map[genome.bot_id] += _score_log_for_bot(rep_log, genome.bot_id)

        # C. Equal-Strength Peer Gauntlet: every bot faces 2 top benchmark peers
        # (Seed #0 and Seed #1; if the bot IS Seed #0 or #1, it faces the other + Seed #2)
        p_seed = effective_seed + gen_num * 2000
        pair_cache: Dict[Tuple[str, str], Tuple[BattleReplayLog, BattleReplayLog]] = {}

        def _get_paired_result(ga: BotGenome, gb: BotGenome) -> Tuple[BattleReplayLog, BattleReplayLog]:
            key = (ga.bot_id, gb.bot_id)
            rev_key = (gb.bot_id, ga.bot_id)
            if key in pair_cache:
                return pair_cache[key]
            if rev_key in pair_cache:
                r1, r2 = pair_cache[rev_key]
                return (r2, r1)
            ctrl_a = GeneticBotController(ga, record_telemetry=True)
            ctrl_b = GeneticBotController(gb, record_telemetry=True)
            res = run_paired_skirmish(
                ctrl_a,
                ctrl_b,
                round_number=6,
                budget=18,
                max_turns=max_turns,
                seed=p_seed,
                record_log=(record_upsets or record_all_games),
            )
            pair_cache[key] = res
            for lg in res:
                telemetry.record_log(lg, north_is_classic=False, south_is_classic=False)
                if log_dir and (record_all_games or (record_upsets and lg.is_upset)):
                    nonlocal saved_log_count
                    if saved_log_count < 40:
                        lg.save_json(os.path.join(log_dir, f"gen{gen_num}_{lg.match_id}.json"))
                        saved_log_count += 1
            return res

        for idx_a, ga in enumerate(population):
            opp_indices = [i for i in (0, 1, 2) if i < len(population) and i != idx_a][:2]
            for idx_b in opp_indices:
                gb = population[idx_b]
                log1, log2 = _get_paired_result(ga, gb)
                for lg in (log1, log2):
                    fitness_map[ga.bot_id] += _score_log_for_bot(lg, ga.bot_id)

        # Update fitness on a uniform scale (no veteran damping penalty!)
        for g in population:
            new_fit = fitness_map[g.bot_id]
            if g.fitness > 0.0:
                # Keep the higher of its rolling average or its current gauntlet performance
                # plus a small consistency bonus for proven multi-generation veterans
                blended = 0.45 * g.fitness + 0.55 * new_fit
                g.fitness = round(max(new_fit, blended) + min(g.championship_streak * 4.0, 20.0), 2)
            else:
                g.fitness = round(new_fit, 2)
            win_rate_cls = (
                g.wins_vs_classic / float(g.games_vs_classic)
                if g.games_vs_classic > 0
                else 0.5
            )
            g.elo_rating = round(1200.0 + g.fitness * 0.65 + (win_rate_cls - 0.5) * 240.0, 1)

        population.sort(key=lambda b: (b.fitness, b.wins_vs_classic, b.elo_rating), reverse=True)

        # Track championship streak for the #1 bot
        top_now = population[0]
        if reigning_champ_id is not None and top_now.bot_id == reigning_champ_id:
            top_now.championship_streak += 1
        else:
            top_now.championship_streak = max(1, top_now.championship_streak)
            reigning_champ_id = top_now.bot_id

        for g in population:
            if g.bot_id != top_now.bot_id and g.championship_streak == 0:
                pass
            stable.upsert_bot(g)

        stable.save(stable_path)

        if verbose:
            print("=" * 86)
            print(
                f"GENERATION {gen_num} SUMMARY  (Top {min(5, len(population))}/{len(population)} Bots)"
            )
            print("-" * 86)
            for rank_i, b in enumerate(population[:5], start=1):
                wr = (
                    f"{b.wins_vs_classic}/{b.games_vs_classic} "
                    f"({100.0 * b.wins_vs_classic / max(1, b.games_vs_classic):.0f}%)"
                )
                streak_str = f" [Streak:{b.championship_streak}G]" if b.championship_streak > 0 else ""
                print(
                    f"  #{rank_i} {b.name:20s} [{b.bot_id:18s}] | Ply {b.lookahead.lookahead_ply} | "
                    f"Fitness: {b.fitness:7.1f} | Elo: {b.elo_rating:6.1f} | vs Classic: {wr}{streak_str}"
                )
            print("-" * 86)
            print(telemetry.format_summary())

        # Breed next generation if more generations remain
        if gen_offset < generations:
            population, _, _, _ = _breed_population_from_parents(population, gen_num + 1)

    return (stable, telemetry)


# ============================================================================
# CLI Entrypoint (`python bots.py ...`)
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Denizens — Headless Genetic Bot Evolution Arena & Replay Inspector"
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=4, help="Number of evolutionary generations to run"
    )
    parser.add_argument(
        "--population", "-p", type=int, default=8, help="Bot population size per generation"
    )
    parser.add_argument(
        "--inject", "-i", type=int, default=2, help="Number of random-gene bots injected per generation"
    )
    parser.add_argument(
        "--stable", type=str, default=DEFAULT_STABLE_FILENAME, help="Path to persisted bot stable JSON"
    )
    parser.add_argument(
        "--reset-stable",
        action="store_true",
        default=False,
        help="Delete existing stable file before starting a fresh evolution run",
    )
    parser.add_argument(
        "--list-stable",
        action="store_true",
        default=False,
        help="List all evolved bots currently persisted in bot_stable.json and exit",
    )
    parser.add_argument(
        "--log-dir", type=str, default="battle_logs", help="Directory to store recorded battle logs"
    )
    parser.add_argument(
        "--record-upsets",
        action="store_true",
        default=True,
        help="Automatically save game logs when an odd-ply or Classic bot defeats an even-ply bot",
    )
    parser.add_argument(
        "--record-all",
        action="store_true",
        default=False,
        help="Save JSON replay logs for every tournament battle",
    )
    parser.add_argument(
        "--inspect-log",
        type=str,
        default=None,
        help="Path to a recorded battle JSON log to format and print for debugging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (defaults to auto-incrementing per generation)",
    )
    args = parser.parse_args()

    if args.inspect_log:
        replay = BattleReplayLog.load_json(args.inspect_log)
        print(replay.format_human_readable())
        return

    if args.list_stable:
        stable = BotStable.load(args.stable)
        print("=" * 86)
        print(
            f"PERSISTED BOT STABLE ('{args.stable}')  •  "
            f"Total Generations Run: {stable.total_generations_run}  •  Bots: {len(stable.bots)}"
        )
        print("-" * 86)
        for idx, b in enumerate(stable.bots, start=1):
            wr = (
                f"{b.wins_vs_classic}/{b.games_vs_classic} "
                f"({100.0 * b.wins_vs_classic / max(1, b.games_vs_classic):.0f}%)"
            )
            streak_str = f" [Streak:{b.championship_streak}G]" if b.championship_streak > 0 else ""
            print(
                f"  #{idx:2d} {b.name:20s} [{b.bot_id:18s}] | Gen {b.generation:2d} | "
                f"Ply {b.lookahead.lookahead_ply} | Fitness: {b.fitness:7.1f} | "
                f"Elo: {b.elo_rating:6.1f} | vs Classic: {wr}{streak_str}"
            )
        if stable.bots:
            print(format_genome_dump(stable.get_top_bot(0)))
        else:
            print("=" * 86)
        return

    if args.reset_stable and os.path.exists(args.stable):
        os.remove(args.stable)

    t0 = time.time()
    stable, telemetry = run_evolution(
        generations=args.generations,
        population_size=args.population,
        random_injection_count=args.inject,
        stable_path=args.stable,
        log_dir=args.log_dir,
        record_upsets=args.record_upsets,
        record_all_games=args.record_all,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0
    top = stable.get_top_bot(0)
    print(format_genome_dump(top))
    print(
        f"Evolution Complete in {elapsed:.1f}s  •  Persisted {len(stable.bots)} bots to '{args.stable}'"
    )
    print(
        f"Current #1 Stable Champion: {top.name} [{top.bot_id}] "
        f"(Ply {top.lookahead.lookahead_ply}, Streak {top.championship_streak}G, "
        f"Fitness {top.fitness:.1f}, Elo {top.elo_rating:.1f})"
    )
    if telemetry.upset_logs:
        print(
            f"Recorded {len(telemetry.upset_logs)} upset game log(s) to '{args.log_dir}/' "
            f"(inspect with: .venv/bin/python bots.py --inspect-log <path>)"
        )
    print("=" * 86)


if __name__ == "__main__":
    main()

