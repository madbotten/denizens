"""
Genetic AI Bot Controllers, Genome Definitions, X-Ply Lookahead Search,
and Decision Telemetry for Denizens.

All bot setup placements, rotations, movements, juncture attacks, and turn transitions
execute strictly through shared functions in `rules.py`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from constants import (
    ABS_FACING_NAMES,
    CLASS_MAMMAL,
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
    REAR,
    TEAM_ENEMY,
    TEAM_PLAYER,
)
from creatures import CREATURE_BY_ID, MAMMAL_LINEUP, CreatureSpec
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
        Spend `budget` points to purchase up to `max_units` Mammal specs using
        `genome.setup.mammal_purchase_weights` and `genome.setup.elite_vs_swarm_bias`.
        """
        if rng is None:
            rng = random.Random()

        setup = self.genome.setup
        remaining = max(1, budget)
        purchased: List[CreatureSpec] = []

        if round_number <= 1:
            max_single_cost = 1
        else:
            max_single_cost = min(budget, 1 + round_number * 3)

        while remaining > 0 and len(purchased) < max_units:
            slots_left = max_units - len(purchased)
            min_desired_cost = max(1, remaining // slots_left)

            affordable = [
                spec
                for spec in MAMMAL_LINEUP
                if spec.point_cost <= remaining and spec.point_cost <= max_single_cost
            ]
            if not affordable:
                break

            preferred = [s for s in affordable if s.point_cost >= min_desired_cost]
            pool = preferred if preferred else [max(affordable, key=lambda s: s.point_cost)]

            # Exponent combines baseline 1.15 with elite_vs_swarm_bias (-1..+1 -> 0.45..1.85)
            cost_exp = 1.15 + 0.70 * setup.elite_vs_swarm_bias
            weights: List[float] = []
            for spec in pool:
                species_w = max(0.05, setup.mammal_purchase_weights.get(spec.creature_id, 1.0))
                w = species_w * (float(spec.point_cost) ** cost_exp)
                weights.append(max(0.01, w))

            chosen = rng.choices(pool, weights=weights, k=1)[0]
            purchased.append(chosen)
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

                    # Small deterministic jitter from rng for variety on tied hexes
                    h_score += rng.uniform(-0.15, 0.15)
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
            is_kill = dmg >= dfd.hp

            # Base damage reward
            sub_score = effective_dmg * 28.0 + (dmg - effective_dmg) * 4.0

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

            # "Both" hybrid strategy: massive reward for confirming a kill to remove an enemy action
            if is_kill:
                kill_bonus = (260.0 + dfd.spec.point_cost * 18.0 + dfd.spec.attack * 10.0)
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
            score = -abs(dist_primary - 1) * m_genes.aggression_weight - dist_primary * 3.5

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
        score -= hex_angle_distance(cand_facing, best_dir) * (0.45 * m_genes.self_defense_weight)

        # 3. Flanking, Cleave alignment, Hit-and-Run safety & Self-Defense around nearby enemies
        enemies_in_my_arc = 0
        for ou in opp_units:
            if ou.hex_pos is None:
                continue
            d = hex_distance(cand_hex, ou.hex_pos)
            if d == 1:
                # Are we threatening `ou`'s vulnerable arc?
                inc_on_ou = hex_direction_between(cand_hex, ou.hex_pos)
                if inc_on_ou is not None:
                    rel_on_ou = relative_direction(ou.facing, inc_on_ou)
                    if rel_on_ou in ou.spec.vulnerable_arcs:
                        score += m_genes.flanking_weight

                # Is `ou` in our offensive attack_arcs?
                dir_to_ou = hex_direction_between(cand_hex, ou.hex_pos)
                if dir_to_ou is not None:
                    rel_to_ou = relative_direction(cand_facing, dir_to_ou)
                    if rel_to_ou in unit.spec.attack_arcs:
                        enemies_in_my_arc += 1
                        score += 10.0

                # Are we exposing our own vulnerable_arcs to `ou`?
                inc_on_us = hex_direction_between(ou.hex_pos, cand_hex)
                if inc_on_us is not None:
                    rel_on_us = relative_direction(cand_facing, inc_on_us)
                    if rel_on_us in unit.spec.vulnerable_arcs:
                        score -= m_genes.self_defense_weight * 1.25

                    # Is `cand_hex` directly inside `ou`'s current attack_arcs?
                    rel_from_ou = relative_direction(ou.facing, inc_on_us)
                    if rel_from_ou in ou.spec.attack_arcs:
                        if attacked_this_turn:
                            score -= m_genes.hit_and_run_weight
                        else:
                            score -= m_genes.self_defense_weight * 0.45

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

                if ply >= 3:
                    # Ply 3: Simulate our own follow-up turn after opponent's response
                    my_followup, st_after_my_followup = self._estimate_team_counter_turn_impact(
                        st_after_opp, my_team
                    )
                    followup_net = my_followup * l_genes.followup_discount

                    if ply >= 4:
                        # Ply 4: Also simulate the opponent's 2nd counter-turn response!
                        opp_second_impact, _ = self._estimate_team_counter_turn_impact(
                            st_after_my_followup, opp_team
                        )
                        followup_net -= (
                            opp_second_impact
                            * l_genes.enemy_response_weight
                            * l_genes.followup_discount
                        )

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
