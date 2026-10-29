"""
Creature specifications, evolution progression math, and class lineups for Denizens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from constants import (
    CLASS_LIZARD,
    CLASS_MAMMAL,
    CLASS_SERPENT,
    CLASS_TURTLE,
    DEFAULT_VULNERABLE_ARCS,
    DIRS_ALL_SIX,
    DIRS_FORWARD_ONLY,
    DIRS_FORWARD_REAR,
    DIRS_FRONT_ARC,
    DIRS_FRONT_ARC_AND_REAR,
    FLANK_LEFT,
    FLANK_RIGHT,
    FORWARD,
    REAR,
)


def evolution_battles_required(stage: int) -> int:
    """
    Cumulative battles survived required to reach Stage `stage` (1..8).
    Triangular progression: (stage - 1) * stage // 2 -> 0, 1, 3, 6, 10, 15, 21, 28.
    """
    if stage <= 1:
        return 0
    return ((stage - 1) * stage) // 2


def stage_for_battles_survived(battles_survived: int) -> int:
    """Return the evolution stage (1..8) corresponding to total battles survived."""
    stage = 1
    for s in range(1, 9):
        if battles_survived >= evolution_battles_required(s):
            stage = s
        else:
            break
    return stage


@dataclass(frozen=True)
class CreatureSpec:
    """
    Data-driven specification for a creature stage or enemy form.

    Movement, Attack & Strength Attributes:
      - num_rotate_moves: Number of (Rotate -> Move) steps allowed per turn.
      - max_rotate_per_move: Maximum 60° hex faces (0..3) the creature can rotate
        from its step-start facing during each Rotate-Move step.
      - legal_move_dirs: Allowed relative movement directions (relative to
        instantaneous facing at the moment of the move).
      - total_attacks_per_move: Maximum attacks allowed across the creature's turn
        (taken at BEFORE_ROTATE, AFTER_ROTATE, or AFTER_MOVE junctures, max 1 per juncture).
      - can_fly: Whether the creature can fly over occupied hexes during intermediate
        move steps as long as it can land on an empty hex before ending its turn.
      - can_rotate_after_move: Whether the creature can perform a post-move rotation
        (up to `max_rotate_per_move`) after completing its move (e.g. Newt and the
        high-mobility Lizard lineage can Move -> Rotate or Rotate -> Move -> Rotate).
      - point_cost: Strength point value used by the computer opponent's wave-buying
        algorithm and adaptive wave scaling.
    """
    creature_id: str
    name: str
    creature_class: str
    stage: int
    asset_slug: str
    point_cost: int = 1
    num_rotate_moves: int = 1
    max_rotate_per_move: int = 1
    legal_move_dirs: FrozenSet[int] = DIRS_FORWARD_ONLY
    total_attacks_per_move: int = 1
    can_fly: bool = False
    can_rotate_after_move: bool = False
    max_hp: int = 10
    attack: int = 4
    defense: int = 2
    attack_arcs: FrozenSet[int] = frozenset({FORWARD})
    vulnerable_arcs: FrozenSet[int] = DEFAULT_VULNERABLE_ARCS
    special_powers: Tuple[str, ...] = ()
    unlock_round: int = 1

    @property
    def topdown_path(self) -> str:
        return f"assets/monsters/{self.creature_class}/topdown/{self.asset_slug}.png"

    @property
    def portrait_path(self) -> str:
        return f"assets/monsters/{self.creature_class}/portraits/{self.asset_slug}.png"


SERPENT_LINEUP: Tuple[CreatureSpec, ...] = (
    CreatureSpec(
        creature_id="serpent_01_adder",
        name="Adder",
        creature_class=CLASS_SERPENT,
        stage=1,
        asset_slug="01_adder",
        point_cost=1,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_ONLY,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=8,
        attack=5,
        defense=1,
    ),
    CreatureSpec(
        creature_id="serpent_02_viper",
        name="Viper",
        creature_class=CLASS_SERPENT,
        stage=2,
        asset_slug="02_viper",
        point_cost=2,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=10,
        attack=6,
        defense=1,
        special_powers=("venom_fangs",),
    ),
    CreatureSpec(
        creature_id="serpent_03_serpent",
        name="Serpent",
        creature_class=CLASS_SERPENT,
        stage=3,
        asset_slug="03_serpent",
        point_cost=4,
        num_rotate_moves=2,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=13,
        attack=8,
        defense=2,
        special_powers=("constrict",),
    ),
    CreatureSpec(
        creature_id="serpent_04_jaculus",
        name="Jaculus",
        creature_class=CLASS_SERPENT,
        stage=4,
        asset_slug="04_jaculus",
        point_cost=6,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC,
        total_attacks_per_move=1,
        can_fly=True,
        max_hp=15,
        attack=10,
        defense=2,
        special_powers=("winged_javelin",),
    ),
    CreatureSpec(
        creature_id="serpent_05_amphisbaena",
        name="Amphisbaena",
        creature_class=CLASS_SERPENT,
        stage=5,
        asset_slug="05_amphisbaena",
        point_cost=9,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=18,
        attack=11,
        defense=3,
        attack_arcs=DIRS_FORWARD_REAR,
        vulnerable_arcs=frozenset({FLANK_LEFT, FLANK_RIGHT}),
        special_powers=("twin_heads",),
    ),
    CreatureSpec(
        creature_id="serpent_06_wyrm",
        name="Wyrm",
        creature_class=CLASS_SERPENT,
        stage=6,
        asset_slug="06_wyrm",
        point_cost=13,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=True,
        max_hp=22,
        attack=13,
        defense=4,
        special_powers=("draconic_sweep",),
    ),
    CreatureSpec(
        creature_id="serpent_07_basilisk",
        name="Basilisk",
        creature_class=CLASS_SERPENT,
        stage=7,
        asset_slug="07_basilisk",
        point_cost=18,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=26,
        attack=15,
        defense=5,
        special_powers=("petrifying_gaze", "lethal_venom"),
    ),
    CreatureSpec(
        creature_id="serpent_08_hydra",
        name="Hydra",
        creature_class=CLASS_SERPENT,
        stage=8,
        asset_slug="08_hydra",
        point_cost=24,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=3,
        can_fly=False,
        max_hp=32,
        attack=17,
        defense=6,
        attack_arcs=DIRS_FRONT_ARC,
        special_powers=("multi_headed_strike", "regeneration"),
    ),
)


LIZARD_LINEUP: Tuple[CreatureSpec, ...] = (
    CreatureSpec(
        creature_id="lizard_01_newt",
        name="Newt",
        creature_class=CLASS_LIZARD,
        stage=1,
        asset_slug="01_newt",
        point_cost=1,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_ONLY,
        total_attacks_per_move=1,
        can_fly=False,
        can_rotate_after_move=True,
        max_hp=9,
        attack=3,
        defense=2,
    ),
    CreatureSpec(
        creature_id="lizard_02_horned",
        name="Horned Lizard",
        creature_class=CLASS_LIZARD,
        stage=2,
        asset_slug="02_horned",
        point_cost=2,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        can_rotate_after_move=True,
        max_hp=11,
        attack=4,
        defense=3,
        special_powers=("thorny_hide",),
    ),
    CreatureSpec(
        creature_id="lizard_03_monitor",
        name="Monitor",
        creature_class=CLASS_LIZARD,
        stage=3,
        asset_slug="03_monitor",
        point_cost=4,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        can_rotate_after_move=True,
        max_hp=14,
        attack=6,
        defense=3,
        special_powers=("predator_sprint",),
    ),
    CreatureSpec(
        creature_id="lizard_04_komodo",
        name="Komodo",
        creature_class=CLASS_LIZARD,
        stage=4,
        asset_slug="04_komodo",
        point_cost=6,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_ALL_SIX,
        total_attacks_per_move=2,
        can_fly=False,
        can_rotate_after_move=True,
        max_hp=17,
        attack=8,
        defense=4,
        special_powers=("serrated_bite",),
    ),
    CreatureSpec(
        creature_id="lizard_05_wyrmling",
        name="Wyrmling",
        creature_class=CLASS_LIZARD,
        stage=5,
        asset_slug="05_wyrmling",
        point_cost=9,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=True,
        can_rotate_after_move=True,
        max_hp=20,
        attack=9,
        defense=4,
        special_powers=("fledgling_flight",),
    ),
    CreatureSpec(
        creature_id="lizard_06_drake",
        name="Drake",
        creature_class=CLASS_LIZARD,
        stage=6,
        asset_slug="06_drake",
        point_cost=13,
        num_rotate_moves=3,
        max_rotate_per_move=3,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=True,
        can_rotate_after_move=True,
        max_hp=24,
        attack=11,
        defense=5,
        special_powers=("wing_gust",),
    ),
    CreatureSpec(
        creature_id="lizard_07_young_dragon",
        name="Young Dragon",
        creature_class=CLASS_LIZARD,
        stage=7,
        asset_slug="07_young_dragon",
        point_cost=18,
        num_rotate_moves=3,
        max_rotate_per_move=3,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=True,
        can_rotate_after_move=True,
        max_hp=28,
        attack=13,
        defense=6,
        attack_arcs=DIRS_FRONT_ARC,
        special_powers=("fire_breath",),
    ),
    CreatureSpec(
        creature_id="lizard_08_tian_long",
        name="Tian Long",
        creature_class=CLASS_LIZARD,
        stage=8,
        asset_slug="08_tian_long",
        point_cost=24,
        num_rotate_moves=4,
        max_rotate_per_move=3,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=3,
        can_fly=True,
        can_rotate_after_move=True,
        max_hp=33,
        attack=15,
        defense=7,
        attack_arcs=DIRS_FRONT_ARC,
        special_powers=("celestial_flight", "tempest_fury"),
    ),
)


TURTLE_LINEUP: Tuple[CreatureSpec, ...] = (
    CreatureSpec(
        creature_id="turtle_01_pond_turtle",
        name="Pond Turtle",
        creature_class=CLASS_TURTLE,
        stage=1,
        asset_slug="01_pond_turtle",
        point_cost=1,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_ONLY,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=12,
        attack=3,
        defense=4,
    ),
    CreatureSpec(
        creature_id="turtle_02_box_turtle",
        name="Box Turtle",
        creature_class=CLASS_TURTLE,
        stage=2,
        asset_slug="02_box_turtle",
        point_cost=2,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=15,
        attack=4,
        defense=5,
        special_powers=("hinged_shell",),
    ),
    CreatureSpec(
        creature_id="turtle_03_snapper",
        name="Snapper",
        creature_class=CLASS_TURTLE,
        stage=3,
        asset_slug="03_snapper",
        point_cost=4,
        num_rotate_moves=1,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=18,
        attack=6,
        defense=6,
        special_powers=("snapping_jaws",),
    ),
    CreatureSpec(
        creature_id="turtle_04_tortoise",
        name="Tortoise",
        creature_class=CLASS_TURTLE,
        stage=4,
        asset_slug="04_tortoise",
        point_cost=6,
        num_rotate_moves=1,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=22,
        attack=7,
        defense=7,
        vulnerable_arcs=frozenset({REAR}),
        special_powers=("fortress_carapace",),
    ),
    CreatureSpec(
        creature_id="turtle_05_giant_snapper",
        name="Giant Snapper",
        creature_class=CLASS_TURTLE,
        stage=5,
        asset_slug="05_giant_snapper",
        point_cost=9,
        num_rotate_moves=2,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=26,
        attack=9,
        defense=8,
        vulnerable_arcs=frozenset({REAR}),
        special_powers=("crushing_bite",),
    ),
    CreatureSpec(
        creature_id="turtle_06_alligator_snapper",
        name="Alligator Snapper",
        creature_class=CLASS_TURTLE,
        stage=6,
        asset_slug="06_alligator_snapper",
        point_cost=13,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=30,
        attack=11,
        defense=9,
        vulnerable_arcs=frozenset({REAR}),
        special_powers=("spiked_ridge",),
    ),
    CreatureSpec(
        creature_id="turtle_07_war_turtle",
        name="War Turtle",
        creature_class=CLASS_TURTLE,
        stage=7,
        asset_slug="07_war_turtle",
        point_cost=18,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=35,
        attack=12,
        defense=10,
        vulnerable_arcs=frozenset(),
        special_powers=("ironclad_bastion",),
    ),
    CreatureSpec(
        creature_id="turtle_08_dragon_turtle",
        name="Dragon Turtle",
        creature_class=CLASS_TURTLE,
        stage=8,
        asset_slug="08_dragon_turtle",
        point_cost=24,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=42,
        attack=14,
        defense=12,
        attack_arcs=DIRS_FRONT_ARC,
        vulnerable_arcs=frozenset(),
        special_powers=("molten_carapace", "steam_breath"),
    ),
)


MAMMAL_LINEUP: Tuple[CreatureSpec, ...] = (
    CreatureSpec(
        creature_id="mammal_01_rat",
        name="Rat",
        creature_class=CLASS_MAMMAL,
        stage=1,
        asset_slug="01_rat",
        point_cost=1,
        unlock_round=1,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_ONLY,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=8,
        attack=4,
        defense=1,
    ),
    CreatureSpec(
        creature_id="mammal_02_wolf",
        name="Wolf",
        creature_class=CLASS_MAMMAL,
        stage=2,
        asset_slug="02_wolf",
        point_cost=2,
        unlock_round=2,
        num_rotate_moves=2,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_ONLY,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=11,
        attack=5,
        defense=2,
        special_powers=("pack_tactics",),
    ),
    CreatureSpec(
        creature_id="mammal_03_bear",
        name="Bear",
        creature_class=CLASS_MAMMAL,
        stage=3,
        asset_slug="03_bear",
        point_cost=4,
        unlock_round=3,
        num_rotate_moves=1,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=18,
        attack=7,
        defense=4,
        special_powers=("maul",),
    ),
    CreatureSpec(
        creature_id="mammal_04_lion",
        name="Lion",
        creature_class=CLASS_MAMMAL,
        stage=4,
        asset_slug="04_lion",
        point_cost=5,
        unlock_round=4,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC,
        total_attacks_per_move=1,
        can_fly=False,
        max_hp=16,
        attack=8,
        defense=3,
        special_powers=("pounce",),
    ),
    CreatureSpec(
        creature_id="mammal_05_griffon",
        name="Griffon",
        creature_class=CLASS_MAMMAL,
        stage=5,
        asset_slug="05_griffon",
        point_cost=7,
        unlock_round=5,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC,
        total_attacks_per_move=2,
        can_fly=True,
        max_hp=20,
        attack=10,
        defense=4,
        special_powers=("diving_talons",),
    ),
    CreatureSpec(
        creature_id="mammal_06_owlbear",
        name="Owlbear",
        creature_class=CLASS_MAMMAL,
        stage=6,
        asset_slug="06_owlbear",
        point_cost=9,
        unlock_round=6,
        num_rotate_moves=2,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=24,
        attack=11,
        defense=5,
        special_powers=("ferocious_hug",),
    ),
    CreatureSpec(
        creature_id="mammal_07_behemoth",
        name="Behemoth",
        creature_class=CLASS_MAMMAL,
        stage=7,
        asset_slug="07_behemoth",
        point_cost=17,
        unlock_round=11,
        num_rotate_moves=2,
        max_rotate_per_move=1,
        legal_move_dirs=DIRS_FORWARD_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=32,
        attack=13,
        defense=7,
        special_powers=("gore_charge",),
    ),
    CreatureSpec(
        creature_id="mammal_08_displacer_beast",
        name="Displacer Beast",
        creature_class=CLASS_MAMMAL,
        stage=8,
        asset_slug="08_displacer_beast",
        point_cost=11,
        unlock_round=7,
        num_rotate_moves=3,
        max_rotate_per_move=3,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=22,
        attack=12,
        defense=5,
        attack_arcs=DIRS_FRONT_ARC,
        special_powers=("displacement", "tentacle_lash"),
    ),
    CreatureSpec(
        creature_id="mammal_09_manticore",
        name="Manticore",
        creature_class=CLASS_MAMMAL,
        stage=9,
        asset_slug="09_manticore",
        point_cost=14,
        unlock_round=9,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=True,
        max_hp=26,
        attack=13,
        defense=5,
        special_powers=("tail_spikes",),
    ),
    CreatureSpec(
        creature_id="mammal_10_rajang",
        name="Rajang",
        creature_class=CLASS_MAMMAL,
        stage=10,
        asset_slug="10_rajang",
        point_cost=22,
        unlock_round=13,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_ALL_SIX,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=30,
        attack=16,
        defense=6,
        special_powers=("thunder_rage",),
    ),
    CreatureSpec(
        creature_id="mammal_11_dire_wolf",
        name="Dire Wolf",
        creature_class=CLASS_MAMMAL,
        stage=11,
        asset_slug="11_dire_wolf",
        point_cost=12,
        unlock_round=8,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=24,
        attack=13,
        defense=5,
        special_powers=("alpha_howl",),
    ),
    CreatureSpec(
        creature_id="mammal_12_sabertooth",
        name="Sabertooth",
        creature_class=CLASS_MAMMAL,
        stage=12,
        asset_slug="12_sabertooth",
        point_cost=15,
        unlock_round=10,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=26,
        attack=15,
        defense=5,
        special_powers=("rending_fangs",),
    ),
    CreatureSpec(
        creature_id="mammal_13_hell_hound",
        name="Hell Hound",
        creature_class=CLASS_MAMMAL,
        stage=13,
        asset_slug="13_hell_hound",
        point_cost=19,
        unlock_round=12,
        num_rotate_moves=3,
        max_rotate_per_move=2,
        legal_move_dirs=DIRS_FRONT_ARC_AND_REAR,
        total_attacks_per_move=2,
        can_fly=False,
        max_hp=28,
        attack=15,
        defense=6,
        attack_arcs=DIRS_FRONT_ARC,
        special_powers=("hellfire_breath",),
    ),
)

CLASS_LINEUPS: Dict[str, Tuple[CreatureSpec, ...]] = {
    CLASS_SERPENT: SERPENT_LINEUP,
    CLASS_LIZARD: LIZARD_LINEUP,
    CLASS_TURTLE: TURTLE_LINEUP,
    CLASS_MAMMAL: MAMMAL_LINEUP,
}

CREATURE_BY_ID: Dict[str, CreatureSpec] = {
    spec.creature_id: spec
    for lineup in CLASS_LINEUPS.values()
    for spec in lineup
}


def get_creature_spec(creature_class: str, stage: int) -> CreatureSpec:
    """Return the CreatureSpec for the given class and 1-based stage/index."""
    lineup = CLASS_LINEUPS[creature_class]
    clamped_idx = max(0, min(len(lineup) - 1, stage - 1))
    return lineup[clamped_idx]


def is_mammal_unlocked(spec: CreatureSpec, round_number: int) -> bool:
    """
    Return True if `spec` is unlocked for Mammal wave purchasing in `round_number` (1-based).
    Mammals unlock progressively one per round in ascending power/cost order (Rounds 1..13).
    """
    return max(1, round_number) >= spec.unlock_round


def get_unlocked_mammals(round_number: int) -> List[CreatureSpec]:
    """
    Return all Mammal enemy specs unlocked on or before `round_number`, sorted by
    `unlock_round` (ascending power/cost order).
    """
    r = max(1, round_number)
    unlocked = [spec for spec in MAMMAL_LINEUP if spec.unlock_round <= r]
    unlocked.sort(key=lambda s: (s.unlock_round, s.point_cost))
    return unlocked
