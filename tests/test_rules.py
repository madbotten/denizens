"""
Unit tests for Denizens shared rules (`rules.py`).
"""

import rules


def test_arena_hex_count_and_bounds() -> None:
    hexes = rules.get_all_arena_hexes(radius=4)
    assert len(hexes) == 61
    assert (0, 0) in hexes
    assert (4, -4) in hexes
    assert not rules.is_valid_hex(5, 0, radius=4)
    assert not rules.is_valid_hex(3, 2, radius=4)


def test_relative_and_angular_hex_math() -> None:
    assert rules.relative_direction(rules.FACING_N, rules.FACING_N) == rules.FORWARD
    assert rules.relative_direction(rules.FACING_N, rules.FACING_NE) == rules.FORWARD_RIGHT
    assert rules.relative_direction(rules.FACING_N, rules.FACING_SE) == rules.FLANK_RIGHT
    assert rules.relative_direction(rules.FACING_N, rules.FACING_S) == rules.REAR
    assert rules.relative_direction(rules.FACING_N, rules.FACING_SW) == rules.FLANK_LEFT
    assert rules.relative_direction(rules.FACING_N, rules.FACING_NW) == rules.FORWARD_LEFT

    assert rules.hex_angle_distance(0, 0) == 0
    assert rules.hex_angle_distance(0, 1) == 1
    assert rules.hex_angle_distance(0, 5) == 1
    assert rules.hex_angle_distance(0, 2) == 2
    assert rules.hex_angle_distance(0, 4) == 2
    assert rules.hex_angle_distance(0, 3) == 3


def test_evolution_triangular_progression() -> None:
    expected = [0, 1, 3, 6, 10, 15, 21, 28]
    actual = [rules.evolution_battles_required(s) for s in range(1, 9)]
    assert actual == expected
    assert rules.stage_for_battles_survived(0) == 1
    assert rules.stage_for_battles_survived(1) == 2
    assert rules.stage_for_battles_survived(2) == 2
    assert rules.stage_for_battles_survived(3) == 3
    assert rules.stage_for_battles_survived(27) == 7
    assert rules.stage_for_battles_survived(28) == 8


def test_basic_stage1_creature_rotate_and_move() -> None:
    """
    Basic Stage 1 creatures have num_rotate_moves=1, max_rotate_per_move=1,
    and legal_move_dirs={FORWARD}.
    """
    state = rules.GameState(arena_radius=4)
    adder_spec = rules.get_creature_spec(rules.CLASS_SERPENT, 1)
    assert adder_spec.num_rotate_moves == 1
    assert adder_spec.max_rotate_per_move == 1
    assert adder_spec.legal_move_dirs == frozenset({rules.FORWARD})

    unit = rules.Unit(
        unit_id="u1",
        spec=adder_spec,
        team=rules.TEAM_PLAYER,
        hex_pos=(0, 0),
        facing=rules.FACING_N,
    )
    state.units["u1"] = unit

    # Can only rotate within +-1 of initial facing N (0): NW (5), N (0), NE (1)
    legal_rots = rules.get_legal_rotations(state, "u1")
    assert set(legal_rots) == {rules.FACING_NW, rules.FACING_N, rules.FACING_NE}
    assert not rules.rotate_unit(state, "u1", rules.FACING_SE)
    assert not rules.rotate_unit(state, "u1", rules.FACING_S)

    # Rotate 1 step clockwise to NE (1)
    assert rules.rotate_unit(state, "u1", rules.FACING_NE)
    assert unit.facing == rules.FACING_NE
    assert unit.juncture == rules.JUNCTURE_AFTER_ROTATE

    # Can only move FORWARD relative to new facing NE -> (1, -1)
    legal_moves = rules.get_legal_moves(state, "u1")
    assert legal_moves == [(1, -1)]
    assert rules.move_unit(state, "u1", (1, -1))
    assert unit.hex_pos == (1, -1)
    assert unit.juncture == rules.JUNCTURE_AFTER_MOVE

    # With num_rotate_moves=1, no further moves or rotations are allowed this turn
    assert rules.get_legal_moves(state, "u1") == []
    assert rules.get_legal_rotations(state, "u1") == []


def test_tian_long_four_rotate_moves_and_omnidirectional_rotation() -> None:
    """
    Stage 8 Tian Long has num_rotate_moves=4, max_rotate_per_move=3,
    legal_move_dirs={FORWARD, FORWARD_LEFT, FORWARD_RIGHT, REAR}, can_fly=True.
    """
    state = rules.GameState(arena_radius=4)
    tian_spec = rules.get_creature_spec(rules.CLASS_LIZARD, 8)
    assert tian_spec.num_rotate_moves == 4
    assert tian_spec.max_rotate_per_move == 3
    assert tian_spec.legal_move_dirs == frozenset(
        {rules.FORWARD, rules.FORWARD_LEFT, rules.FORWARD_RIGHT, rules.REAR}
    )
    assert tian_spec.can_fly is True

    unit = rules.Unit(
        unit_id="tian",
        spec=tian_spec,
        team=rules.TEAM_PLAYER,
        hex_pos=(0, 2),
        facing=rules.FACING_N,
    )
    state.units["tian"] = unit

    # Step 1: Rotate 180 degrees to South (3 faces away, allowed by max_rotate_per_move=3)
    assert set(rules.get_legal_rotations(state, "tian")) == {0, 1, 2, 3, 4, 5}
    assert rules.rotate_unit(state, "tian", rules.FACING_S)
    # Move REAR relative to South facing -> moves North to (0, 1)
    assert (0, 1) in rules.get_legal_moves(state, "tian")
    assert rules.move_unit(state, "tian", (0, 1))
    assert unit.current_step == 0
    assert unit.juncture == rules.JUNCTURE_AFTER_MOVE

    # Step 2: Rotate to NE (1) and move FORWARD_LEFT (which is N -> (0, 0))
    assert rules.rotate_unit(state, "tian", rules.FACING_NE)
    assert unit.current_step == 1
    assert unit.juncture == rules.JUNCTURE_AFTER_ROTATE
    assert (0, 0) in rules.get_legal_moves(state, "tian")
    assert rules.move_unit(state, "tian", (0, 0))

    # Step 3: Move without rotating (optional rotation) FORWARD (NE -> (1, -1))
    assert (1, -1) in rules.get_legal_moves(state, "tian")
    assert rules.move_unit(state, "tian", (1, -1))
    assert unit.current_step == 2

    # Step 4: Rotate to NW (5) and skip moving (rotate-move-rotate style)
    assert rules.rotate_unit(state, "tian", rules.FACING_NW)
    assert unit.current_step == 3
    assert unit.juncture == rules.JUNCTURE_AFTER_ROTATE
    # Still has Step 4's optional move available
    assert len(rules.get_legal_moves(state, "tian")) > 0
    assert rules.move_unit(state, "tian", (0, -1))
    assert unit.juncture == rules.JUNCTURE_AFTER_MOVE

    # All 4 rotate-moves exhausted
    assert rules.get_legal_moves(state, "tian") == []
    assert rules.get_legal_rotations(state, "tian") == []


def test_attack_junctures_and_limits() -> None:
    """
    Attacks can be taken at BEFORE_ROTATE, AFTER_ROTATE, and AFTER_MOVE,
    at most 1 attack per juncture, up to total_attacks_per_move.
    """
    state = rules.GameState(arena_radius=4)
    # Stage 4 Komodo has total_attacks_per_move = 2
    komodo_spec = rules.get_creature_spec(rules.CLASS_LIZARD, 4)
    assert komodo_spec.total_attacks_per_move == 2

    unit = rules.Unit(
        unit_id="k1",
        spec=komodo_spec,
        team=rules.TEAM_PLAYER,
        hex_pos=(0, 0),
        facing=rules.FACING_N,
    )
    state.units["k1"] = unit

    # 1. Attack at BEFORE_ROTATE
    assert unit.juncture == rules.JUNCTURE_BEFORE_ROTATE
    assert rules.can_unit_attack_now(state, "k1")
    assert rules.record_juncture_attack(state, "k1")
    # Cannot attack a second time at the exact same BEFORE_ROTATE juncture
    assert not rules.can_unit_attack_now(state, "k1")

    # 2. Rotate to NE -> enters AFTER_ROTATE juncture, enabling the 2nd attack
    assert rules.rotate_unit(state, "k1", rules.FACING_NE)
    assert unit.juncture == rules.JUNCTURE_AFTER_ROTATE
    assert rules.can_unit_attack_now(state, "k1")
    assert rules.record_juncture_attack(state, "k1")
    assert unit.attacks_used == 2

    # 3. Move to (1, -1) -> enters AFTER_MOVE juncture, but total_attacks_per_move (2) is reached
    assert rules.move_unit(state, "k1", (1, -1))
    assert unit.juncture == rules.JUNCTURE_AFTER_MOVE
    assert not rules.can_unit_attack_now(state, "k1")


def test_flying_over_opponent_requires_empty_landing_hex() -> None:
    """
    Winged creatures (`can_fly=True`) can move over an occupied hex during intermediate
    steps if and only if they can land on an empty hex before their moves run out.
    """
    state = rules.GameState(arena_radius=4)
    # Jaculus (Stage 4 Serpent): num_rotate_moves=2, can_fly=True, dirs={FORWARD, FORWARD_LEFT, FORWARD_RIGHT}
    jaculus_spec = rules.get_creature_spec(rules.CLASS_SERPENT, 4)
    rat_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 1)

    flyer = rules.Unit("flyer", jaculus_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N)
    blocker1 = rules.Unit("b1", rat_spec, rules.TEAM_ENEMY, (0, 0), rules.FACING_S)
    state.units["flyer"] = flyer
    state.units["b1"] = blocker1

    # Step 1: (0, 0) is occupied by b1, but flyer has 1 move left after Step 1 and (0, -1) is empty
    legal_mvs_1 = rules.get_legal_moves(state, "flyer")
    assert (0, 0) in legal_mvs_1
    assert rules.move_unit(state, "flyer", (0, 0))

    # While overlapping b1 at (0, 0), flyer cannot end turn or attack
    assert rules.is_unit_overlapping(state, "flyer")
    assert not rules.can_end_turn(state)
    assert not rules.can_unit_attack_now(state, "flyer")

    # Step 2: Flyer lands on empty hex (0, -1)
    assert (0, -1) in rules.get_legal_moves(state, "flyer")
    assert rules.move_unit(state, "flyer", (0, -1))
    assert not rules.is_unit_overlapping(state, "flyer")
    assert rules.can_end_turn(state)


def test_initial_game_state_army_and_round1_rats_wave() -> None:
    """
    Round 1 starts in PHASE_SETUP with 6 random Stage 1 reptiles in the left reserve tray
    and 4 enemy Rats (budget = 4 pts @ 1 pt each) in the upper half of the battlefield.
    """
    import random

    rng = random.Random(101)
    state = rules.create_initial_game_state(game_mode=rules.GAME_MODE_BASIC, rng=rng)

    assert state.phase == rules.PHASE_SETUP
    assert state.round_number == 1
    assert state.wave_budget == 4

    player_units = state.get_player_army_units()
    assert len(player_units) == rules.INITIAL_ARMY_SIZE == 6
    for idx, pu in enumerate(player_units):
        assert pu.in_reserve
        assert not pu.is_deployed
        assert pu.reserve_slot == idx
        assert pu.spec.stage == 1
        assert pu.spec.creature_class in rules.PLAYER_CLASSES
        assert pu.spec.point_cost == 1

    enemy_units = [u for u in state.units.values() if u.team == rules.TEAM_ENEMY]
    assert len(enemy_units) == 4
    for eu in enemy_units:
        assert eu.is_deployed
        assert eu.spec.name == "Rat"
        assert eu.spec.point_cost == 1
        assert eu.hex_pos is not None
        assert rules.is_upper_half_hex(eu.hex_pos)
        assert eu.facing == rules.FACING_S


def test_lower_half_deployment_swap_undeploy_and_confirm() -> None:
    """
    During PHASE_SETUP, player units can be placed only in the 28 lower-half hexes (2*r + q > 0),
    one per hex (swapping if occupied), undeployed back to reserve, and confirmed to start battle.
    """
    import random

    state = rules.create_initial_game_state(rng=random.Random(7))
    deploy_hexes = rules.get_valid_player_deploy_hexes(state)
    assert len(deploy_hexes) == 28
    assert (0, 2) in deploy_hexes
    assert (0, 0) not in deploy_hexes
    assert (0, -2) not in deploy_hexes

    player_units = state.get_player_army_units()
    u1, u2 = player_units[0], player_units[1]

    # Cannot confirm deployment before at least 1 unit is placed on the board
    assert not rules.can_confirm_deployment(state)
    assert not rules.confirm_deployment(state)

    # Cannot deploy into upper half or midline
    assert not rules.deploy_unit_to_hex(state, u1.unit_id, (0, -2))
    assert not rules.deploy_unit_to_hex(state, u1.unit_id, (0, 0))

    # Deploy u1 to lower-half hex (0, 2) — still cannot confirm because 5 units remain in reserve
    assert rules.deploy_unit_to_hex(state, u1.unit_id, (0, 2))
    assert u1.is_deployed and u1.hex_pos == (0, 2)
    assert not rules.can_confirm_deployment(state)

    # Deploying u2 from reserve onto (0, 2) places u2 at (0, 2) and returns u1 to reserve
    assert rules.deploy_unit_to_hex(state, u2.unit_id, (0, 2))
    assert u2.hex_pos == (0, 2)
    assert u1.in_reserve and u1.hex_pos is None

    # Undeploy u2 back to reserve
    assert rules.undeploy_unit_to_reserve(state, u2.unit_id)
    assert u2.in_reserve
    assert not rules.can_confirm_deployment(state)

    # Deploy all 6 player units onto distinct lower-half hexes, rotate u1, and confirm deployment
    for idx, pu in enumerate(player_units):
        assert rules.deploy_unit_to_hex(state, pu.unit_id, deploy_hexes[idx])
    assert rules.can_confirm_deployment(state)
    assert rules.rotate_unit(state, u1.unit_id, rules.FACING_SW)
    assert u1.facing == rules.FACING_SW
    assert rules.confirm_deployment(state)
    assert state.phase == rules.PHASE_BATTLE


def test_base_reinforcements_and_max_army_cap() -> None:
    """
    grant_base_reinforcements adds REINFORCEMENTS_BASE (4) Stage 1 reptiles and caps at MAX_ARMY_SIZE (16).
    """
    import random

    rng = random.Random(99)
    state = rules.create_initial_game_state(rng=rng)
    assert len(state.get_player_army_units()) == 6

    added_1 = rules.grant_base_reinforcements(state, count=rules.REINFORCEMENTS_BASE, rng=rng)
    assert len(added_1) == 4
    assert len(state.get_player_army_units()) == 10

    added_2 = rules.grant_base_reinforcements(state, count=rules.REINFORCEMENTS_BASE, rng=rng)
    assert len(added_2) == 4
    assert len(state.get_player_army_units()) == 14

    # Next reinforcement grant can only add 2 units before hitting MAX_ARMY_SIZE = 16
    added_3 = rules.grant_base_reinforcements(state, count=rules.REINFORCEMENTS_BASE, rng=rng)
    assert len(added_3) == 2
    assert len(state.get_player_army_units()) == rules.MAX_ARMY_SIZE == 16


def test_wave_points_basic_adaptive_vs_advanced_schedule() -> None:
    """
    Advanced mode uses fixed schedule 4 + 3*(R-1) + (R-1)^2; Basic mode scales adaptively
    to the player's surviving army points. Wave purchases respect budget and MAX_ARMY_SIZE.
    """
    import random

    rng = random.Random(123)
    state_adv = rules.create_initial_game_state(game_mode=rules.GAME_MODE_ADVANCED, rng=rng)
    expected_adv = [4, 8, 14, 22, 32, 44]
    for r, exp in enumerate(expected_adv, start=1):
        state_adv.round_number = r
        assert rules.enemy_wave_budget(state_adv) == exp

    state_basic = rules.create_initial_game_state(game_mode=rules.GAME_MODE_BASIC, rng=rng)
    state_basic.round_number = 4
    # With 6 Stage 1 units (6 pts total), adaptive budget is modest
    low_budget = rules.enemy_wave_budget(state_basic)
    # Upgrade one unit to Stage 8 (24 pts) -> adaptive budget rises with player strength
    u0 = state_basic.get_player_army_units()[0]
    u0.spec = rules.get_creature_spec(rules.CLASS_LIZARD, 8)
    high_budget = rules.enemy_wave_budget(state_basic)
    assert high_budget > low_budget

    # Purchasing a large wave never exceeds budget or MAX_ARMY_SIZE
    wave = rules.purchase_enemy_wave(round_number=6, budget=85, max_units=rules.MAX_ARMY_SIZE, rng=rng)
    assert 1 <= len(wave) <= rules.MAX_ARMY_SIZE
    assert sum(s.point_cost for s in wave) <= 85

