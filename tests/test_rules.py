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

    # With num_rotate_moves=1 and can_rotate_after_move=False, no further moves or rotations are allowed this turn
    assert rules.get_legal_moves(state, "u1") == []
    assert rules.get_legal_rotations(state, "u1") == []

    # By contrast, Stage 1 Newt (Lizard) has can_rotate_after_move=True so it can move then rotate (and attack!)
    newt_spec = rules.get_creature_spec(rules.CLASS_LIZARD, 1)
    assert newt_spec.can_rotate_after_move is True
    newt = rules.Unit(
        unit_id="n1",
        spec=newt_spec,
        team=rules.TEAM_PLAYER,
        hex_pos=(0, 1),
        facing=rules.FACING_N,
    )
    state.units["n1"] = newt

    # Newt moves 1 hex forward to (0, 0) first, then rotates CW to NE (1) and can still attack
    assert rules.move_unit(state, "n1", (0, 0))
    assert newt.hex_pos == (0, 0)
    assert newt.juncture == rules.JUNCTURE_AFTER_MOVE
    assert set(rules.get_legal_rotations(state, "n1")) == {
        rules.FACING_NW,
        rules.FACING_N,
        rules.FACING_NE,
    }
    assert rules.rotate_unit(state, "n1", rules.FACING_NE)
    assert newt.facing == rules.FACING_NE
    assert newt.juncture == rules.JUNCTURE_AFTER_POST_ROTATE
    assert rules.can_unit_attack_now(state, "n1")
    assert rules.get_legal_moves(state, "n1") == []


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

    # All 4 moves exhausted, and as a Lizard (can_rotate_after_move=True), Tian Long
    # can still perform its post-move rotation (which locks once it attacks at AFTER_POST_ROTATE)
    assert rules.get_legal_moves(state, "tian") == []
    assert set(rules.get_legal_rotations(state, "tian")) == {0, 1, 2, 3, 4, 5}
    assert rules.rotate_unit(state, "tian", rules.FACING_N)
    assert unit.juncture == rules.JUNCTURE_AFTER_POST_ROTATE
    assert rules.record_juncture_attack(state, "tian")
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

    # Now test that if ALL reachable landing hexes beyond an occupied hex are ALSO occupied,
    # the rules layer prevents the flyer from moving onto the occupied hex at all.
    state2 = rules.GameState(arena_radius=4)
    # Place flyer at (0, 4) facing S (toward edge of radius-4 board so (0, 4) has no S neighbor;
    # Wait: let's place flyer at (0, -3) facing N (0), where (0, -4) is the northernmost board edge hex!
    # If (0, -4) is occupied by a blocker, and all valid hexes reachable in 1 step from (0, -4)
    # are also occupied, then flying onto (0, -4) has NO empty landing hex!)
    flyer2 = rules.Unit("flyer2", jaculus_spec, rules.TEAM_PLAYER, (0, -3), rules.FACING_N)
    state2.units["flyer2"] = flyer2
    # Block (0, -4) and every valid neighbor of (0, -4): (1, -4), (1, -3), (0, -3), (-1, -3)
    for idx, coord in enumerate([(0, -4), (1, -4), (1, -3), (-1, -3)]):
        state2.units[f"blk_{idx}"] = rules.Unit(
            f"blk_{idx}", rat_spec, rules.TEAM_ENEMY, coord, rules.FACING_S
        )
    # Because there is no empty hex reachable from (0, -4) on Step 2, (0, -4) MUST NOT be a legal move!
    assert (0, -4) not in rules.get_legal_moves(state2, "flyer2")
    assert not rules.move_unit(state2, "flyer2", (0, -4))


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


def test_directional_combat_front_flank_rear_vulnerability() -> None:
    """
    Attacking into a defender's FRONT deals base damage (ATK - DEF), while attacking
    into FLANK_LEFT / FLANK_RIGHT (+2) or REAR (+3) in defender.spec.vulnerable_arcs
    grants directional vulnerability bonus damage. Armored turtles ignore flank/rear bonuses.
    """
    state = rules.GameState(arena_radius=4)
    adder_spec = rules.get_creature_spec(rules.CLASS_SERPENT, 1)  # ATK 5, DEF 1, HP 8
    rat_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 1)     # ATK 4, DEF 1, HP 8
    war_turtle_spec = rules.get_creature_spec(rules.CLASS_TURTLE, 7)  # vulnerable_arcs = empty

    # Defender Rat at (0, 0) facing South (3):
    # - Hex (0, 1) [dir 3 = S] is in front of Rat (FORWARD = 0)
    # - Hex (1, -1) [dir 1 = NE] is to Rat's FLANK_LEFT ((1 - 3) % 6 == 4)
    # - Hex (0, -1) [dir 0 = N] is to Rat's REAR ((0 - 3) % 6 == 3)
    rat = rules.Unit("rat", rat_spec, rules.TEAM_ENEMY, (0, 0), rules.FACING_S)
    atk_front = rules.Unit("a_front", adder_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N)
    atk_flank = rules.Unit("a_flank", adder_spec, rules.TEAM_PLAYER, (1, -1), rules.FACING_SW)
    atk_rear = rules.Unit("a_rear", adder_spec, rules.TEAM_PLAYER, (0, -1), rules.FACING_S)

    state.units = {
        "rat": rat,
        "a_front": atk_front,
        "a_flank": atk_flank,
        "a_rear": atk_rear,
    }

    dmg_front, rel_front, vuln_front, _ = rules.compute_attack_damage(state, atk_front, rat)
    assert rel_front == rules.FORWARD
    assert vuln_front is False
    assert dmg_front == 5 - 1 == 4

    dmg_flank, rel_flank, vuln_flank, _ = rules.compute_attack_damage(state, atk_flank, rat)
    assert rel_flank == rules.FLANK_LEFT
    assert vuln_flank is True
    assert dmg_flank == (5 - 1) + 2 == 6

    dmg_rear, rel_rear, vuln_rear, _ = rules.compute_attack_damage(state, atk_rear, rat)
    assert rel_rear == rules.REAR
    assert vuln_rear is True
    assert dmg_rear == (5 - 1) + 3 == 7

    # War Turtle has vulnerable_arcs = frozenset(), so rear attacks get no vulnerability bonus
    wt = rules.Unit("wt", war_turtle_spec, rules.TEAM_ENEMY, (0, 0), rules.FACING_S)
    state.units["wt"] = wt
    _, rel_wt, vuln_wt, _ = rules.compute_attack_damage(state, atk_rear, wt)
    assert rel_wt == rules.REAR
    assert vuln_wt is False


def test_multi_arc_cleave_and_slaying_mammals_across_turns() -> None:
    """
    Player units can execute juncture attacks across turns (end_turn refreshes player units
    while enemy_ai_enabled is False) and slay all enemy mammals.
    """
    state = rules.GameState(arena_radius=4, enemy_ai_enabled=False)
    adder_spec = rules.get_creature_spec(rules.CLASS_SERPENT, 1)  # ATK 5, DEF 1, HP 8
    rat_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 1)     # ATK 4, DEF 1, HP 8

    adder = rules.Unit("adder", adder_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N)
    rat = rules.Unit("rat", rat_spec, rules.TEAM_ENEMY, (0, 0), rules.FACING_S)
    state.units = {"adder": adder, "rat": rat}

    # Turn 1: Adder attacks Rat from the front for 4 dmg (Rat HP: 8 -> 4)
    assert rules.get_legal_attack_targets(state, "adder") == [(0, 0)]
    res1 = rules.execute_attack(state, "adder", (0, 0))
    assert len(res1) == 1
    assert res1[0].damage == 4
    assert not res1[0].defender_slain
    assert rat.hp == 4
    assert not rules.can_unit_attack_now(state, "adder")
    assert not rules.are_all_enemies_slain(state)

    # End Turn -> advances to Turn 2, keeps active_team == TEAM_PLAYER, refreshes Adder
    assert rules.end_turn(state)
    assert state.turn_number == 2
    assert state.active_team == rules.TEAM_PLAYER
    assert rules.can_unit_attack_now(state, "adder")

    # Turn 2: Adder attacks Rat again for 4 dmg -> Rat is slain (HP 0, hex_pos None)
    # and the game automatically enters PHASE_EVOLUTION with Adder leveled up to Stage 2 (Viper).
    res2 = rules.execute_attack(state, "adder", (0, 0))
    assert len(res2) == 1
    assert res2[0].defender_slain is True
    assert not rat.is_alive
    assert rat.hex_pos is None
    assert rules.are_all_enemies_slain(state)
    assert state.phase == rules.PHASE_EVOLUTION
    assert len(state.evolution_records) == 1
    rec = state.evolution_records[0]
    assert rec.unit_id == "adder"
    assert rec.leveled_up is True
    assert rec.old_spec.stage == 1
    assert rec.new_spec.stage == 2
    assert adder.spec.stage == 2


def test_evolution_phase_pairs_vs_single_and_next_round_reinforcements() -> None:
    """
    Winning a battle enters PHASE_EVOLUTION:
      - Round 1 survivors (0 -> 1 battle survived) all level up from Stage 1 -> Stage 2
        (leveled_up=True, displayed in pairs with right arrow).
      - Confirming evolution returns all survivors to the Left Reserve Tray at full HP,
        grants REINFORCEMENTS_BASE (4) new Stage 1 reptiles, increments round_number to 2,
        spawns the Round 2 Mammal wave, and enters PHASE_SETUP.
      - After Round 2 battle: the 6 Round 1 veterans (1 -> 2 battles survived; Stage 3
        needs 3 battles) have leveled_up=False (displayed alone without arrow), while the
        4 Round 2 recruits (0 -> 1 battle survived) have leveled_up=True (Stage 1 -> Stage 2).
    """
    import random

    rng = random.Random(2026)
    state = rules.create_initial_game_state(game_mode=rules.GAME_MODE_BASIC, rng=rng)
    deploy_hexes = rules.get_valid_player_deploy_hexes(state)

    # Deploy all 6 initial Stage 1 reptiles and confirm deployment for Round 1
    r1_units = state.get_player_army_units()
    assert len(r1_units) == 6
    for idx, pu in enumerate(r1_units):
        assert rules.deploy_unit_to_hex(state, pu.unit_id, deploy_hexes[idx])
    assert rules.confirm_deployment(state)
    assert state.phase == rules.PHASE_BATTLE

    # Damage one player unit slightly, slay all Round 1 enemies, and trigger PHASE_EVOLUTION
    r1_units[0].hp -= 3
    for eu in list(state.units.values()):
        if eu.team == rules.TEAM_ENEMY:
            eu.hp = 0
            eu.hex_pos = None
    assert rules.complete_battle_to_evolution(state)
    assert state.phase == rules.PHASE_EVOLUTION
    assert len(state.evolution_records) == 6

    # All 6 Round 1 survivors went from 0 -> 1 battle survived, so all 6 leveled up (Stage 1 -> Stage 2)
    for rec in state.evolution_records:
        assert rec.battles_survived == 1
        assert rec.leveled_up is True
        assert rec.old_spec.stage == 1
        assert rec.new_spec.stage == 2

    # Confirm evolution -> transitions to Round 2 PHASE_SETUP with 6 + 4 = 10 reptiles in Reserve
    assert rules.confirm_evolution_phase(state, rng=rng)
    assert state.phase == rules.PHASE_SETUP
    assert state.round_number == 2
    assert len(state.evolution_records) == 0

    r2_units = state.get_player_army_units()
    assert len(r2_units) == 10
    for pu in r2_units:
        assert pu.in_reserve
        assert not pu.is_deployed
        assert pu.hp == pu.max_hp

    # Deploy all 10 reptiles and win Round 2
    for idx, pu in enumerate(r2_units):
        assert rules.deploy_unit_to_hex(state, pu.unit_id, deploy_hexes[idx])
    assert rules.confirm_deployment(state)
    assert state.phase == rules.PHASE_BATTLE

    for eu in list(state.units.values()):
        if eu.team == rules.TEAM_ENEMY:
            eu.hp = 0
            eu.hex_pos = None
    assert rules.complete_battle_to_evolution(state)
    assert state.phase == rules.PHASE_EVOLUTION
    assert len(state.evolution_records) == 10

    # First 6 veterans have battles_survived == 2 -> NOT leveled up (Stage 2 stays Stage 2, shown alone)
    veteran_recs = [r for r in state.evolution_records if r.battles_survived == 2]
    recruit_recs = [r for r in state.evolution_records if r.battles_survived == 1]
    assert len(veteran_recs) == 6
    assert len(recruit_recs) == 4

    for vrec in veteran_recs:
        assert vrec.leveled_up is False
        assert vrec.old_spec.stage == 2
        assert vrec.new_spec.stage == 2

    for rrec in recruit_recs:
        assert rrec.leveled_up is True
        assert rrec.old_spec.stage == 1
        assert rrec.new_spec.stage == 2


def test_enemy_mammal_ai_advances_attacks_and_ends_turn() -> None:
    """
    With `enemy_ai_enabled=True`, `end_turn` passes control to `TEAM_ENEMY`,
    `run_enemy_turn` advances and attacks using shared `rules.py` helpers,
    and finishes by returning control to `TEAM_PLAYER` on `turn_number == 2`.
    """
    state = rules.GameState(arena_radius=4, enemy_ai_enabled=True)
    pond_spec = rules.get_creature_spec(rules.CLASS_TURTLE, 1)  # HP 12, ATK 3, DEF 3
    rat_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 1)   # HP 8, ATK 4, DEF 1

    # Player Pond Turtle at (0, 1) facing N; Enemy Rat 2 hexes north at (0, -1) facing S
    turtle = rules.Unit("t1", pond_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N)
    rat = rules.Unit("r1", rat_spec, rules.TEAM_ENEMY, (0, -1), rules.FACING_S)
    state.units = {"t1": turtle, "r1": rat}

    # Player ends Turn 1 -> switches active_team to TEAM_ENEMY
    assert rules.end_turn(state)
    assert state.active_team == rules.TEAM_ENEMY
    assert state.turn_number == 1

    # Run the Enemy Mammal AI turn: Rat moves south from (0, -1) to (0, 0), attacks Turtle, and ends turn
    ai_records = rules.run_enemy_turn(state)
    action_types = [r.action_type for r in ai_records]
    assert "move" in action_types
    assert "attack" in action_types
    assert action_types[-1] == "end_turn"
    assert rat.hex_pos == (0, 0)
    assert turtle.hp < turtle.max_hp
    assert state.active_team == rules.TEAM_PLAYER
    assert state.turn_number == 2


def test_enemy_mammal_ai_flying_griffon_crosses_blocker_and_lands_empty() -> None:
    """
    A flying Enemy Mammal (Griffon, `num_rotate_moves=2, can_fly=True`) uses shared flight
    rules to fly over an intervening blocker onto an empty landing hex adjacent to a player
    reptile, attacks, and ends its turn on an empty hex (`can_end_turn(state) is True`).
    """
    state = rules.GameState(arena_radius=4, active_team=rules.TEAM_ENEMY, enemy_ai_enabled=True)
    griffon_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 5)  # num_rotate_moves=2, can_fly=True
    rat_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 1)
    turtle_spec = rules.get_creature_spec(rules.CLASS_TURTLE, 4)   # Tortoise HP 25

    # Griffon at (0, -1) facing S; allied Rat blocking (0, 0); player Tortoise at (0, 2)
    # Empty hex (0, 1) is directly between the blocker at (0, 0) and Tortoise at (0, 2)
    griffon = rules.Unit("g1", griffon_spec, rules.TEAM_ENEMY, (0, -1), rules.FACING_S)
    blocker = rules.Unit("b1", rat_spec, rules.TEAM_ENEMY, (0, 0), rules.FACING_S)
    # Prevent blocker from moving first by marking it acted
    state.ai_acted_unit_ids.add("b1")
    tortoise = rules.Unit("t1", turtle_spec, rules.TEAM_PLAYER, (0, 2), rules.FACING_N)
    state.units = {"g1": griffon, "b1": blocker, "t1": tortoise}

    records = rules.run_enemy_turn(state)
    assert griffon.hex_pos == (0, 1)
    assert not rules.is_unit_overlapping(state, "g1")
    assert tortoise.hp < tortoise.max_hp
    assert state.active_team == rules.TEAM_PLAYER
    assert any(r.action_type == "attack" for r in records)


def test_defeat_phase_when_all_player_reptiles_slain() -> None:
    """
    When the Enemy Mammal AI slays the last remaining player reptile, `are_all_players_slain`
    becomes True and `state.phase` transitions to `PHASE_DEFEAT`.
    """
    state = rules.GameState(arena_radius=4, active_team=rules.TEAM_ENEMY, enemy_ai_enabled=True)
    adder_spec = rules.get_creature_spec(rules.CLASS_SERPENT, 1)
    bear_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 3)  # ATK 7

    # 1-HP Adder at (0, 1) facing N; Bear at (0, 0) facing S
    adder = rules.Unit("a1", adder_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N, hp=1)
    bear = rules.Unit("bear", bear_spec, rules.TEAM_ENEMY, (0, 0), rules.FACING_S)
    state.units = {"a1": adder, "bear": bear}

    records = rules.run_enemy_turn(state)
    assert len(records) >= 1
    assert records[0].action_type == "attack"
    assert not adder.is_alive
    assert rules.are_all_players_slain(state)
    assert state.phase == rules.PHASE_DEFEAT


def test_splash_instructions_and_start_campaign_modes() -> None:
    """
    Phase 6:
      - `create_demo_battle()` / `create_splash_game_state()` starts in `PHASE_SPLASH`.
      - `open_instructions(state)` transitions `PHASE_SPLASH -> PHASE_INSTRUCTIONS`,
        and `close_instructions(state)` returns `PHASE_INSTRUCTIONS -> PHASE_SPLASH`.
      - `start_campaign(state, GAME_MODE_BASIC)` initializes Round 1 `PHASE_SETUP`
        in Basic mode (adaptive waves), with 6 Stage 1 reptiles in Reserve and 4 Rats on the map.
      - `start_campaign(state, GAME_MODE_ADVANCED)` initializes Round 1 `PHASE_SETUP`
        in Advanced mode (fixed wave curve `4 + 3*(R-1) + (R-1)^2`), and in Round 3 its budget
        is strictly `4 + 3*2 + 2^2 = 14` regardless of player casualties.
    """
    import random

    rng = random.Random(99)
    state = rules.create_demo_battle()
    assert state.phase == rules.PHASE_SPLASH

    # View Instructions flow
    assert rules.open_instructions(state)
    assert state.phase == rules.PHASE_INSTRUCTIONS
    assert not rules.open_instructions(state)  # Already in instructions
    assert rules.close_instructions(state)
    assert state.phase == rules.PHASE_SPLASH

    # Start Play Basic campaign
    assert rules.start_campaign(state, rules.GAME_MODE_BASIC, rng=rng)
    assert state.phase == rules.PHASE_SETUP
    assert state.game_mode == rules.GAME_MODE_BASIC
    assert state.round_number == 1
    assert state.wave_budget == 4
    assert len(state.get_player_army_units()) == rules.INITIAL_ARMY_SIZE

    # Return to splash and start Play Advanced campaign
    assert rules.return_to_splash(state)
    assert state.phase == rules.PHASE_SPLASH
    assert len(state.units) == 0

    assert rules.start_campaign(state, rules.GAME_MODE_ADVANCED, rng=rng)
    assert state.phase == rules.PHASE_SETUP
    assert state.game_mode == rules.GAME_MODE_ADVANCED
    assert state.round_number == 1
    assert state.wave_budget == 4

    # Even if only 1 Stage 1 reptile survives to Round 3, Advanced mode uses fixed schedule (14 pts)
    player_ids = [u.unit_id for u in state.get_player_army_units()]
    for uid in player_ids[1:]:
        del state.units[uid]
    state.round_number = 3
    assert rules.enemy_wave_budget(state) == 14


def test_bot_genome_serialization_and_rule_compliant_setup() -> None:
    """
    Phase 1 (Bot Evolution Plan):
      - `BotGenome` round-trips cleanly via `to_dict()` / `from_dict()` with all 13
        Mammal species genes, setup genes, targeting genes, and 1..4-ply lookahead genes.
      - `GeneticBotController.purchase_and_deploy_army` deploys legal armies on both
        `TEAM_ENEMY` (upper half) and `TEAM_PLAYER` (lower half) within budget.
    """
    import random
    import random
    import bots

    rng = random.Random(2027)
    genome = bots.create_random_genome(
        rng=rng, bot_id="bot_test_1", name="Test Strain", generation=2
    )
    assert len(genome.mammal_genes) == 13
    assert len(genome.setup.mammal_purchase_weights) == 13
    assert genome.lookahead.lookahead_ply in (1, 2, 3, 4)

    serialized = genome.to_dict()
    restored = bots.BotGenome.from_dict(serialized)
    assert restored.bot_id == "bot_test_1"
    assert restored.name == "Test Strain"
    assert restored.lookahead.lookahead_ply == genome.lookahead.lookahead_ply
    assert restored.targeting.strong_vs_weak_bias == genome.targeting.strong_vs_weak_bias

    state = rules.GameState(arena_radius=4, phase=rules.PHASE_SETUP, round_number=4)
    ctrl_north = bots.GeneticBotController(restored)
    ctrl_south = bots.LegacyBotController()

    north_units = ctrl_north.purchase_and_deploy_army(
        state, rules.TEAM_ENEMY, budget=18, rng=rng
    )
    south_units = ctrl_south.purchase_and_deploy_army(
        state, rules.TEAM_PLAYER, budget=18, rng=rng
    )

    assert 1 <= len(north_units) <= rules.MAX_ARMY_SIZE
    assert sum(u.spec.point_cost for u in north_units) <= 18
    assert all(
        u.hex_pos is not None and rules.is_valid_team_deploy_hex(state, rules.TEAM_ENEMY, u.hex_pos)
        for u in north_units
    )

    assert 1 <= len(south_units) <= rules.MAX_ARMY_SIZE
    assert sum(u.spec.point_cost for u in south_units) <= 18
    assert all(
        u.hex_pos is not None and rules.is_valid_team_deploy_hex(state, rules.TEAM_PLAYER, u.hex_pos)
        for u in south_units
    )


def test_bot_2ply_lookahead_avoids_suicidal_lunge_and_records_telemetry() -> None:
    """
    Phase 1 (Bot Evolution Plan):
      - A 1-ply bot (`lookahead_ply=1`) ignores the opponent's counter-turn response and
        steps from distance 3 to distance 2 directly in front of a powerful 1-move enemy
        Behemoth (entering its next-turn charge & maul range without being able to attack).
      - A 2-ply bot (`lookahead_ply=2`) with the exact same genes simulates the Behemoth's
        counter-turn response, sees the heavy counter-attack penalty in `DecisionTrace`,
        and avoids stepping into the Behemoth's 1-move kill zone.
    """
    import random
    import bots

    rat_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 1)       # 1 move, HP 8
    behemoth_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 7)  # 2 moves, ATK 13

    # Place our Rat at (0, -2) facing S, and an enemy Behemoth at (0, 1) facing N (distance 3).
    # Wait: Behemoth has num_rotate_moves=2, so from (0, 1) it can move 2 hexes N to (0, -1) and attack (0, -2)!
    # Let's use a 1-move Bear (stage 3, num_rotate_moves=1, ATK 7) at (0, 1) facing N!
    # From (0, 1), the 1-move Bear can move to (0, 0) and attack (0, -1),
    # bit CANNOT reach (0, -2) in 1 move!
    bear_spec = rules.get_creature_spec(rules.CLASS_MAMMAL, 3)      # 1 move, ATK 7

    # 1-Ply Bot test
    state_1ply = rules.GameState(
        arena_radius=4, phase=rules.PHASE_BATTLE, active_team=rules.TEAM_ENEMY
    )
    state_1ply.units["rat"] = rules.Unit(
        "rat", rat_spec, rules.TEAM_ENEMY, (0, -2), rules.FACING_S, hp=6
    )
    state_1ply.units["bear"] = rules.Unit(
        "bear", bear_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N
    )

    g1 = bots.BotGenome(
        bot_id="ply1",
        name="Myopic 1-Ply",
        lookahead=bots.LookaheadGenes(lookahead_ply=1, beam_width=3),
    )
    bot1 = bots.GeneticBotController(g1, record_telemetry=True)
    bot1.run_turn(state_1ply)
    # 1-ply bot blindly steps forward from (0, -2) to (0, -1) to reduce distance to Bear
    assert state_1ply.units["rat"].hex_pos == (0, -1)
    assert len(bot1.decision_traces) == 1
    assert bot1.decision_traces[0].chosen_enemy_response_penalty == 0.0

    # 2-Ply Bot test (identical genome except lookahead_ply=2)
    state_2ply = rules.GameState(
        arena_radius=4, phase=rules.PHASE_BATTLE, active_team=rules.TEAM_ENEMY
    )
    state_2ply.units["rat"] = rules.Unit(
        "rat", rat_spec, rules.TEAM_ENEMY, (0, -2), rules.FACING_S, hp=6
    )
    state_2ply.units["bear"] = rules.Unit(
        "bear", bear_spec, rules.TEAM_PLAYER, (0, 1), rules.FACING_N
    )

    g2 = bots.BotGenome(
        bot_id="ply2",
        name="Tactical 2-Ply",
        lookahead=bots.LookaheadGenes(lookahead_ply=2, beam_width=3),
    )
    bot2 = bots.GeneticBotController(g2, record_telemetry=True)
    bot2.run_turn(state_2ply)
    # 2-ply bot sees that moving to (0, -1) allows the Bear at (0, 1) to step to (0, 0) and kill the 6-HP Rat!
    assert state_2ply.units["rat"].hex_pos != (0, -1)
    assert len(bot2.decision_traces) == 1
    assert any(
        c.enemy_response_penalty > 100.0 for c in bot2.decision_traces[0].top_candidates
    )


def test_headless_skirmish_and_replay_log_instrumentation(tmp_path) -> None:
    """
    Phase 2 (Bot Evolution Plan):
      - `run_paired_skirmish` runs North/South-swapped Mammal-vs-Mammal matches
        completely UI-free and returns `BattleReplayLog` instances with initial setups,
        turn logs, and `DecisionTrace` lookahead diagnostics.
      - `BattleReplayLog` round-trips through `save_json` / `load_json` and formats
        human-readable diagnostic output.
    """
    import random
    import bots

    champs = bots.create_default_champion_genomes()
    champ_ctrl = bots.GeneticBotController(champs[0], record_telemetry=True)
    classic_ctrl = bots.LegacyBotController()

    log1, log2 = bots.run_paired_skirmish(
        champ_ctrl,
        classic_ctrl,
        round_number=2,
        budget=8,
        max_turns=18,
        seed=777,
        record_log=True,
    )

    assert len(log1.initial_units) >= 2
    assert len(log1.turn_logs) >= 1
    assert log1.north_bot_id == champ_ctrl.bot_id
    assert log2.south_bot_id == champ_ctrl.bot_id

    log_file = tmp_path / "replay_test.json"
    log1.save_json(str(log_file))
    loaded = bots.BattleReplayLog.load_json(str(log_file))
    assert loaded.match_id == log1.match_id
    assert len(loaded.turn_logs) == len(log1.turn_logs)

    report_txt = loaded.format_human_readable(max_turns=5)
    assert "BATTLE REPLAY LOG" in report_txt
    assert "INITIAL DEPLOYMENT:" in report_txt


def test_bot_stable_persistence_and_evolution_loop(tmp_path) -> None:
    """
    Phase 2 (Bot Evolution Plan):
      - `run_evolution` executes multi-generation tournaments, records ply-vs-ply
        telemetry, injects random-gene bots, and persists top bots to `bot_stable.json`.
      - Subsequent `run_evolution` calls load existing stable bots and increment
        `total_generations_run`.
    """
    import random
    import bots

    stable_file = tmp_path / "test_stable.json"
    log_dir = tmp_path / "logs"

    stable1, telemetry1 = bots.run_evolution(
        generations=1,
        population_size=4,
        random_injection_count=1,
        stable_path=str(stable_file),
        log_dir=str(log_dir),
        record_upsets=True,
        max_turns=12,
        seed=1234,
        verbose=False,
    )
    assert stable_file.exists()
    assert stable1.total_generations_run == 1
    assert len(stable1.bots) >= 4
    assert "PLY-VS-PLY WIN RATE TELEMETRY" in telemetry1.format_summary()

    # Resume evolution from the persisted stable
    stable2, _ = bots.run_evolution(
        generations=1,
        population_size=4,
        random_injection_count=1,
        stable_path=str(stable_file),
        log_dir=str(log_dir),
        record_upsets=True,
        max_turns=12,
        seed=5678,
        verbose=False,
    )
    assert stable2.total_generations_run == 2
    top_bot = stable2.get_top_bot(0)
    assert top_bot.games_vs_classic > 0
    assert top_bot.championship_streak >= 1

    # Verify deterministic gene-driven wave purchasing & champion gene dump
    ctrl = bots.GeneticBotController(top_bot)
    wave_a = [s.creature_id for s in ctrl.purchase_wave_specs(3, 14, rng=random.Random(1))]
    wave_b = [s.creature_id for s in ctrl.purchase_wave_specs(3, 14, rng=random.Random(999))]
    assert wave_a == wave_b

    dump_txt = bots.format_genome_dump(top_bot)
    assert "CHAMPION BOT GENE DUMP:" in dump_txt
    assert "X-PLY LOOKAHEAD GENES:" in dump_txt
    assert "ARMY SETUP & FORMATION GENES:" in dump_txt
    assert "TARGETING & COORDINATION GENES:" in dump_txt
    assert "SPECIES TACTICAL GENES" in dump_txt


def test_three_campaign_modes_basic_intermediate_and_advanced() -> None:
    """
    Phase 3 (Bot Evolution Plan):
      - `GAME_MODE_BASIC` uses Classic Bot (`active_bot_id == "legacy_classic"`) and
        adaptive wave point budgets.
      - `GAME_MODE_INTERMEDIATE` uses Classic Bot (`active_bot_id == "legacy_classic"`)
        and the fixed mathematical wave schedule (`4 + 3*(R-1) + (R-1)^2`).
      - `GAME_MODE_ADVANCED` loads the #1 Evolved Champion from `bot_stable.json`,
        uses the fixed mathematical wave schedule, deploys waves via evolved setup genes,
        and executes combat turns via `GeneticBotController`.
    """
    import random
    import bots

    rng = random.Random(2026)
    state = rules.create_demo_battle()

    # 1. Play Basic
    assert rules.start_campaign(state, rules.GAME_MODE_BASIC, rng=rng)
    assert state.game_mode == rules.GAME_MODE_BASIC
    assert state.active_bot_id == "legacy_classic"
    assert state.active_bot_name == "Classic Bot"

    # 2. Play Intermediate (Classic Bot + Fixed Schedule)
    assert rules.start_campaign(state, rules.GAME_MODE_INTERMEDIATE, rng=rng)
    assert state.game_mode == rules.GAME_MODE_INTERMEDIATE
    assert state.active_bot_id == "legacy_classic"
    assert state.active_bot_name == "Classic Bot"
    # Even if only 1 player reptile remains in Round 3, Intermediate uses fixed schedule (14 pts)
    p_ids = [u.unit_id for u in state.get_player_army_units()]
    for uid in p_ids[1:]:
        del state.units[uid]
    state.round_number = 3
    assert rules.enemy_wave_budget(state) == 14

    # 3. Play Advanced (Evolved Stable Bot + Fixed Schedule)
    assert rules.start_campaign(state, rules.GAME_MODE_ADVANCED, rng=rng)
    assert state.game_mode == rules.GAME_MODE_ADVANCED
    stable = bots.BotStable.load()
    expected_top = stable.get_top_bot(0)
    assert state.active_bot_id == expected_top.bot_id
    assert state.active_bot_name == expected_top.name
    assert state.active_bot_ply == expected_top.lookahead.lookahead_ply

    # Deploy all player units and confirm deployment, then run the Evolved Bot's Turn 1
    lower_hexes = rules.get_valid_player_deploy_hexes(state)
    for idx, pu in enumerate(state.get_player_army_units()):
        assert rules.deploy_unit_to_hex(state, pu.unit_id, lower_hexes[idx])
    assert rules.confirm_deployment(state)
    assert state.phase == rules.PHASE_BATTLE
    assert rules.end_turn(state)
    assert state.active_team == rules.TEAM_ENEMY

    records = rules.run_enemy_turn(state)
    assert len(records) >= 1
    assert records[-1].action_type == "end_turn"
    assert state.active_team == rules.TEAM_PLAYER
    assert state.turn_number == 2


def test_mammal_round_by_round_unlock_progression() -> None:
    """
    Mammals unlock progressively one per round in ascending power/cost order:
      - Round 1: Rat (1 pt)
      - Round 2: +Wolf (2 pts)
      - Round 3: +Bear (4 pts)
      - Round 4: +Lion (5 pts)
      - Round 5: +Griffon (7 pts)
      - ... up to Round 13: +Rajang (22 pts)
    Both Classic `rules.purchase_enemy_wave` and Evolved `GeneticBotController.purchase_wave_specs`
    strictly enforce the unlock gate so early waves never spawn heavy hitters like Griffons.
    """
    import random
    import bots
    from creatures import get_unlocked_mammals

    rng = random.Random(42)
    expected_names_by_round = [
        "Rat",
        "Wolf",
        "Bear",
        "Lion",
        "Griffon",
        "Owlbear",
        "Displacer Beast",
        "Dire Wolf",
        "Manticore",
        "Sabertooth",
        "Behemoth",
        "Hell Hound",
        "Rajang",
    ]
    for r, expected_newest in enumerate(expected_names_by_round, start=1):
        unlocked = get_unlocked_mammals(r)
        assert len(unlocked) == r
        assert unlocked[-1].name == expected_newest

    # Even with a generous 16-pt budget in Round 2, neither Classic nor Evolved bots
    # may purchase anything above Wolf (2 pts; unlock_round <= 2)
    classic_r2 = rules.purchase_enemy_wave(round_number=2, budget=16, rng=rng)
    assert all(s.unlock_round <= 2 and s.name in ("Rat", "Wolf") for s in classic_r2)

    champ_ctrl = bots.GeneticBotController(bots.BotStable.load().get_top_bot(0))
    evolved_r2 = champ_ctrl.purchase_wave_specs(round_number=2, budget=16, rng=rng)
    assert all(s.unlock_round <= 2 and s.name in ("Rat", "Wolf") for s in evolved_r2)

    # In Round 3, Bears (4 pts) unlock, but Lions (R4) and Griffons (R5) are still locked
    classic_r3 = rules.purchase_enemy_wave(round_number=3, budget=20, rng=rng)
    assert all(s.unlock_round <= 3 and s.name in ("Rat", "Wolf", "Bear") for s in classic_r3)
    evolved_r3 = champ_ctrl.purchase_wave_specs(round_number=3, budget=20, rng=rng)
    assert all(s.unlock_round <= 3 and s.name in ("Rat", "Wolf", "Bear") for s in evolved_r3)










