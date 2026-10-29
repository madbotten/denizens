"""
Denizens — Application Entry Point & Main Pygame Event Loop.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import pygame

import constants as C
import graphics
import hexgrid
import rules


class DenizensApp:
    """Initializes Pygame subsystems and runs the main event and render loop."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Denizens — Tactical Hex Battle")
        self.screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets = graphics.AssetCache(base_dir)
        self.camera = hexgrid.HexCamera(
            C.MAP_VIEWPORT_WIDTH, C.SCREEN_HEIGHT, viewport_x=C.MAP_VIEWPORT_X
        )
        self.renderer = graphics.BattleRenderer(self.screen, self.assets, self.camera)

        self.state: rules.GameState = rules.create_demo_battle()
        self.arena_hexes: List[hexgrid.HexCoord] = hexgrid.get_all_arena_hexes(
            self.state.arena_radius
        )

        self.hovered_hex: Optional[hexgrid.HexCoord] = None
        self.hovered_reserve_slot: Optional[int] = None

        # Drag-and-drop state for PHASE_SETUP
        self.dragging_unit_id: Optional[str] = None
        self.drag_start_pos: Tuple[int, int] = (0, 0)
        self.is_dragging: bool = False

        # Optional camera pan state (when ENABLE_CAMERA_ZOOM_PAN=True)
        self.is_panning: bool = False
        self.pan_start_pos: Tuple[int, int] = (0, 0)
        self.pan_drag_distance: float = 0.0

        self.status_message: str = (
            "Round 1 Setup: Drag all 6 reptiles from the Left Reserve onto the lower half of the battlefield, then click Confirm Deployment [Enter]."
        )

    def _report_attack_results(self, results: List[rules.AttackResult]) -> None:
        self.renderer.add_attack_callouts(results)
        parts: List[str] = []
        for res in results:
            arc_name = C.REL_DIR_NAMES[res.incoming_rel_dir]
            vuln_str = f" [{arc_name.upper()} VULNERABLE!]" if res.is_vulnerable else f" [{arc_name}]"
            slain_str = " — SLAIN!" if res.defender_slain else ""
            parts.append(
                f"{res.attacker_name} hit {res.defender_name} for {res.damage} dmg{vuln_str}{slain_str}"
            )
        summary = "  |  ".join(parts)
        if self.state.phase == C.PHASE_EVOLUTION:
            leveled_cnt = sum(1 for r in self.state.evolution_records if r.leveled_up)
            summary += (
                f"   *** VICTORY! Entering Evolution Phase ({len(self.state.evolution_records)} survivors, "
                f"{leveled_cnt} evolved) — Press Enter or click Confirm Evolution! ***"
            )
        elif rules.are_all_enemies_slain(self.state):
            summary += "   *** VICTORY! All Enemy Mammals Slain! ***"
        self.status_message = summary

    def _in_map_viewport(self, mx: int) -> bool:
        if self.state.phase == C.PHASE_SETUP:
            return C.MAP_VIEWPORT_X <= mx < C.SIDEBAR_X
        return 0 <= mx < C.SIDEBAR_X

    def handle_action(self, action_id: str) -> None:
        """Dispatch UI/keyboard intent to shared `rules.py` helpers."""
        sel_id = self.state.selected_unit_id
        if action_id == "rotate_ccw" and sel_id:
            if rules.rotate_unit_by_step(self.state, sel_id, -1):
                u = self.state.get_unit(sel_id)
                if u:
                    self.status_message = (
                        f"{u.spec.name} rotated CCW to {C.ABS_FACING_NAMES[u.facing]} ({u.juncture})."
                    )
            else:
                self.status_message = "Rotation CCW not legal right now (unit must be deployed and within rotation limit)."

        elif action_id == "rotate_cw" and sel_id:
            if rules.rotate_unit_by_step(self.state, sel_id, +1):
                u = self.state.get_unit(sel_id)
                if u:
                    self.status_message = (
                        f"{u.spec.name} rotated CW to {C.ABS_FACING_NAMES[u.facing]} ({u.juncture})."
                    )
            else:
                self.status_message = "Rotation CW not legal right now (unit must be deployed and within rotation limit)."

        elif action_id == "undeploy_unit" and sel_id:
            u = self.state.get_unit(sel_id)
            if u and rules.undeploy_unit_to_reserve(self.state, sel_id):
                self.status_message = f"Returned {u.spec.name} to the Army Reserve Tray."
            else:
                self.status_message = "Cannot return unit to reserve (only deployed player units during Setup Phase)."

        elif action_id == "confirm_deployment":
            if rules.confirm_deployment(self.state):
                self.camera.set_viewport(0, C.SIDEBAR_X)
                deployed_cnt = sum(
                    1 for u in self.state.get_player_army_units() if u.is_deployed
                )
                self.status_message = (
                    f"Battle Started! {deployed_cnt} reptiles deployed vs Enemy Wave ({self.state.wave_budget} pts). "
                    f"Turn 1 — Active Team: {self.state.active_team.upper()}."
                )
            else:
                unplaced_cnt = sum(
                    1 for u in self.state.get_player_army_units() if u.in_reserve
                )
                self.status_message = (
                    f"All reptiles in your army must be deployed before confirming! ({unplaced_cnt} still in Reserve)"
                )

        elif action_id == "juncture_attack" and sel_id:
            results = rules.execute_attack(self.state, sel_id)
            if results:
                self._report_attack_results(results)
            else:
                self.status_message = (
                    "Cannot attack right now (need an enemy in attack arc and an available attack at this juncture)."
                )

        elif action_id == "end_turn":
            if rules.end_turn(self.state):
                if self.state.phase == C.PHASE_EVOLUTION:
                    leveled_cnt = sum(
                        1 for r in self.state.evolution_records if r.leveled_up
                    )
                    self.status_message = (
                        f"Battle Won! Evolution Phase: {len(self.state.evolution_records)} survivors "
                        f"({leveled_cnt} evolved). Press Enter or click Confirm Evolution."
                    )
                else:
                    enemies_left = sum(
                        1
                        for u in self.state.units.values()
                        if u.is_alive and u.team == C.TEAM_ENEMY
                    )
                    self.status_message = (
                        f"Combat Turn {self.state.turn_number} — All player units refreshed! "
                        f"({enemies_left} enemy mammals remaining)"
                    )
            else:
                self.status_message = "Cannot end turn while a flying creature is overlapping another unit!"

        elif action_id == "confirm_evolution":
            if rules.confirm_evolution_phase(self.state):
                self.camera.set_viewport(C.MAP_VIEWPORT_X, C.MAP_VIEWPORT_WIDTH)
                army_cnt = len(self.state.get_player_army_units())
                enemy_cnt = sum(
                    1
                    for u in self.state.units.values()
                    if u.is_alive and u.team == C.TEAM_ENEMY
                )
                self.status_message = (
                    f"Round {self.state.round_number} Setup: {army_cnt} reptiles in Reserve "
                    f"(+{C.REINFORCEMENTS_BASE} reinforcements, max {C.MAX_ARMY_SIZE}) vs "
                    f"{enemy_cnt} Enemy Mammals ({self.state.wave_budget} pts). Drag all reptiles onto the lower half!"
                )
            else:
                self.status_message = "Confirm Evolution is only available during the Evolution Phase."

        elif action_id == "reset_camera":
            self.camera.reset()
            self.status_message = "Camera view reset to center."

    def handle_reserve_left_down(self, screen_pos: Tuple[int, int]) -> None:
        slot_idx = self.renderer.get_reserve_slot_at_pos(screen_pos)
        if slot_idx is None:
            return
        unit = self.state.get_unit_by_reserve_slot(slot_idx)
        if unit is None or not unit.in_reserve:
            return

        self.state.selected_unit_id = unit.unit_id
        self.status_message = (
            f"Selected {unit.spec.name} (Stage {unit.spec.stage}, {unit.spec.point_cost} pts) in Reserve."
        )

        if self.state.phase == C.PHASE_SETUP:
            self.dragging_unit_id = unit.unit_id
            self.drag_start_pos = screen_pos
            self.is_dragging = False

    def handle_map_left_down(self, screen_pos: Tuple[int, int]) -> None:
        clicked_hex = self.camera.pixel_to_hex(screen_pos[0], screen_pos[1])
        if not hexgrid.is_valid_hex(clicked_hex[0], clicked_hex[1], self.state.arena_radius):
            return

        if self.state.phase == C.PHASE_SETUP:
            units_here = self.state.get_units_at_hex(clicked_hex)
            if units_here:
                chosen = units_here[-1]
                self.state.selected_unit_id = chosen.unit_id
                self.status_message = (
                    f"Selected {chosen.spec.name} ({chosen.team.upper()}) at {clicked_hex}."
                )
                if chosen.team == C.TEAM_PLAYER:
                    self.dragging_unit_id = chosen.unit_id
                    self.drag_start_pos = screen_pos
                    self.is_dragging = False
                return

            # If a reserve player unit is currently selected and user clicks an empty lower-half hex, deploy it
            sel_unit = self.state.get_unit(self.state.selected_unit_id)
            if (
                sel_unit is not None
                and sel_unit.team == C.TEAM_PLAYER
                and sel_unit.in_reserve
                and rules.is_valid_player_deploy_hex(self.state, clicked_hex)
            ):
                if rules.deploy_unit_to_hex(self.state, sel_unit.unit_id, clicked_hex):
                    self.status_message = (
                        f"Deployed {sel_unit.spec.name} to {clicked_hex}."
                    )
            return

        # PHASE_BATTLE left click
        sel_id = self.state.selected_unit_id
        legal_atk_targets = (
            rules.get_legal_attack_targets(self.state, sel_id) if sel_id else []
        )
        if sel_id and clicked_hex in legal_atk_targets:
            results = rules.execute_attack(self.state, sel_id, clicked_hex)
            if results:
                self._report_attack_results(results)
                return

        legal_moves = rules.get_legal_moves(self.state, sel_id) if sel_id else []

        if sel_id and clicked_hex in legal_moves:
            if rules.move_unit(self.state, sel_id, clicked_hex):
                u = self.state.get_unit(sel_id)
                if u:
                    overlap_note = (
                        " [AIRBORNE OVER UNIT — MUST LAND]"
                        if rules.is_unit_overlapping(self.state, sel_id)
                        else ""
                    )
                    self.status_message = (
                        f"{u.spec.name} moved to {clicked_hex} "
                        f"(Step {u.current_step + 1}/{u.spec.num_rotate_moves}, {u.juncture}){overlap_note}"
                    )
                return

        units_here = self.state.get_units_at_hex(clicked_hex)
        if units_here:
            chosen = units_here[-1]
            self.state.selected_unit_id = chosen.unit_id
            self.status_message = (
                f"Selected {chosen.spec.name} ({chosen.team.upper()}) at {clicked_hex}."
            )

    def handle_left_mouse_up(self, screen_pos: Tuple[int, int]) -> None:
        if (
            self.state.phase == C.PHASE_SETUP
            and self.dragging_unit_id is not None
            and self.is_dragging
        ):
            mx, my = screen_pos
            unit = self.state.get_unit(self.dragging_unit_id)
            if unit is not None:
                if self._in_map_viewport(mx):
                    target_hex = self.camera.pixel_to_hex(mx, my)
                    if rules.deploy_unit_to_hex(self.state, unit.unit_id, target_hex):
                        self.state.selected_unit_id = unit.unit_id
                        self.status_message = (
                            f"Placed {unit.spec.name} on lower-half hex {target_hex}."
                        )
                    else:
                        self.status_message = (
                            "Invalid deployment hex! Place your reptiles in the green lower half of the battlefield."
                        )
                elif mx < C.MAP_VIEWPORT_X:
                    target_slot = self.renderer.get_reserve_slot_at_pos((mx, my))
                    if rules.undeploy_unit_to_reserve(
                        self.state, unit.unit_id, preferred_slot=target_slot
                    ):
                        self.status_message = (
                            f"Returned {unit.spec.name} to the Army Reserve Tray."
                        )

        self.dragging_unit_id = None
        self.is_dragging = False

    def handle_map_right_click(self, screen_pos: Tuple[int, int]) -> None:
        sel_id = self.state.selected_unit_id
        unit = self.state.get_unit(sel_id)
        if unit is None or unit.hex_pos is None:
            return
        clicked_hex = self.camera.pixel_to_hex(screen_pos[0], screen_pos[1])
        target_dir = hexgrid.hex_direction_between(unit.hex_pos, clicked_hex)
        if target_dir is not None:
            if rules.rotate_unit(self.state, unit.unit_id, target_dir):
                self.status_message = (
                    f"{unit.spec.name} rotated to {C.ABS_FACING_NAMES[unit.facing]} ({unit.juncture})."
                )
            else:
                self.status_message = (
                    f"Cannot rotate {unit.spec.name} to {C.ABS_FACING_NAMES[target_dir]} "
                    f"(max_rotate_per_move={unit.spec.max_rotate_per_move})."
                )

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key in (pygame.K_q, pygame.K_LEFT):
                    self.handle_action("rotate_ccw")
                elif event.key in (pygame.K_e, pygame.K_RIGHT):
                    self.handle_action("rotate_cw")
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self.state.phase == C.PHASE_SETUP:
                        self.handle_action("confirm_deployment")
                    elif self.state.phase == C.PHASE_EVOLUTION:
                        self.handle_action("confirm_evolution")
                elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                    if self.state.phase == C.PHASE_SETUP:
                        self.handle_action("undeploy_unit")
                elif event.key == pygame.K_f:
                    self.handle_action("juncture_attack")
                elif event.key == pygame.K_SPACE:
                    if self.state.phase == C.PHASE_BATTLE:
                        self.handle_action("end_turn")
                elif event.key in (pygame.K_c, pygame.K_HOME):
                    self.handle_action("reset_camera")
                elif C.ENABLE_CAMERA_ZOOM_PAN and event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_KP_PLUS,
                ):
                    cx = C.MAP_VIEWPORT_X + C.MAP_VIEWPORT_WIDTH * 0.5
                    self.camera.zoom_at(cx, C.SCREEN_HEIGHT * 0.5, 1.12)
                elif C.ENABLE_CAMERA_ZOOM_PAN and event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    cx = C.MAP_VIEWPORT_X + C.MAP_VIEWPORT_WIDTH * 0.5
                    self.camera.zoom_at(cx, C.SCREEN_HEIGHT * 0.5, 1.0 / 1.12)

            elif event.type == pygame.MOUSEWHEEL:
                if C.ENABLE_CAMERA_ZOOM_PAN:
                    mx, my = pygame.mouse.get_pos()
                    if self._in_map_viewport(mx):
                        factor = 1.10 if event.y > 0 else (1.0 / 1.10)
                        self.camera.zoom_at(mx, my, factor)
                else:
                    # Mouse wheel rotates the selected creature 60° CW / CCW
                    if event.y > 0:
                        self.handle_action("rotate_cw")
                    elif event.y < 0:
                        self.handle_action("rotate_ccw")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    clicked_button = False
                    for btn in self.renderer.buttons:
                        if btn.enabled and btn.rect.collidepoint(mx, my):
                            self.handle_action(btn.action_id)
                            clicked_button = True
                            break
                    if not clicked_button:
                        if self.state.phase == C.PHASE_EVOLUTION and mx < C.SIDEBAR_X:
                            uid = self.renderer.get_evolution_unit_at_pos(
                                self.state, (mx, my)
                            )
                            if uid is not None:
                                self.state.selected_unit_id = uid
                                u = self.state.get_unit(uid)
                                if u:
                                    self.status_message = (
                                        f"Inspecting survivor {u.spec.name} "
                                        f"(Stage {u.spec.stage}, {u.battles_survived} battles survived)."
                                    )
                        elif self.state.phase == C.PHASE_SETUP and mx < C.MAP_VIEWPORT_X:
                            self.handle_reserve_left_down((mx, my))
                        elif mx < C.SIDEBAR_X:
                            self.handle_map_left_down((mx, my))
                elif (
                    event.button == 3
                    and self._in_map_viewport(mx)
                    and not C.ENABLE_CAMERA_ZOOM_PAN
                ):
                    self.handle_map_right_click((mx, my))
                elif (
                    C.ENABLE_CAMERA_ZOOM_PAN
                    and event.button in (2, 3)
                    and self._in_map_viewport(mx)
                ):
                    self.is_panning = True
                    self.pan_start_pos = (mx, my)
                    self.pan_drag_distance = 0.0

            elif event.type == pygame.MOUSEBUTTONUP:
                mx, my = event.pos
                if event.button == 1:
                    self.handle_left_mouse_up((mx, my))
                elif (
                    C.ENABLE_CAMERA_ZOOM_PAN
                    and event.button in (2, 3)
                    and self.is_panning
                ):
                    self.is_panning = False
                    if (
                        event.button == 3
                        and self.pan_drag_distance < 6.0
                        and self._in_map_viewport(mx)
                    ):
                        self.handle_map_right_click((mx, my))

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self.dragging_unit_id is not None and not self.is_dragging:
                    dist = math.hypot(
                        mx - self.drag_start_pos[0], my - self.drag_start_pos[1]
                    )
                    if dist >= 6.0:
                        self.is_dragging = True

                if C.ENABLE_CAMERA_ZOOM_PAN and self.is_panning:
                    dx = mx - self.pan_start_pos[0]
                    dy = my - self.pan_start_pos[1]
                    self.pan_drag_distance += math.hypot(dx, dy)
                    self.camera.pan(dx, dy)
                    self.pan_start_pos = (mx, my)

                self.hovered_reserve_slot = self.renderer.get_reserve_slot_at_pos(
                    (mx, my)
                )

                if self._in_map_viewport(mx):
                    cand = self.camera.pixel_to_hex(mx, my)
                    self.hovered_hex = (
                        cand
                        if hexgrid.is_valid_hex(
                            cand[0], cand[1], self.state.arena_radius
                        )
                        else None
                    )
                else:
                    self.hovered_hex = None

        if C.ENABLE_CAMERA_ZOOM_PAN:
            keys = pygame.key.get_pressed()
            pan_speed = 10.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.camera.pan(0, pan_speed)
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.camera.pan(0, -pan_speed)
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.camera.pan(pan_speed, 0)
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.camera.pan(-pan_speed, 0)

        return True

    def run(self) -> None:
        running = True
        while running:
            running = self.process_events()
            mouse_pos = pygame.mouse.get_pos()
            self.renderer.draw_frame(
                self.state,
                self.arena_hexes,
                self.hovered_hex,
                self.status_message,
                hovered_reserve_slot=self.hovered_reserve_slot,
                dragging_unit_id=self.dragging_unit_id,
                is_dragging=self.is_dragging,
                mouse_pos=mouse_pos,
            )
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main() -> None:
    app = DenizensApp()
    app.run()


if __name__ == "__main__":
    main()
