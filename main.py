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
        self.camera = hexgrid.HexCamera(C.MAP_VIEWPORT_WIDTH, C.SCREEN_HEIGHT)
        self.renderer = graphics.BattleRenderer(self.screen, self.assets, self.camera)

        self.state: rules.GameState = rules.create_demo_battle()
        self.arena_hexes: List[hexgrid.HexCoord] = hexgrid.get_all_arena_hexes(
            self.state.arena_radius
        )

        self.hovered_hex: Optional[hexgrid.HexCoord] = None
        self.is_panning: bool = False
        self.pan_start_pos: Tuple[int, int] = (0, 0)
        self.pan_drag_distance: float = 0.0
        self.status_message: str = (
            "Select a unit. Q/E rotates within max_rotate_per_move; Click green hex to move."
        )

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
                self.status_message = "Rotation CCW not legal from current step/juncture."

        elif action_id == "rotate_cw" and sel_id:
            if rules.rotate_unit_by_step(self.state, sel_id, +1):
                u = self.state.get_unit(sel_id)
                if u:
                    self.status_message = (
                        f"{u.spec.name} rotated CW to {C.ABS_FACING_NAMES[u.facing]} ({u.juncture})."
                    )
            else:
                self.status_message = "Rotation CW not legal from current step/juncture."

        elif action_id == "juncture_attack" and sel_id:
            u = self.state.get_unit(sel_id)
            if u and rules.record_juncture_attack(self.state, sel_id):
                self.status_message = (
                    f"{u.spec.name} executed attack at {u.juncture} "
                    f"({u.attacks_used}/{u.spec.total_attacks_per_move} used)."
                )
            else:
                self.status_message = "Cannot attack at this juncture (already attacked or limit reached)."

        elif action_id == "prev_stage" and sel_id and C.DEBUG:
            if rules.cycle_unit_stage(self.state, sel_id, -1):
                u = self.state.get_unit(sel_id)
                if u:
                    self.status_message = f"Switched unit to Stage {u.spec.stage}: {u.spec.name}."

        elif action_id == "next_stage" and sel_id and C.DEBUG:
            if rules.cycle_unit_stage(self.state, sel_id, +1):
                u = self.state.get_unit(sel_id)
                if u:
                    self.status_message = f"Switched unit to Stage {u.spec.stage}: {u.spec.name}."

        elif action_id == "end_turn":
            if rules.end_turn(self.state):
                self.status_message = (
                    f"Turn {self.state.turn_number} — Active Team: {self.state.active_team.upper()}"
                )
            else:
                self.status_message = "Cannot end turn while a flying creature is overlapping another unit!"

        elif action_id == "reset_camera":
            self.camera.reset()
            self.status_message = "Camera view reset to center."

    def handle_map_left_click(self, screen_pos: Tuple[int, int]) -> None:
        clicked_hex = self.camera.pixel_to_hex(screen_pos[0], screen_pos[1])
        if not hexgrid.is_valid_hex(clicked_hex[0], clicked_hex[1], self.state.arena_radius):
            return

        sel_id = self.state.selected_unit_id
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

    def handle_map_right_click(self, screen_pos: Tuple[int, int]) -> None:
        sel_id = self.state.selected_unit_id
        unit = self.state.get_unit(sel_id)
        if unit is None:
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
                elif event.key == pygame.K_f:
                    self.handle_action("juncture_attack")
                elif event.key == pygame.K_SPACE:
                    self.handle_action("end_turn")
                elif event.key in (pygame.K_c, pygame.K_HOME):
                    self.handle_action("reset_camera")
                elif event.key == pygame.K_LEFTBRACKET and C.DEBUG:
                    self.handle_action("prev_stage")
                elif event.key == pygame.K_RIGHTBRACKET and C.DEBUG:
                    self.handle_action("next_stage")
                elif C.ENABLE_CAMERA_ZOOM_PAN and event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_KP_PLUS,
                ):
                    self.camera.zoom_at(C.MAP_VIEWPORT_WIDTH * 0.5, C.SCREEN_HEIGHT * 0.5, 1.12)
                elif C.ENABLE_CAMERA_ZOOM_PAN and event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    self.camera.zoom_at(
                        C.MAP_VIEWPORT_WIDTH * 0.5, C.SCREEN_HEIGHT * 0.5, 1.0 / 1.12
                    )

            elif event.type == pygame.MOUSEWHEEL:
                if C.ENABLE_CAMERA_ZOOM_PAN:
                    mx, my = pygame.mouse.get_pos()
                    if mx < C.MAP_VIEWPORT_WIDTH:
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
                    if mx >= C.MAP_VIEWPORT_WIDTH:
                        for btn in self.renderer.buttons:
                            if btn.enabled and btn.rect.collidepoint(mx, my):
                                self.handle_action(btn.action_id)
                                break
                    else:
                        self.handle_map_left_click((mx, my))
                elif event.button == 3 and mx < C.MAP_VIEWPORT_WIDTH and not C.ENABLE_CAMERA_ZOOM_PAN:
                    self.handle_map_right_click((mx, my))
                elif C.ENABLE_CAMERA_ZOOM_PAN and event.button in (2, 3) and mx < C.MAP_VIEWPORT_WIDTH:
                    self.is_panning = True
                    self.pan_start_pos = (mx, my)
                    self.pan_drag_distance = 0.0

            elif event.type == pygame.MOUSEBUTTONUP:
                mx, my = event.pos
                if C.ENABLE_CAMERA_ZOOM_PAN and event.button in (2, 3) and self.is_panning:
                    self.is_panning = False
                    if event.button == 3 and self.pan_drag_distance < 6.0 and mx < C.MAP_VIEWPORT_WIDTH:
                        self.handle_map_right_click((mx, my))

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if C.ENABLE_CAMERA_ZOOM_PAN and self.is_panning:
                    dx = mx - self.pan_start_pos[0]
                    dy = my - self.pan_start_pos[1]
                    self.pan_drag_distance += math.hypot(dx, dy)
                    self.camera.pan(dx, dy)
                    self.pan_start_pos = (mx, my)

                if mx < C.MAP_VIEWPORT_WIDTH:
                    cand = self.camera.pixel_to_hex(mx, my)
                    self.hovered_hex = (
                        cand if hexgrid.is_valid_hex(cand[0], cand[1], self.state.arena_radius) else None
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
            self.renderer.draw_frame(
                self.state, self.arena_hexes, self.hovered_hex, self.status_message
            )
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main() -> None:
    app = DenizensApp()
    app.run()


if __name__ == "__main__":
    main()
