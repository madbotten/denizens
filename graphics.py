"""
Pygame asset caching, hex battlefield rendering, and gothic inspector UI for Denizens.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

import constants as C
import creatures
import hexgrid
import rules


# ============================================================================
# Asset Cache (Top-Down RGBA Sprites & Framed Gothic Portraits)
# ============================================================================

class AssetCache:
    """Loads and caches rotated top-down creature sprites and scaled portraits."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self._raw_topdown: Dict[str, pygame.Surface] = {}
        self._rotated_base: Dict[Tuple[str, int], pygame.Surface] = {}
        self._scaled_topdown: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self._portraits: Dict[Tuple[str, int], pygame.Surface] = {}

    def _load_surface(self, rel_path: str) -> pygame.Surface:
        full_path = os.path.join(self.base_dir, rel_path)
        if not os.path.exists(full_path):
            surf = pygame.Surface((256, 256), pygame.SRCALPHA)
            pygame.draw.circle(surf, (180, 60, 60, 220), (128, 128), 96)
            return surf
        return pygame.image.load(full_path).convert_alpha()

    def get_topdown_sprite(self, rel_path: str, facing: int, pixel_size: int) -> pygame.Surface:
        """
        Return the creature's top-down sprite rotated to `facing` (0..5) and scaled to `pixel_size`.
        Facing 0 is North (0°); each clockwise step rotates by -60° in Pygame coordinates.
        """
        facing = facing % 6
        quant_size = max(24, int(round(pixel_size / 2.0) * 2))
        cache_key = (rel_path, facing, quant_size)
        cached = self._scaled_topdown.get(cache_key)
        if cached is not None:
            return cached

        rot_key = (rel_path, facing)
        rot_base = self._rotated_base.get(rot_key)
        if rot_base is None:
            raw = self._raw_topdown.get(rel_path)
            if raw is None:
                loaded = self._load_surface(rel_path)
                raw = pygame.transform.smoothscale(loaded, (512, 512))
                self._raw_topdown[rel_path] = raw
            if facing == 0:
                rot_base = raw
            else:
                angle_deg = -60.0 * facing
                rot_base = pygame.transform.rotozoom(raw, angle_deg, 1.0)
            self._rotated_base[rot_key] = rot_base

        bw, bh = rot_base.get_size()
        scale_factor = quant_size / 512.0
        target_w = max(16, int(round(bw * scale_factor)))
        target_h = max(16, int(round(bh * scale_factor)))
        scaled = pygame.transform.smoothscale(rot_base, (target_w, target_h))

        if len(self._scaled_topdown) > 600:
            self._scaled_topdown.clear()
        self._scaled_topdown[cache_key] = scaled
        return scaled

    def get_portrait(self, rel_path: str, size: int) -> pygame.Surface:
        cache_key = (rel_path, size)
        cached = self._portraits.get(cache_key)
        if cached is not None:
            return cached
        raw = self._load_surface(rel_path)
        scaled = pygame.transform.smoothscale(raw, (size, size))
        self._portraits[cache_key] = scaled
        return scaled


# ============================================================================
# UI Button Model
# ============================================================================

@dataclass
class UIButton:
    rect: pygame.Rect
    label: str
    action_id: str
    enabled: bool = True
    accent_color: Tuple[int, int, int] = (75, 92, 115)


# ============================================================================
# Battle Renderer (Hex Map Viewport & Gothic Inspector Sidebar)
# ============================================================================

class BattleRenderer:
    """Renders the 61-hex arena, unit sprites, tactical overlays, and sidebar inspector."""

    def __init__(self, screen: pygame.Surface, assets: AssetCache, camera: hexgrid.HexCamera) -> None:
        self.screen = screen
        self.assets = assets
        self.camera = camera

        self.font_title = pygame.font.SysFont("georgia", 26, bold=True)
        self.font_subtitle = pygame.font.SysFont("georgia", 18, bold=True)
        self.font_body = pygame.font.SysFont("menlo", 14)
        self.font_small = pygame.font.SysFont("menlo", 12)
        self.font_badge = pygame.font.SysFont("menlo", 13, bold=True)

        self.buttons: List[UIButton] = []

    def draw_frame(
        self,
        state: rules.GameState,
        arena_hexes: List[hexgrid.HexCoord],
        hovered_hex: Optional[hexgrid.HexCoord],
        status_message: str,
    ) -> None:
        self.draw_map_viewport(state, arena_hexes, hovered_hex, status_message)
        self.draw_sidebar(state, hovered_hex)

    def draw_map_viewport(
        self,
        state: rules.GameState,
        arena_hexes: List[hexgrid.HexCoord],
        hovered_hex: Optional[hexgrid.HexCoord],
        status_message: str,
    ) -> None:
        viewport_rect = pygame.Rect(0, 0, C.MAP_VIEWPORT_WIDTH, C.SCREEN_HEIGHT)
        self.screen.set_clip(viewport_rect)
        self.screen.fill(C.COLOR_BG_VOID, viewport_rect)

        sel_unit = state.get_unit(state.selected_unit_id)
        legal_moves: List[hexgrid.HexCoord] = (
            rules.get_legal_moves(state, sel_unit.unit_id) if sel_unit else []
        )
        legal_rotations: List[int] = (
            rules.get_legal_rotations(state, sel_unit.unit_id) if sel_unit else []
        )
        attack_arc_hexes: List[hexgrid.HexCoord] = (
            rules.get_attack_arc_hexes(state, sel_unit.unit_id) if sel_unit else []
        )

        # 1. Draw all 61 arena hexes
        for coord in arena_hexes:
            q, r = coord
            pts = self.camera.hex_polygon_points(coord, scale=0.98)
            if coord == (0, 0):
                fill_col = C.COLOR_HEX_CENTER
            elif (q - r) % 2 == 0:
                fill_col = C.COLOR_HEX_FILL_A
            else:
                fill_col = C.COLOR_HEX_FILL_B

            pygame.draw.polygon(self.screen, fill_col, pts)
            pygame.draw.polygon(self.screen, C.COLOR_HEX_BORDER, pts, width=2)

            if coord in attack_arc_hexes:
                inner_pts = self.camera.hex_polygon_points(coord, scale=0.88)
                pygame.draw.polygon(self.screen, C.COLOR_ATTACK_ARC, inner_pts, width=2)

            if coord in legal_moves and sel_unit is not None:
                occupant = state.get_unit_at_hex(coord, exclude_unit_id=sel_unit.unit_id)
                hl_col = C.COLOR_LEGAL_FLY_OVER if occupant is not None else C.COLOR_LEGAL_MOVE
                inner_pts = self.camera.hex_polygon_points(coord, scale=0.84)
                pygame.draw.polygon(self.screen, hl_col, inner_pts, width=4)

                abs_dir = hexgrid.hex_direction_between(sel_unit.hex_pos, coord)
                if abs_dir is not None and self.camera.zoom >= 0.70:
                    rel_dir = hexgrid.relative_direction(sel_unit.facing, abs_dir)
                    tag = "FLY OVER" if occupant is not None else C.REL_DIR_NAMES[rel_dir]
                    lbl = self.font_small.render(tag, True, hl_col)
                    cx, cy = self.camera.hex_to_pixel(coord)
                    self.screen.blit(
                        lbl,
                        (cx - lbl.get_width() * 0.5, cy + self.camera.hex_radius * 0.42),
                    )

            if coord == hovered_hex:
                hover_pts = self.camera.hex_polygon_points(coord, scale=0.95)
                pygame.draw.polygon(self.screen, C.COLOR_HEX_HOVER, hover_pts, width=2)

            if C.DEBUG and self.camera.zoom >= 0.65:
                cx, cy = self.camera.hex_to_pixel(coord)
                coord_surf = self.font_small.render(f"{q},{r}", True, (82, 92, 108))
                self.screen.blit(
                    coord_surf,
                    (cx - coord_surf.get_width() * 0.5, cy - self.camera.hex_radius * 0.74),
                )

        # 2. Highlight selected unit's hex and its directional edges
        if sel_unit is not None:
            sel_pts = self.camera.hex_polygon_points(sel_unit.hex_pos, scale=0.95)
            pygame.draw.polygon(self.screen, C.COLOR_SELECTED_HEX, sel_pts, width=3)

            for vuln_rel in sel_unit.spec.vulnerable_arcs:
                abs_edge = hexgrid.absolute_direction(sel_unit.facing, vuln_rel)
                p1, p2 = self.camera.hex_edge_endpoints(sel_unit.hex_pos, abs_edge, scale=0.88)
                pygame.draw.line(self.screen, C.COLOR_VULNERABLE_EDGE, p1, p2, width=4)

            for atk_rel in sel_unit.spec.attack_arcs:
                abs_edge = hexgrid.absolute_direction(sel_unit.facing, atk_rel)
                p1, p2 = self.camera.hex_edge_endpoints(sel_unit.hex_pos, abs_edge, scale=0.88)
                col = C.COLOR_FRONT_EDGE if atk_rel == C.FORWARD else C.COLOR_ATTACK_ARC
                pygame.draw.line(self.screen, col, p1, p2, width=6)

            for rot_facing in legal_rotations:
                if rot_facing == sel_unit.facing:
                    continue
                p1, p2 = self.camera.hex_edge_endpoints(sel_unit.hex_pos, rot_facing, scale=0.76)
                mx = 0.5 * (p1[0] + p2[0])
                my = 0.5 * (p1[1] + p2[1])
                pygame.draw.circle(self.screen, C.COLOR_TEXT_GOLD, (int(mx), int(my)), 5)

        # 3. Draw units on the board
        sprite_px = int(self.camera.hex_radius * 1.62)
        for coord in arena_hexes:
            units_at_coord = state.get_units_at_hex(coord)
            if not units_at_coord:
                continue
            cx, cy = self.camera.hex_to_pixel(coord)
            for idx, unit in enumerate(units_at_coord):
                offset_x = 12 * idx
                offset_y = -14 * idx
                ux = cx + offset_x
                uy = cy + offset_y

                ring_col = C.COLOR_PLAYER_RING if unit.team == C.TEAM_PLAYER else C.COLOR_ENEMY_RING
                ring_radius = int(self.camera.hex_radius * 0.64)
                pygame.draw.circle(
                    self.screen, ring_col, (int(ux), int(uy)), ring_radius, width=3
                )

                facing_angle_rad = math.radians(60.0 * unit.facing - 90.0)
                tip_dist = ring_radius + 7
                tip_x = ux + math.cos(facing_angle_rad) * tip_dist
                tip_y = uy + math.sin(facing_angle_rad) * tip_dist
                pygame.draw.circle(self.screen, C.COLOR_FRONT_EDGE, (int(tip_x), int(tip_y)), 5)

                sprite = self.assets.get_topdown_sprite(
                    unit.spec.topdown_path, unit.facing, sprite_px
                )
                self.screen.blit(
                    sprite,
                    (ux - sprite.get_width() * 0.5, uy - sprite.get_height() * 0.5),
                )

                if self.camera.zoom >= 0.65:
                    lbl = self.font_small.render(unit.spec.name, True, C.COLOR_TEXT_MAIN)
                    bg_rect = pygame.Rect(
                        int(ux - lbl.get_width() * 0.5 - 4),
                        int(uy + self.camera.hex_radius * 0.52),
                        lbl.get_width() + 8,
                        lbl.get_height() + 2,
                    )
                    pygame.draw.rect(self.screen, (16, 18, 24), bg_rect, border_radius=4)
                    pygame.draw.rect(self.screen, ring_col, bg_rect, width=1, border_radius=4)
                    self.screen.blit(lbl, (bg_rect.x + 4, bg_rect.y + 1))

        self.draw_viewport_hud(state, status_message)
        self.screen.set_clip(None)

    def draw_viewport_hud(self, state: rules.GameState, status_message: str) -> None:
        hud_rect = pygame.Rect(18, 16, 860, 74)
        pygame.draw.rect(self.screen, (20, 24, 30), hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, C.COLOR_PANEL_BORDER, hud_rect, width=2, border_radius=8)

        team_col = (
            C.COLOR_TEXT_GREEN if state.active_team == C.TEAM_PLAYER else C.COLOR_TEXT_RED
        )
        header = self.font_subtitle.render(
            f"TURN {state.turn_number}  •  ACTIVE TEAM: {state.active_team.upper()}",
            True,
            team_col,
        )
        self.screen.blit(header, (32, 24))

        controls_hint = self.font_small.render(
            "L-Click: Select/Move  |  Wheel / Left-Right / Q-E / R-Click Adj Hex: Rotate  |  F: Attack  |  [/]: Stage  |  Space: End Turn",
            True,
            C.COLOR_TEXT_MUTED,
        )
        self.screen.blit(controls_hint, (32, 50))

        status_rect = pygame.Rect(18, C.SCREEN_HEIGHT - 48, C.MAP_VIEWPORT_WIDTH - 36, 34)
        pygame.draw.rect(self.screen, (20, 24, 30), status_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 70, 84), status_rect, width=1, border_radius=6)
        status_surf = self.font_body.render(status_message, True, C.COLOR_TEXT_GOLD)
        self.screen.blit(status_surf, (30, C.SCREEN_HEIGHT - 40))

    def draw_sidebar(
        self, state: rules.GameState, hovered_hex: Optional[hexgrid.HexCoord]
    ) -> None:
        panel_rect = pygame.Rect(C.MAP_VIEWPORT_WIDTH, 0, C.SIDEBAR_WIDTH, C.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, C.COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(
            self.screen,
            C.COLOR_PANEL_BORDER,
            (C.MAP_VIEWPORT_WIDTH, 0),
            (C.MAP_VIEWPORT_WIDTH, C.SCREEN_HEIGHT),
            width=3,
        )

        unit = state.get_unit(state.selected_unit_id)
        if unit is None and hovered_hex is not None:
            unit = state.get_unit_at_hex(hovered_hex)

        x0 = C.MAP_VIEWPORT_WIDTH + 24
        y = 20

        title_surf = self.font_title.render("DENIZENS INSPECTOR", True, C.COLOR_TEXT_GOLD)
        self.screen.blit(title_surf, (x0, y))
        y += 38

        if unit is None:
            empty_msg = self.font_body.render(
                "Click a creature on the hex arena to inspect.", True, C.COLOR_TEXT_MUTED
            )
            self.screen.blit(empty_msg, (x0, y + 40))
            return

        spec = unit.spec

        # 1. Framed Gothic Portrait
        portrait_size = 380
        px = C.MAP_VIEWPORT_WIDTH + (C.SIDEBAR_WIDTH - portrait_size) // 2
        portrait_surf = self.assets.get_portrait(spec.portrait_path, portrait_size)
        self.screen.blit(portrait_surf, (px, y))
        pygame.draw.rect(
            self.screen,
            C.COLOR_PANEL_BORDER,
            pygame.Rect(px, y, portrait_size, portrait_size),
            width=2,
        )
        y += portrait_size + 14

        # 2. Creature Name, Class, Stage & Evolution Info
        team_badge_col = C.COLOR_TEXT_GREEN if unit.team == C.TEAM_PLAYER else C.COLOR_TEXT_RED
        if spec.creature_class in C.PLAYER_CLASSES:
            stage_label = f"Stage {spec.stage}/8 {spec.creature_class.title()}"
            next_req = (
                creatures.evolution_battles_required(spec.stage + 1)
                if spec.stage < 8
                else creatures.evolution_battles_required(8)
            )
            evo_str = (
                f"Battles Survived: {unit.battles_survived}  (Next Stage at {next_req})"
                if spec.stage < 8
                else f"Battles Survived: {unit.battles_survived}  (MAX STAGE 8 APEX)"
            )
        else:
            stage_label = f"Enemy Mammal #{spec.stage}/13 (Non-Evolving)"
            evo_str = "Enemy Opposing Force"

        name_surf = self.font_title.render(spec.name, True, C.COLOR_TEXT_MAIN)
        self.screen.blit(name_surf, (x0, y))
        badge_surf = self.font_badge.render(
            f"[{unit.team.upper()} • {stage_label}]", True, team_badge_col
        )
        self.screen.blit(badge_surf, (x0 + name_surf.get_width() + 14, y + 6))
        y += 32

        evo_surf = self.font_body.render(evo_str, True, C.COLOR_TEXT_GOLD)
        self.screen.blit(evo_surf, (x0, y))
        y += 26

        # 3. Turn Step & Juncture Status Card
        card_w = C.SIDEBAR_WIDTH - 48
        step_card = pygame.Rect(x0, y, card_w, 128)
        pygame.draw.rect(self.screen, C.COLOR_CARD_BG, step_card, border_radius=8)
        pygame.draw.rect(self.screen, C.COLOR_PANEL_BORDER, step_card, width=1, border_radius=8)

        can_atk = rules.can_unit_attack_now(state, unit.unit_id)
        legal_rots = rules.get_legal_rotations(state, unit.unit_id)
        legal_mvs = rules.get_legal_moves(state, unit.unit_id)

        lines_step = [
            (
                f"Step: {min(unit.current_step + 1, spec.num_rotate_moves)}/{spec.num_rotate_moves}  •  "
                f"Moves Left: {unit.moves_remaining}  •  Juncture: {unit.juncture}",
                C.COLOR_TEXT_CYAN,
            ),
            (
                f"Facing: {C.ABS_FACING_NAMES[unit.facing]}  •  "
                f"Step-Start: {C.ABS_FACING_NAMES[unit.step_start_facing]}  •  "
                f"Legal Rotations: {len(legal_rots)}",
                C.COLOR_TEXT_MAIN,
            ),
            (
                f"Attacks Used: {unit.attacks_used}/{spec.total_attacks_per_move}  •  "
                f"Can Attack ({unit.juncture}): {'YES' if can_atk else 'NO'}",
                C.COLOR_TEXT_GREEN if can_atk else C.COLOR_TEXT_MUTED,
            ),
            (
                f"Hex Position: {unit.hex_pos}  •  Legal Destination Hexes: {len(legal_mvs)}",
                C.COLOR_TEXT_MAIN,
            ),
        ]
        cy_line = y + 10
        max_line_w = card_w - 24
        for text, col in lines_step:
            surf = self.font_body.render(text, True, col)
            if surf.get_width() > max_line_w:
                surf = self.font_small.render(text, True, col)
            self.screen.blit(surf, (x0 + 12, cy_line))
            cy_line += 28
        y += 138

        # 4. Creature Attributes & Capabilities Card
        attr_card = pygame.Rect(x0, y, card_w, 198)
        pygame.draw.rect(self.screen, C.COLOR_CARD_BG, attr_card, border_radius=8)
        pygame.draw.rect(self.screen, (62, 70, 84), attr_card, width=1, border_radius=8)

        move_dir_names = ", ".join(C.REL_DIR_NAMES[d] for d in sorted(spec.legal_move_dirs))
        atk_arc_names = ", ".join(C.REL_DIR_NAMES[d] for d in sorted(spec.attack_arcs))
        vuln_arc_names = (
            ", ".join(C.REL_DIR_NAMES[d] for d in sorted(spec.vulnerable_arcs))
            if spec.vulnerable_arcs
            else "None (Armored All Sides)"
        )
        powers_str = ", ".join(spec.special_powers) if spec.special_powers else "None (Basic Form)"

        attr_lines = [
            (
                f"Combat Stats:  HP {unit.hp}/{spec.max_hp}   •   ATK {spec.attack}   •   DEF {spec.defense}",
                C.COLOR_TEXT_GOLD,
            ),
            (
                f"Movement Specs:  num_rotate_moves={spec.num_rotate_moves}   •   max_rotate_per_move={spec.max_rotate_per_move}",
                C.COLOR_TEXT_MAIN,
            ),
            (
                f"Attack & Flight: total_attacks_per_move={spec.total_attacks_per_move}   •   can_fly={'YES' if spec.can_fly else 'NO'}",
                C.COLOR_TEXT_CYAN if spec.can_fly else C.COLOR_TEXT_MAIN,
            ),
            (f"Legal Move Dirs: {move_dir_names}", C.COLOR_TEXT_GREEN),
            (f"Attack Arcs:     {atk_arc_names}", C.COLOR_TEXT_MAIN),
            (f"Vulnerable Arcs: {vuln_arc_names}", C.COLOR_TEXT_RED),
            (f"Special Powers:  {powers_str}", C.COLOR_TEXT_GOLD),
        ]
        cy_line = y + 10
        for text, col in attr_lines:
            surf = self.font_small.render(text, True, col)
            self.screen.blit(surf, (x0 + 12, cy_line))
            cy_line += 26
        y += 210

        # 5. Interactive Action & Debug Buttons
        self.buttons.clear()
        btn_w = (card_w - 12) // 2
        btn_h = 40

        is_active_team = unit.team == state.active_team
        can_rot_ccw = is_active_team and ((unit.facing - 1) % 6 in legal_rots)
        can_rot_cw = is_active_team and ((unit.facing + 1) % 6 in legal_rots)

        row1_y = y
        self.buttons.append(
            UIButton(
                pygame.Rect(x0, row1_y, btn_w, btn_h),
                "Rotate CCW [Left / Q]",
                "rotate_ccw",
                enabled=can_rot_ccw,
                accent_color=(65, 115, 155),
            )
        )
        self.buttons.append(
            UIButton(
                pygame.Rect(x0 + btn_w + 12, row1_y, btn_w, btn_h),
                "Rotate CW [Right / E]",
                "rotate_cw",
                enabled=can_rot_cw,
                accent_color=(65, 115, 155),
            )
        )

        row2_y = row1_y + btn_h + 10
        self.buttons.append(
            UIButton(
                pygame.Rect(x0, row2_y, btn_w, btn_h),
                f"Attack ({unit.juncture[:10]}) [F]",
                "juncture_attack",
                enabled=can_atk,
                accent_color=(165, 70, 65),
            )
        )
        self.buttons.append(
            UIButton(
                pygame.Rect(x0 + btn_w + 12, row2_y, btn_w, btn_h),
                "End Turn [Space]",
                "end_turn",
                enabled=rules.can_end_turn(state),
                accent_color=(70, 140, 90),
            )
        )

        row3_y = row2_y + btn_h + 10
        if C.DEBUG:
            if C.ENABLE_CAMERA_ZOOM_PAN:
                third_w = (card_w - 24) // 3
                self.buttons.append(
                    UIButton(
                        pygame.Rect(x0, row3_y, third_w, btn_h),
                        "Prev Stage [[]",
                        "prev_stage",
                        enabled=True,
                        accent_color=(115, 95, 60),
                    )
                )
                self.buttons.append(
                    UIButton(
                        pygame.Rect(x0 + third_w + 12, row3_y, third_w, btn_h),
                        "Next Stage []]",
                        "next_stage",
                        enabled=True,
                        accent_color=(115, 95, 60),
                    )
                )
                self.buttons.append(
                    UIButton(
                        pygame.Rect(x0 + (third_w + 12) * 2, row3_y, third_w, btn_h),
                        "Reset Cam [C]",
                        "reset_camera",
                        enabled=True,
                        accent_color=(80, 90, 110),
                    )
                )
            else:
                self.buttons.append(
                    UIButton(
                        pygame.Rect(x0, row3_y, btn_w, btn_h),
                        "Prev Stage [[]",
                        "prev_stage",
                        enabled=True,
                        accent_color=(115, 95, 60),
                    )
                )
                self.buttons.append(
                    UIButton(
                        pygame.Rect(x0 + btn_w + 12, row3_y, btn_w, btn_h),
                        "Next Stage []]",
                        "next_stage",
                        enabled=True,
                        accent_color=(115, 95, 60),
                    )
                )

        mx, my = pygame.mouse.get_pos()
        for btn in self.buttons:
            hovered = btn.rect.collidepoint(mx, my) and btn.enabled
            if not btn.enabled:
                bg = (34, 38, 44)
                fg = (95, 100, 110)
                border = (52, 56, 64)
            elif hovered:
                bg = tuple(min(255, c + 28) for c in btn.accent_color)
                fg = (255, 255, 255)
                border = C.COLOR_TEXT_GOLD
            else:
                bg = btn.accent_color
                fg = C.COLOR_TEXT_MAIN
                border = (120, 130, 148)

            pygame.draw.rect(self.screen, bg, btn.rect, border_radius=6)
            pygame.draw.rect(self.screen, border, btn.rect, width=2, border_radius=6)
            lbl = self.font_badge.render(btn.label, True, fg)
            self.screen.blit(
                lbl,
                (
                    btn.rect.centerx - lbl.get_width() * 0.5,
                    btn.rect.centery - lbl.get_height() * 0.5,
                ),
            )

        footer_y = C.SCREEN_HEIGHT - 28
        footer_surf = self.font_small.render(
            f"DEBUG={C.DEBUG}  •  MAX_ARMY_SIZE={state.max_army_size}  •  "
            f"REINFORCEMENTS_PER_ROUND={state.reinforcements_per_round}",
            True,
            C.COLOR_TEXT_MUTED,
        )
        self.screen.blit(footer_surf, (x0, footer_y))
