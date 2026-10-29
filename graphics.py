"""
Pygame asset caching, 3-pane layout (Left Army Reserve Tray, Center Hex Battlefield,
and Right Gothic Inspector Sidebar), and drag-and-drop rendering for Denizens.
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
# Battle Renderer (Left Reserve Tray + Center Hex Map + Right Gothic Sidebar)
# ============================================================================

class BattleRenderer:
    """Renders the 2x8 Army Reserve Tray, 61-hex arena, unit sprites, and inspector."""

    def __init__(
        self, screen: pygame.Surface, assets: AssetCache, camera: hexgrid.HexCamera
    ) -> None:
        self.screen = screen
        self.assets = assets
        self.camera = camera

        self.font_title = pygame.font.SysFont("georgia", 24, bold=True)
        self.font_subtitle = pygame.font.SysFont("georgia", 17, bold=True)
        self.font_body = pygame.font.SysFont("menlo", 14)
        self.font_small = pygame.font.SysFont("menlo", 12)
        self.font_tiny = pygame.font.SysFont("menlo", 11)
        self.font_badge = pygame.font.SysFont("menlo", 13, bold=True)

        self.buttons: List[UIButton] = []

    # ------------------------------------------------------------------------
    # Reserve Slot Geometry (2 columns x 8 rows = 16 slots)
    # ------------------------------------------------------------------------

    def get_reserve_slot_rect(self, slot_idx: int) -> pygame.Rect:
        """
        Return the screen rectangle for `slot_idx` (0 .. MAX_ARMY_SIZE - 1)
        arranged in 2 columns of 8 rows inside the Left Army Reserve Panel.
        """
        col = slot_idx // C.RESERVE_ROWS
        row = slot_idx % C.RESERVE_ROWS
        pad_x = 14
        gap_x = 12
        slot_w = (C.RESERVE_PANEL_WIDTH - 2 * pad_x - gap_x) // C.RESERVE_COLS  # 150px
        top_y = 92
        gap_y = 8
        slot_h = 124
        x = pad_x + col * (slot_w + gap_x)
        y = top_y + row * (slot_h + gap_y)
        return pygame.Rect(x, y, slot_w, slot_h)

    def get_reserve_slot_at_pos(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Return the reserve slot index (0..15) at `screen_pos`, or None."""
        mx, my = screen_pos
        if mx < 0 or mx >= C.RESERVE_PANEL_WIDTH:
            return None
        for slot_idx in range(C.MAX_ARMY_SIZE):
            if self.get_reserve_slot_rect(slot_idx).collidepoint(mx, my):
                return slot_idx
        return None

    # ------------------------------------------------------------------------
    # Master Frame Render
    # ------------------------------------------------------------------------

    def draw_frame(
        self,
        state: rules.GameState,
        arena_hexes: List[hexgrid.HexCoord],
        hovered_hex: Optional[hexgrid.HexCoord],
        status_message: str,
        hovered_reserve_slot: Optional[int] = None,
        dragging_unit_id: Optional[str] = None,
        is_dragging: bool = False,
        mouse_pos: Tuple[int, int] = (0, 0),
    ) -> None:
        if state.phase == C.PHASE_SETUP:
            self.camera.set_viewport(C.MAP_VIEWPORT_X, C.MAP_VIEWPORT_WIDTH)
            self.draw_reserve_panel(
                state,
                hovered_reserve_slot=hovered_reserve_slot,
                dragging_unit_id=dragging_unit_id if is_dragging else None,
            )
        else:
            # Left Reserve Panel goes away once deployment is confirmed (PHASE_BATTLE)
            self.camera.set_viewport(0, C.SIDEBAR_X)

        self.draw_map_viewport(
            state,
            arena_hexes,
            hovered_hex,
            status_message,
            dragging_unit_id=dragging_unit_id if is_dragging else None,
        )
        self.draw_sidebar(state, hovered_hex)

        if is_dragging and dragging_unit_id is not None:
            self.draw_dragged_unit_ghost(state, dragging_unit_id, mouse_pos)

    # ------------------------------------------------------------------------
    # Pane 1: Left Army Reserve Panel (x = 0 .. 340)
    # ------------------------------------------------------------------------

    def draw_reserve_panel(
        self,
        state: rules.GameState,
        hovered_reserve_slot: Optional[int] = None,
        dragging_unit_id: Optional[str] = None,
    ) -> None:
        panel_rect = pygame.Rect(0, 0, C.RESERVE_PANEL_WIDTH, C.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, C.COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(
            self.screen,
            C.COLOR_PANEL_BORDER,
            (C.RESERVE_PANEL_WIDTH - 2, 0),
            (C.RESERVE_PANEL_WIDTH - 2, C.SCREEN_HEIGHT),
            width=3,
        )

        # Header
        title = self.font_title.render("ARMY RESERVE", True, C.COLOR_TEXT_GOLD)
        self.screen.blit(title, (16, 16))

        army_units = state.get_player_army_units()
        deployed_count = sum(1 for u in army_units if u.is_deployed)
        reserve_count = sum(1 for u in army_units if u.in_reserve)
        army_pts = rules.calculate_player_army_points(state)

        sub1 = self.font_small.render(
            f"Roster: {len(army_units)}/{state.max_army_size}  •  Deployed: {deployed_count}",
            True,
            C.COLOR_TEXT_GREEN,
        )
        self.screen.blit(sub1, (16, 48))

        sub2 = self.font_small.render(
            f"Unplaced: {reserve_count}  •  Army Value: {army_pts} pts",
            True,
            C.COLOR_TEXT_MUTED,
        )
        self.screen.blit(sub2, (16, 66))

        # 16 Slots (2 columns of 8 rows)
        for slot_idx in range(C.MAX_ARMY_SIZE):
            rect = self.get_reserve_slot_rect(slot_idx)
            unit = state.get_unit_by_reserve_slot(slot_idx)
            is_selected = unit is not None and unit.unit_id == state.selected_unit_id
            is_hovered = slot_idx == hovered_reserve_slot

            # Empty slot OR unit already dragged/deployed onto the board -> blank box
            if unit is None or unit.is_deployed or unit.unit_id == dragging_unit_id:
                pygame.draw.rect(self.screen, C.COLOR_SLOT_EMPTY, rect, border_radius=7)
                pygame.draw.rect(
                    self.screen, (44, 50, 60), rect, width=1, border_radius=7
                )
                continue

            # Unplaced reserve unit in this slot
            bg_col = (34, 40, 50) if not is_hovered else (42, 50, 62)
            border_col = C.COLOR_TEXT_GOLD if is_selected else (84, 96, 114)
            border_w = 3 if is_selected else (2 if is_hovered else 1)
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=7)
            pygame.draw.rect(
                self.screen, border_col, rect, width=border_w, border_radius=7
            )

            # Top row inside slot: Stage & Cost badge
            stg_text = f"S{unit.spec.stage} • {unit.spec.point_cost}pt"
            stg_surf = self.font_tiny.render(stg_text, True, C.COLOR_TEXT_GOLD)
            self.screen.blit(stg_surf, (rect.x + 7, rect.y + 5))

            hp_text = f"HP {unit.hp}"
            hp_surf = self.font_tiny.render(hp_text, True, C.COLOR_TEXT_MUTED)
            self.screen.blit(
                hp_surf, (rect.right - hp_surf.get_width() - 7, rect.y + 5)
            )

            # Creature top-down sprite preview
            sprite_size = 64
            sprite = self.assets.get_topdown_sprite(
                unit.spec.topdown_path, C.FACING_N, sprite_size
            )
            self.screen.blit(
                sprite,
                (
                    rect.centerx - sprite.get_width() * 0.5,
                    rect.y + 20,
                ),
            )

            # Creature Name
            name_str = unit.spec.name
            if len(name_str) > 15:
                name_str = name_str[:14] + "."
            name_surf = self.font_small.render(name_str, True, C.COLOR_TEXT_MAIN)
            self.screen.blit(
                name_surf,
                (rect.centerx - name_surf.get_width() * 0.5, rect.y + 84),
            )

            # Status tag
            status_txt = "DRAG TO MAP" if state.phase == C.PHASE_SETUP else "IN RESERVE"
            status_col = (
                C.COLOR_TEXT_CYAN if state.phase == C.PHASE_SETUP else C.COLOR_TEXT_MUTED
            )
            st_surf = self.font_tiny.render(status_txt, True, status_col)
            self.screen.blit(
                st_surf,
                (rect.centerx - st_surf.get_width() * 0.5, rect.y + 103),
            )

        # Bottom hint inside left panel
        hint_txt = (
            "Drag units to lower half of map"
            if state.phase == C.PHASE_SETUP
            else "Click any unit to inspect stats"
        )
        hint_surf = self.font_tiny.render(hint_txt, True, C.COLOR_TEXT_MUTED)
        self.screen.blit(
            hint_surf,
            (
                C.RESERVE_PANEL_WIDTH * 0.5 - hint_surf.get_width() * 0.5,
                C.SCREEN_HEIGHT - 24,
            ),
        )

    # ------------------------------------------------------------------------
    # Pane 2: Center Hex Battlefield Viewport (x = 340 .. 1580)
    # ------------------------------------------------------------------------

    def draw_map_viewport(
        self,
        state: rules.GameState,
        arena_hexes: List[hexgrid.HexCoord],
        hovered_hex: Optional[hexgrid.HexCoord],
        status_message: str,
        dragging_unit_id: Optional[str] = None,
    ) -> None:
        viewport_rect = pygame.Rect(
            self.camera.viewport_x, 0, self.camera.viewport_w, C.SCREEN_HEIGHT
        )
        self.screen.set_clip(viewport_rect)
        self.screen.fill(C.COLOR_BG_VOID, viewport_rect)

        sel_unit = state.get_unit(state.selected_unit_id)
        sel_deployed = sel_unit if (sel_unit is not None and sel_unit.is_deployed) else None

        legal_moves: List[hexgrid.HexCoord] = (
            rules.get_legal_moves(state, sel_deployed.unit_id) if sel_deployed else []
        )
        legal_rotations: List[int] = (
            rules.get_legal_rotations(state, sel_deployed.unit_id) if sel_deployed else []
        )
        attack_arc_hexes: List[hexgrid.HexCoord] = (
            rules.get_attack_arc_hexes(state, sel_deployed.unit_id) if sel_deployed else []
        )

        is_setup = state.phase == C.PHASE_SETUP

        # 1. Draw all 61 arena hexes
        for coord in arena_hexes:
            q, r = coord
            pts = self.camera.hex_polygon_points(coord, scale=0.98)
            in_deploy_zone = is_setup and rules.is_valid_player_deploy_hex(state, coord)

            if in_deploy_zone:
                fill_col = (
                    C.COLOR_HEX_DEPLOY_A if (q - r) % 2 == 0 else C.COLOR_HEX_DEPLOY_B
                )
                border_col = C.COLOR_HEX_DEPLOY_BORDER
            elif coord == (0, 0):
                fill_col = C.COLOR_HEX_CENTER
                border_col = C.COLOR_HEX_BORDER
            elif (q - r) % 2 == 0:
                fill_col = C.COLOR_HEX_FILL_A
                border_col = C.COLOR_HEX_BORDER
            else:
                fill_col = C.COLOR_HEX_FILL_B
                border_col = C.COLOR_HEX_BORDER

            pygame.draw.polygon(self.screen, fill_col, pts)
            pygame.draw.polygon(self.screen, border_col, pts, width=2)

            if coord in attack_arc_hexes:
                inner_pts = self.camera.hex_polygon_points(coord, scale=0.88)
                pygame.draw.polygon(self.screen, C.COLOR_ATTACK_ARC, inner_pts, width=2)

            if coord in legal_moves and sel_deployed is not None and sel_deployed.hex_pos is not None:
                occupant = state.get_unit_at_hex(coord, exclude_unit_id=sel_deployed.unit_id)
                hl_col = (
                    C.COLOR_LEGAL_FLY_OVER if occupant is not None else C.COLOR_LEGAL_MOVE
                )
                inner_pts = self.camera.hex_polygon_points(coord, scale=0.84)
                pygame.draw.polygon(self.screen, hl_col, inner_pts, width=4)

                abs_dir = hexgrid.hex_direction_between(sel_deployed.hex_pos, coord)
                if abs_dir is not None and self.camera.zoom >= 0.70:
                    rel_dir = hexgrid.relative_direction(sel_deployed.facing, abs_dir)
                    tag = "FLY OVER" if occupant is not None else C.REL_DIR_NAMES[rel_dir]
                    lbl = self.font_small.render(tag, True, hl_col)
                    cx, cy = self.camera.hex_to_pixel(coord)
                    self.screen.blit(
                        lbl,
                        (cx - lbl.get_width() * 0.5, cy + self.camera.hex_radius * 0.42),
                    )

            if coord == hovered_hex:
                hover_pts = self.camera.hex_polygon_points(coord, scale=0.94)
                if dragging_unit_id is not None and is_setup:
                    drag_col = (
                        C.COLOR_LEGAL_MOVE
                        if rules.is_valid_player_deploy_hex(state, coord)
                        else C.COLOR_ATTACK_ARC
                    )
                    pygame.draw.polygon(self.screen, drag_col, hover_pts, width=4)
                else:
                    pygame.draw.polygon(self.screen, C.COLOR_HEX_HOVER, hover_pts, width=2)

            if C.DEBUG and self.camera.zoom >= 0.65:
                cx, cy = self.camera.hex_to_pixel(coord)
                coord_surf = self.font_small.render(f"{q},{r}", True, (82, 92, 108))
                self.screen.blit(
                    coord_surf,
                    (cx - coord_surf.get_width() * 0.5, cy - self.camera.hex_radius * 0.74),
                )

        # 2. Highlight selected deployed unit's hex and its directional edges
        if sel_deployed is not None and sel_deployed.hex_pos is not None:
            sel_pts = self.camera.hex_polygon_points(sel_deployed.hex_pos, scale=0.95)
            pygame.draw.polygon(self.screen, C.COLOR_SELECTED_HEX, sel_pts, width=3)

            for vuln_rel in sel_deployed.spec.vulnerable_arcs:
                abs_edge = hexgrid.absolute_direction(sel_deployed.facing, vuln_rel)
                p1, p2 = self.camera.hex_edge_endpoints(
                    sel_deployed.hex_pos, abs_edge, scale=0.88
                )
                pygame.draw.line(self.screen, C.COLOR_VULNERABLE_EDGE, p1, p2, width=4)

            for atk_rel in sel_deployed.spec.attack_arcs:
                abs_edge = hexgrid.absolute_direction(sel_deployed.facing, atk_rel)
                p1, p2 = self.camera.hex_edge_endpoints(
                    sel_deployed.hex_pos, abs_edge, scale=0.88
                )
                col = C.COLOR_FRONT_EDGE if atk_rel == C.FORWARD else C.COLOR_ATTACK_ARC
                pygame.draw.line(self.screen, col, p1, p2, width=6)

            for rot_facing in legal_rotations:
                if rot_facing == sel_deployed.facing:
                    continue
                p1, p2 = self.camera.hex_edge_endpoints(
                    sel_deployed.hex_pos, rot_facing, scale=0.76
                )
                mx = 0.5 * (p1[0] + p2[0])
                my = 0.5 * (p1[1] + p2[1])
                pygame.draw.circle(self.screen, C.COLOR_TEXT_GOLD, (int(mx), int(my)), 5)

        # 3. Draw deployed units on the board
        sprite_px = int(self.camera.hex_radius * 1.62)
        for coord in arena_hexes:
            units_at_coord = state.get_units_at_hex(coord)
            if not units_at_coord:
                continue
            cx, cy = self.camera.hex_to_pixel(coord)
            for idx, unit in enumerate(units_at_coord):
                if unit.unit_id == dragging_unit_id:
                    continue
                offset_x = 12 * idx
                offset_y = -14 * idx
                ux = cx + offset_x
                uy = cy + offset_y

                ring_col = (
                    C.COLOR_PLAYER_RING
                    if unit.team == C.TEAM_PLAYER
                    else C.COLOR_ENEMY_RING
                )
                ring_radius = int(self.camera.hex_radius * 0.64)
                pygame.draw.circle(
                    self.screen, ring_col, (int(ux), int(uy)), ring_radius, width=3
                )

                facing_angle_rad = math.radians(60.0 * unit.facing - 90.0)
                tip_dist = ring_radius + 7
                tip_x = ux + math.cos(facing_angle_rad) * tip_dist
                tip_y = uy + math.sin(facing_angle_rad) * tip_dist
                pygame.draw.circle(
                    self.screen, C.COLOR_FRONT_EDGE, (int(tip_x), int(tip_y)), 5
                )

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
                    pygame.draw.rect(
                        self.screen, ring_col, bg_rect, width=1, border_radius=4
                    )
                    self.screen.blit(lbl, (bg_rect.x + 4, bg_rect.y + 1))

        self.draw_viewport_hud(state, status_message)
        self.screen.set_clip(None)

    def draw_viewport_hud(self, state: rules.GameState, status_message: str) -> None:
        hud_x = self.camera.viewport_x + 18
        hud_w = self.camera.viewport_w - 36
        hud_rect = pygame.Rect(hud_x, 14, hud_w, 68)
        pygame.draw.rect(self.screen, (20, 24, 30), hud_rect, border_radius=8)
        pygame.draw.rect(
            self.screen, C.COLOR_PANEL_BORDER, hud_rect, width=2, border_radius=8
        )

        if state.phase == C.PHASE_SETUP:
            army_units = state.get_player_army_units()
            deployed_cnt = sum(1 for u in army_units if u.is_deployed)
            header_txt = (
                f"ROUND {state.round_number} — ARMY SETUP PHASE   •   "
                f"DEPLOYED: {deployed_cnt}/{len(army_units)}   •   "
                f"ENEMY WAVE BUDGET: {state.wave_budget} PTS ({state.game_mode.upper()})"
            )
            header_col = C.COLOR_TEXT_GOLD
            controls_txt = (
                "Drag reptiles from Left Reserve onto lower-half green hexes  |  "
                "Wheel / Q-E / Left-Right: Rotate  |  Enter: Confirm Deployment"
            )
        else:
            team_col = (
                C.COLOR_TEXT_GREEN
                if state.active_team == C.TEAM_PLAYER
                else C.COLOR_TEXT_RED
            )
            header_txt = (
                f"ROUND {state.round_number}  •  TURN {state.turn_number}  •  "
                f"ACTIVE TEAM: {state.active_team.upper()}  •  "
                f"WAVE BUDGET: {state.wave_budget} PTS"
            )
            header_col = team_col
            controls_txt = (
                "L-Click: Select/Move  |  Wheel / Left-Right / Q-E / R-Click Adj Hex: Rotate  |  "
                "F: Attack  |  Space: End Turn"
            )

        header = self.font_subtitle.render(header_txt, True, header_col)
        self.screen.blit(header, (hud_x + 16, 22))

        controls_hint = self.font_small.render(controls_txt, True, C.COLOR_TEXT_MUTED)
        self.screen.blit(controls_hint, (hud_x + 16, 50))

        status_rect = pygame.Rect(
            hud_x, C.SCREEN_HEIGHT - 46, hud_w, 34
        )
        pygame.draw.rect(self.screen, (20, 24, 30), status_rect, border_radius=6)
        pygame.draw.rect(
            self.screen, (60, 70, 84), status_rect, width=1, border_radius=6
        )
        status_surf = self.font_body.render(status_message, True, C.COLOR_TEXT_GOLD)
        self.screen.blit(status_surf, (hud_x + 14, C.SCREEN_HEIGHT - 38))

    def draw_dragged_unit_ghost(
        self,
        state: rules.GameState,
        dragging_unit_id: str,
        mouse_pos: Tuple[int, int],
    ) -> None:
        unit = state.get_unit(dragging_unit_id)
        if unit is None:
            return
        sprite_px = int(self.camera.hex_radius * 1.55)
        sprite = self.assets.get_topdown_sprite(
            unit.spec.topdown_path, unit.facing, sprite_px
        )
        ghost = sprite.copy()
        ghost.fill((255, 255, 255, 215), special_flags=pygame.BLEND_RGBA_MULT)
        mx, my = mouse_pos
        self.screen.blit(
            ghost,
            (mx - ghost.get_width() * 0.5, my - ghost.get_height() * 0.5),
        )

    # ------------------------------------------------------------------------
    # Pane 3: Right Gothic Portrait & Stat Inspector (x = 1580 .. 2200)
    # ------------------------------------------------------------------------

    def draw_sidebar(
        self, state: rules.GameState, hovered_hex: Optional[hexgrid.HexCoord]
    ) -> None:
        panel_rect = pygame.Rect(C.SIDEBAR_X, 0, C.SIDEBAR_WIDTH, C.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, C.COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(
            self.screen,
            C.COLOR_PANEL_BORDER,
            (C.SIDEBAR_X, 0),
            (C.SIDEBAR_X, C.SCREEN_HEIGHT),
            width=3,
        )

        unit = state.get_unit(state.selected_unit_id)
        if unit is None and hovered_hex is not None:
            unit = state.get_unit_at_hex(hovered_hex)

        x0 = C.SIDEBAR_X + 24
        y = 18

        title_surf = self.font_title.render("DENIZENS INSPECTOR", True, C.COLOR_TEXT_GOLD)
        self.screen.blit(title_surf, (x0, y))
        y += 36

        if unit is None:
            empty_msg = self.font_body.render(
                "Click any creature in Reserve or on the Map to inspect.",
                True,
                C.COLOR_TEXT_MUTED,
            )
            self.screen.blit(empty_msg, (x0, y + 40))
            return

        spec = unit.spec

        # 1. Framed Gothic Portrait
        portrait_size = 370
        px = C.SIDEBAR_X + (C.SIDEBAR_WIDTH - portrait_size) // 2
        portrait_surf = self.assets.get_portrait(spec.portrait_path, portrait_size)
        self.screen.blit(portrait_surf, (px, y))
        pygame.draw.rect(
            self.screen,
            C.COLOR_PANEL_BORDER,
            pygame.Rect(px, y, portrait_size, portrait_size),
            width=2,
        )
        y += portrait_size + 12

        # 2. Creature Name, Class, Stage, Point Cost & Evolution Info
        team_badge_col = (
            C.COLOR_TEXT_GREEN if unit.team == C.TEAM_PLAYER else C.COLOR_TEXT_RED
        )
        if spec.creature_class in C.PLAYER_CLASSES:
            stage_label = f"Stage {spec.stage}/8 {spec.creature_class.title()} • {spec.point_cost} pts"
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
            stage_label = (
                f"Enemy Mammal #{spec.stage}/13 • Cost: {spec.point_cost} pts"
            )
            evo_str = "Enemy Opposing Force (Non-Evolving)"

        name_surf = self.font_title.render(spec.name, True, C.COLOR_TEXT_MAIN)
        self.screen.blit(name_surf, (x0, y))
        badge_surf = self.font_badge.render(
            f"[{unit.team.upper()} • {stage_label}]", True, team_badge_col
        )
        self.screen.blit(badge_surf, (x0 + name_surf.get_width() + 12, y + 5))
        y += 30

        evo_surf = self.font_body.render(evo_str, True, C.COLOR_TEXT_GOLD)
        self.screen.blit(evo_surf, (x0, y))
        y += 24

        # 3. Turn Step, Location & Juncture Status Card
        card_w = C.SIDEBAR_WIDTH - 48
        step_card = pygame.Rect(x0, y, card_w, 128)
        pygame.draw.rect(self.screen, C.COLOR_CARD_BG, step_card, border_radius=8)
        pygame.draw.rect(
            self.screen, C.COLOR_PANEL_BORDER, step_card, width=1, border_radius=8
        )

        can_atk = rules.can_unit_attack_now(state, unit.unit_id)
        legal_rots = rules.get_legal_rotations(state, unit.unit_id)
        legal_mvs = rules.get_legal_moves(state, unit.unit_id)

        if unit.in_reserve:
            slot_num = (unit.reserve_slot + 1) if unit.reserve_slot is not None else 1
            pos_str = f"Location: IN RESERVE (Slot #{slot_num})  •  Drag to lower half of map"
        else:
            pos_str = f"Hex Position: {unit.hex_pos}  •  Legal Destination Hexes: {len(legal_mvs)}"

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
                pos_str,
                C.COLOR_TEXT_GOLD if unit.in_reserve else C.COLOR_TEXT_MAIN,
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
        pygame.draw.rect(
            self.screen, (62, 70, 84), attr_card, width=1, border_radius=8
        )

        move_dir_names = ", ".join(
            C.REL_DIR_NAMES[d] for d in sorted(spec.legal_move_dirs)
        )
        atk_arc_names = ", ".join(
            C.REL_DIR_NAMES[d] for d in sorted(spec.attack_arcs)
        )
        vuln_arc_names = (
            ", ".join(C.REL_DIR_NAMES[d] for d in sorted(spec.vulnerable_arcs))
            if spec.vulnerable_arcs
            else "None (Armored All Sides)"
        )
        powers_str = (
            ", ".join(spec.special_powers)
            if spec.special_powers
            else "None (Basic Form)"
        )

        attr_lines = [
            (
                f"Combat Stats:  HP {unit.hp}/{spec.max_hp}   •   ATK {spec.attack}   •   DEF {spec.defense}   •   Cost {spec.point_cost}pt",
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
        y += 208

        # 5. Interactive Action & Setup/Battle Buttons
        self.buttons.clear()
        btn_w = (card_w - 12) // 2
        btn_h = 40

        can_rot_ccw = ((unit.facing - 1) % 6) in legal_rots
        can_rot_cw = ((unit.facing + 1) % 6) in legal_rots

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
        if state.phase == C.PHASE_SETUP:
            can_undeploy = unit.team == C.TEAM_PLAYER and unit.is_deployed
            self.buttons.append(
                UIButton(
                    pygame.Rect(x0, row2_y, btn_w, btn_h),
                    "Return to Reserve [Bksp]",
                    "undeploy_unit",
                    enabled=can_undeploy,
                    accent_color=(135, 85, 65),
                )
            )
            self.buttons.append(
                UIButton(
                    pygame.Rect(x0 + btn_w + 12, row2_y, btn_w, btn_h),
                    "Confirm Deployment [Enter]",
                    "confirm_deployment",
                    enabled=rules.can_confirm_deployment(state),
                    accent_color=(60, 155, 95),
                )
            )
        else:
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

        footer_y = C.SCREEN_HEIGHT - 26
        footer_surf = self.font_small.render(
            f"DEBUG={C.DEBUG}  •  MAX_ARMY_SIZE={state.max_army_size}  •  "
            f"REINFORCEMENTS_BASE={state.reinforcements_per_round}",
            True,
            C.COLOR_TEXT_MUTED,
        )
        self.screen.blit(footer_surf, (x0, footer_y))
