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
# UI Button & Floating Combat Callout Models
# ============================================================================

@dataclass
class UIButton:
    rect: pygame.Rect
    label: str
    action_id: str
    enabled: bool = True
    accent_color: Tuple[int, int, int] = (75, 92, 115)


@dataclass
class CombatCallout:
    hex_pos: hexgrid.HexCoord
    text: str
    color: Tuple[int, int, int]
    frames_left: int = 75
    max_frames: int = 75
    y_offset: float = 0.0


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
        self.callouts: List[CombatCallout] = []

    def add_attack_callouts(self, results: List[rules.AttackResult]) -> None:
        """Spawn floating damage/status callouts for each target struck in an attack."""
        for idx, res in enumerate(results):
            vuln_tag = " (VULN!)" if res.is_vulnerable else ""
            slain_tag = " — SLAIN!" if res.defender_slain else ""
            txt = f"-{res.damage} HP{vuln_tag}{slain_tag}"
            col = (
                C.COLOR_TEXT_GOLD
                if (res.defender_slain or res.is_vulnerable)
                else C.COLOR_TEXT_RED
            )
            self.callouts.append(
                CombatCallout(
                    hex_pos=res.target_hex,
                    text=txt,
                    color=col,
                    y_offset=-18.0 * idx,
                )
            )

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

    def get_evolution_slot_rect(self, slot_idx: int) -> pygame.Rect:
        """
        Return the screen rectangle for survivor `slot_idx` (0 .. MAX_ARMY_SIZE - 1)
        arranged in 2 columns of up to 8 rows inside the Evolution Phase viewport.
        """
        col = slot_idx // C.RESERVE_ROWS
        row = slot_idx % C.RESERVE_ROWS
        pad_x = 24
        gap_x = 24
        cell_w = (C.SIDEBAR_X - 2 * pad_x - gap_x) // C.RESERVE_COLS  # 754px
        top_y = 94
        gap_y = 9
        cell_h = 116
        x = pad_x + col * (cell_w + gap_x)
        y = top_y + row * (cell_h + gap_y)
        return pygame.Rect(x, y, cell_w, cell_h)

    def get_evolution_unit_at_pos(
        self, state: rules.GameState, screen_pos: Tuple[int, int]
    ) -> Optional[str]:
        """Return the `unit_id` of the survivor clicked in `PHASE_EVOLUTION`, or None."""
        mx, my = screen_pos
        if mx < 0 or mx >= C.SIDEBAR_X:
            return None
        for idx, rec in enumerate(state.evolution_records):
            rect = self.get_evolution_slot_rect(idx)
            hit_w = rect.width if rec.leveled_up else 328
            hit_rect = pygame.Rect(rect.x, rect.y, hit_w, rect.height)
            if hit_rect.collidepoint(mx, my):
                return rec.unit_id
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
        self.buttons.clear()

        if state.phase == C.PHASE_SETUP:
            self.camera.set_viewport(C.MAP_VIEWPORT_X, C.MAP_VIEWPORT_WIDTH)
            self.draw_reserve_panel(
                state,
                hovered_reserve_slot=hovered_reserve_slot,
                dragging_unit_id=dragging_unit_id if is_dragging else None,
            )
            self.draw_map_viewport(
                state,
                arena_hexes,
                hovered_hex,
                status_message,
                dragging_unit_id=dragging_unit_id if is_dragging else None,
            )
        elif state.phase == C.PHASE_EVOLUTION:
            self.camera.set_viewport(0, C.SIDEBAR_X)
            self.draw_evolution_phase(state, status_message)
        else:
            # Left Reserve Panel goes away once deployment is confirmed (PHASE_BATTLE)
            self.camera.set_viewport(0, C.SIDEBAR_X)
            self.draw_map_viewport(
                state,
                arena_hexes,
                hovered_hex,
                status_message,
                dragging_unit_id=None,
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
        legal_attack_targets: List[hexgrid.HexCoord] = (
            rules.get_legal_attack_targets(state, sel_deployed.unit_id)
            if sel_deployed
            else []
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

            if coord in legal_attack_targets and sel_deployed is not None:
                atk_pts = self.camera.hex_polygon_points(coord, scale=0.86)
                pygame.draw.polygon(self.screen, C.COLOR_ATTACK_ARC, atk_pts, width=4)
            elif coord in attack_arc_hexes:
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
                    # Unit HP Bar
                    bar_w = 54
                    bar_h = 6
                    bx = int(ux - bar_w * 0.5)
                    by = int(uy + self.camera.hex_radius * 0.42)
                    hp_ratio = max(0.0, min(1.0, unit.hp / max(1, unit.spec.max_hp)))
                    fill_w = int(round(bar_w * hp_ratio))
                    hp_col = (
                        (70, 215, 115)
                        if unit.team == C.TEAM_PLAYER
                        else (230, 85, 75)
                    )
                    pygame.draw.rect(
                        self.screen, (18, 20, 26), pygame.Rect(bx, by, bar_w, bar_h), border_radius=3
                    )
                    if fill_w > 0:
                        pygame.draw.rect(
                            self.screen,
                            hp_col,
                            pygame.Rect(bx, by, fill_w, bar_h),
                            border_radius=3,
                        )
                    pygame.draw.rect(
                        self.screen,
                        (75, 85, 98),
                        pygame.Rect(bx, by, bar_w, bar_h),
                        width=1,
                        border_radius=3,
                    )

                    # Unit Name & HP Label
                    lbl_str = f"{unit.spec.name} ({unit.hp})"
                    lbl = self.font_tiny.render(lbl_str, True, C.COLOR_TEXT_MAIN)
                    bg_rect = pygame.Rect(
                        int(ux - lbl.get_width() * 0.5 - 4),
                        by + bar_h + 2,
                        lbl.get_width() + 8,
                        lbl.get_height() + 2,
                    )
                    pygame.draw.rect(self.screen, (16, 18, 24), bg_rect, border_radius=4)
                    pygame.draw.rect(
                        self.screen, ring_col, bg_rect, width=1, border_radius=4
                    )
                    self.screen.blit(lbl, (bg_rect.x + 4, bg_rect.y + 1))

                # If this unit is a legal attack target for the selected unit right now, show attack preview badge
                if (
                    coord in legal_attack_targets
                    and sel_deployed is not None
                    and unit.team != sel_deployed.team
                ):
                    dmg, _, is_vuln, _ = rules.compute_attack_damage(
                        state, sel_deployed, unit
                    )
                    atk_tag = f"ATK -{dmg} HP{' (VULN!)' if is_vuln else ''}"
                    atk_surf = self.font_badge.render(atk_tag, True, (255, 235, 195))
                    tag_rect = pygame.Rect(
                        int(ux - atk_surf.get_width() * 0.5 - 6),
                        int(uy - self.camera.hex_radius * 0.72),
                        atk_surf.get_width() + 12,
                        atk_surf.get_height() + 4,
                    )
                    pygame.draw.rect(
                        self.screen, (155, 38, 38), tag_rect, border_radius=5
                    )
                    pygame.draw.rect(
                        self.screen, C.COLOR_TEXT_GOLD, tag_rect, width=1, border_radius=5
                    )
                    self.screen.blit(atk_surf, (tag_rect.x + 6, tag_rect.y + 2))

        # 4. Draw floating combat callouts
        active_callouts: List[CombatCallout] = []
        for callout in self.callouts:
            cx, cy = self.camera.hex_to_pixel(callout.hex_pos)
            progress = 1.0 - (callout.frames_left / max(1, callout.max_frames))
            float_y = cy - self.camera.hex_radius * 0.55 - progress * 34.0 + callout.y_offset
            surf = self.font_badge.render(callout.text, True, callout.color)
            bg_r = pygame.Rect(
                int(cx - surf.get_width() * 0.5 - 6),
                int(float_y - surf.get_height() * 0.5 - 2),
                surf.get_width() + 12,
                surf.get_height() + 4,
            )
            pygame.draw.rect(self.screen, (16, 18, 24), bg_r, border_radius=5)
            pygame.draw.rect(self.screen, callout.color, bg_r, width=1, border_radius=5)
            self.screen.blit(surf, (bg_r.x + 6, bg_r.y + 2))

            callout.frames_left -= 1
            if callout.frames_left > 0:
                active_callouts.append(callout)
        self.callouts = active_callouts

        self.draw_viewport_hud(state, status_message)
        if state.phase == C.PHASE_DEFEAT:
            self.draw_defeat_banner(state)
        self.screen.set_clip(None)

    def draw_defeat_banner(self, state: rules.GameState) -> None:
        """Render a centered gothic Defeat overlay banner when all player reptiles are slain."""
        bw, bh = 680, 230
        bx = self.camera.viewport_x + (self.camera.viewport_w - bw) // 2
        by = (C.SCREEN_HEIGHT - bh) // 2
        banner_rect = pygame.Rect(bx, by, bw, bh)

        pygame.draw.rect(self.screen, (22, 16, 20), banner_rect, border_radius=12)
        pygame.draw.rect(
            self.screen, C.COLOR_TEXT_RED, banner_rect, width=3, border_radius=12
        )

        title = self.font_title.render(
            "DEFEAT — YOUR REPTILE ARMY HAS FALLEN", True, C.COLOR_TEXT_RED
        )
        self.screen.blit(
            title, (banner_rect.centerx - title.get_width() * 0.5, by + 28)
        )

        sub1 = self.font_subtitle.render(
            f"Overwhelmed by the Mammal Wave in Round {state.round_number} (Turn {state.turn_number})",
            True,
            C.COLOR_TEXT_GOLD,
        )
        self.screen.blit(
            sub1, (banner_rect.centerx - sub1.get_width() * 0.5, by + 76)
        )

        sub2 = self.font_body.render(
            "Press Enter or N to raise a fresh Round 1 army of 6 Stage 1 reptiles.",
            True,
            C.COLOR_TEXT_MAIN,
        )
        self.screen.blit(
            sub2, (banner_rect.centerx - sub2.get_width() * 0.5, by + 114)
        )

        btn_rect = pygame.Rect(banner_rect.centerx - 170, by + 158, 340, 46)
        self.buttons.append(
            UIButton(
                btn_rect,
                "Start New Campaign [Enter / N]",
                "new_game",
                enabled=True,
                accent_color=(155, 58, 58),
            )
        )

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
        elif state.phase == C.PHASE_DEFEAT:
            header_txt = (
                f"ROUND {state.round_number} DEFEAT   •   "
                f"ALL PLAYER REPTILES SLAIN ON TURN {state.turn_number}"
            )
            header_col = C.COLOR_TEXT_RED
            controls_txt = (
                "Press Enter or N (or click Start New Campaign) to begin a new Round 1 campaign."
            )
        else:
            enemies_alive = sum(
                1
                for u in state.units.values()
                if u.is_alive and u.team == C.TEAM_ENEMY
            )
            players_alive = sum(
                1
                for u in state.units.values()
                if u.is_alive and u.team == C.TEAM_PLAYER
            )
            if enemies_alive == 0:
                header_txt = (
                    f"ROUND {state.round_number} VICTORY!   •   "
                    f"ALL ENEMY MAMMALS SLAIN IN {state.turn_number} TURNS!"
                )
                header_col = C.COLOR_TEXT_GOLD
            else:
                team_col = (
                    C.COLOR_TEXT_GREEN
                    if state.active_team == C.TEAM_PLAYER
                    else C.COLOR_TEXT_RED
                )
                turn_tag = (
                    "YOUR TURN"
                    if state.active_team == C.TEAM_PLAYER
                    else "ENEMY MAMMAL AI TURN"
                )
                header_txt = (
                    f"ROUND {state.round_number}  •  TURN {state.turn_number} ({turn_tag})  •  "
                    f"REPTILES: {players_alive}  •  MAMMALS: {enemies_alive}"
                )
                header_col = team_col
            controls_txt = (
                "L-Click: Select / Move / Attack Red Target Hex  |  "
                "Wheel / Left-Right / Q-E / R-Click Adj Hex: Rotate  |  "
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

    def _draw_evolution_creature_box(
        self,
        rect: pygame.Rect,
        spec: creatures.CreatureSpec,
        line4_text: str,
        line4_col: Tuple[int, int, int],
        is_upgraded: bool,
        is_selected: bool,
    ) -> None:
        """Render a single creature card (either Original or Upgraded) inside the Evolution Phase."""
        if is_upgraded:
            bg_col = (26, 42, 36)
            border_col = C.COLOR_TEXT_GOLD if is_selected else C.COLOR_TEXT_GREEN
            ring_col = C.COLOR_TEXT_GOLD
        else:
            bg_col = (30, 35, 44)
            border_col = C.COLOR_TEXT_GOLD if is_selected else (82, 92, 108)
            ring_col = C.COLOR_PLAYER_RING

        border_w = 3 if is_selected else 2
        pygame.draw.rect(self.screen, bg_col, rect, border_radius=8)
        pygame.draw.rect(self.screen, border_col, rect, width=border_w, border_radius=8)

        # Sprite on the left side of the box
        sprite_px = 76
        sx = rect.x + 48
        sy = rect.centery
        pygame.draw.circle(self.screen, ring_col, (sx, sy), 34, width=2)
        sprite = self.assets.get_topdown_sprite(
            spec.topdown_path, C.FACING_N, sprite_px
        )
        self.screen.blit(
            sprite,
            (sx - sprite.get_width() * 0.5, sy - sprite.get_height() * 0.5),
        )

        # Text lines on the right side of the box
        tx = rect.x + 94
        ty = rect.y + 10
        name_col = C.COLOR_TEXT_GOLD if is_upgraded else C.COLOR_TEXT_MAIN
        name_surf = self.font_subtitle.render(spec.name, True, name_col)
        self.screen.blit(name_surf, (tx, ty))
        ty += 24

        stg_str = (
            f"Stage {spec.stage}/8 {spec.creature_class.title()} • {spec.point_cost} pt"
            + ("s" if spec.point_cost != 1 else "")
        )
        stg_surf = self.font_small.render(
            stg_str,
            True,
            C.COLOR_TEXT_GREEN if is_upgraded else C.COLOR_TEXT_GOLD,
        )
        self.screen.blit(stg_surf, (tx, ty))
        ty += 22

        stats_str = f"HP {spec.max_hp}  •  ATK {spec.attack}  •  DEF {spec.defense}"
        stats_surf = self.font_small.render(stats_str, True, C.COLOR_TEXT_MAIN)
        self.screen.blit(stats_surf, (tx, ty))
        ty += 22

        l4_surf = self.font_tiny.render(line4_text, True, line4_col)
        self.screen.blit(l4_surf, (tx, ty))

    def draw_evolution_phase(
        self, state: rules.GameState, status_message: str
    ) -> None:
        """
        Render the post-battle Evolution Phase (`PHASE_EVOLUTION`):
          - Leveled-up survivors are displayed in pairs:
            [Original Survivor]  ───►  [Upgraded Survivor]
          - Survivors that did not level up are displayed by themselves with no right arrow
            and no second form.
        """
        viewport_rect = pygame.Rect(0, 0, C.SIDEBAR_X, C.SCREEN_HEIGHT)
        self.screen.set_clip(viewport_rect)
        self.screen.fill(C.COLOR_BG_VOID, viewport_rect)

        # Top Header Bar
        hud_x = 24
        hud_w = C.SIDEBAR_X - 48
        hud_rect = pygame.Rect(hud_x, 14, hud_w, 68)
        pygame.draw.rect(self.screen, (20, 24, 30), hud_rect, border_radius=8)
        pygame.draw.rect(
            self.screen, C.COLOR_PANEL_BORDER, hud_rect, width=2, border_radius=8
        )

        leveled_cnt = sum(1 for r in state.evolution_records if r.leveled_up)
        header_txt = (
            f"ROUND {state.round_number} COMPLETE — SURVIVOR EVOLUTION PHASE   •   "
            f"SURVIVORS: {len(state.evolution_records)}   •   "
            f"LEVELED UP: {leveled_cnt}"
        )
        header = self.font_subtitle.render(header_txt, True, C.COLOR_TEXT_GOLD)
        self.screen.blit(header, (hud_x + 16, 22))

        sub_txt = (
            f"Click any survivor to inspect. Press Enter or click Confirm Evolution to receive "
            f"+{state.reinforcements_per_round} Stage 1 Reinforcements & begin Round {state.round_number + 1} Deployment!"
        )
        sub_surf = self.font_small.render(sub_txt, True, C.COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (hud_x + 16, 50))

        # Confirm button inside the top-right of the Evolution header bar
        hdr_btn_rect = pygame.Rect(hud_rect.right - 270, hud_rect.y + 14, 254, 40)
        self.buttons.append(
            UIButton(
                hdr_btn_rect,
                "Confirm Evolution [Enter]",
                "confirm_evolution",
                enabled=True,
                accent_color=(60, 155, 95),
            )
        )

        # Draw each surviving player reptile
        box_w = 328
        for idx, rec in enumerate(state.evolution_records):
            if idx >= C.MAX_ARMY_SIZE:
                break
            cell_rect = self.get_evolution_slot_rect(idx)
            is_sel = rec.unit_id == state.selected_unit_id

            left_rect = pygame.Rect(cell_rect.x, cell_rect.y, box_w, cell_rect.height)
            if rec.leveled_up:
                l4_orig = f"Battles Survived: {rec.battles_survived} (LEVEL UP!)"
                l4_orig_col = C.COLOR_TEXT_GOLD
            elif rec.old_spec.stage < 8:
                next_req = creatures.evolution_battles_required(rec.old_spec.stage + 1)
                l4_orig = (
                    f"Battles Survived: {rec.battles_survived}/{next_req} "
                    f"(Next Stage at {next_req})"
                )
                l4_orig_col = C.COLOR_TEXT_MUTED
            else:
                l4_orig = f"Battles Survived: {rec.battles_survived} (MAX STAGE 8)"
                l4_orig_col = C.COLOR_TEXT_GOLD

            # 1. Left Box: Original Survivor
            self._draw_evolution_creature_box(
                left_rect,
                rec.old_spec,
                l4_orig,
                l4_orig_col,
                is_upgraded=False,
                is_selected=is_sel and not rec.leveled_up,
            )

            # 2. If leveled up: draw Right-Pointing Arrow + Upgraded Survivor Box on the right
            if rec.leveled_up:
                arrow_x1 = left_rect.right + 12
                arrow_x2 = left_rect.right + 70
                arrow_y = cell_rect.centery + 6

                evo_tag = self.font_tiny.render("EVOLVED", True, C.COLOR_TEXT_GOLD)
                self.screen.blit(
                    evo_tag,
                    (
                        (arrow_x1 + arrow_x2 + 14) * 0.5 - evo_tag.get_width() * 0.5,
                        arrow_y - 24,
                    ),
                )

                pygame.draw.line(
                    self.screen,
                    C.COLOR_TEXT_GOLD,
                    (arrow_x1, arrow_y),
                    (arrow_x2, arrow_y),
                    width=6,
                )
                pygame.draw.polygon(
                    self.screen,
                    C.COLOR_TEXT_GOLD,
                    [
                        (arrow_x2, arrow_y - 11),
                        (arrow_x2 + 16, arrow_y),
                        (arrow_x2, arrow_y + 11),
                    ],
                )

                right_rect = pygame.Rect(
                    left_rect.right + 98, cell_rect.y, box_w, cell_rect.height
                )
                l4_new = (
                    f"Moves: {rec.new_spec.num_rotate_moves}  •  "
                    f"Rot: {rec.new_spec.max_rotate_per_move}  •  "
                    f"Atks: {rec.new_spec.total_attacks_per_move}"
                )
                self._draw_evolution_creature_box(
                    right_rect,
                    rec.new_spec,
                    l4_new,
                    C.COLOR_TEXT_CYAN,
                    is_upgraded=True,
                    is_selected=is_sel,
                )

        # Bottom Status Bar
        status_rect = pygame.Rect(hud_x, C.SCREEN_HEIGHT - 46, hud_w, 34)
        pygame.draw.rect(self.screen, (20, 24, 30), status_rect, border_radius=6)
        pygame.draw.rect(
            self.screen, (60, 70, 84), status_rect, width=1, border_radius=6
        )
        status_surf = self.font_body.render(status_message, True, C.COLOR_TEXT_GOLD)
        self.screen.blit(status_surf, (hud_x + 14, C.SCREEN_HEIGHT - 38))
        self.screen.set_clip(None)

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
        if unit is None:
            living = [u for u in state.units.values() if u.is_alive]
            if living:
                unit = living[0]

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
            self._draw_all_buttons()
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
        atk_targets = rules.get_legal_attack_targets(state, unit.unit_id)
        can_exec_atk = len(atk_targets) > 0
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
                f"Juncture Ready: {'YES' if can_atk else 'NO'}  •  Targets in Arc: {len(atk_targets)}",
                C.COLOR_TEXT_GREEN if can_exec_atk else C.COLOR_TEXT_MUTED,
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
                f"Attack & Flight: total_attacks={spec.total_attacks_per_move}   •   can_fly={'YES' if spec.can_fly else 'NO'}   •   post_rotate={'YES' if spec.can_rotate_after_move else 'NO'}",
                C.COLOR_TEXT_CYAN if (spec.can_fly or spec.can_rotate_after_move) else C.COLOR_TEXT_MAIN,
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

        # 5. Interactive Action & Setup/Battle/Evolution Buttons
        btn_w = (card_w - 12) // 2
        btn_h = 40

        if state.phase == C.PHASE_EVOLUTION:
            row1_y = y
            self.buttons.append(
                UIButton(
                    pygame.Rect(x0, row1_y, card_w, btn_h + 6),
                    "Confirm Evolution & Next Wave [Enter]",
                    "confirm_evolution",
                    enabled=True,
                    accent_color=(60, 155, 95),
                )
            )
        elif state.phase == C.PHASE_DEFEAT:
            row1_y = y
            self.buttons.append(
                UIButton(
                    pygame.Rect(x0, row1_y, card_w, btn_h + 6),
                    "Start New Campaign [Enter / N]",
                    "new_game",
                    enabled=True,
                    accent_color=(155, 58, 58),
                )
            )
        else:
            is_player_control = (
                state.phase == C.PHASE_SETUP
                or state.active_team == C.TEAM_PLAYER
            )
            can_rot_ccw = is_player_control and (((unit.facing - 1) % 6) in legal_rots)
            can_rot_cw = is_player_control and (((unit.facing + 1) % 6) in legal_rots)

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
                        enabled=is_player_control and can_exec_atk,
                        accent_color=(165, 70, 65),
                    )
                )
                end_lbl = (
                    "End Turn [Space]"
                    if state.active_team == C.TEAM_PLAYER
                    else "Enemy Turn..."
                )
                self.buttons.append(
                    UIButton(
                        pygame.Rect(x0 + btn_w + 12, row2_y, btn_w, btn_h),
                        end_lbl,
                        "end_turn",
                        enabled=(
                            state.active_team == C.TEAM_PLAYER
                            and rules.can_end_turn(state)
                        ),
                        accent_color=(70, 140, 90),
                    )
                )

        self._draw_all_buttons()

        footer_y = C.SCREEN_HEIGHT - 26
        footer_surf = self.font_small.render(
            f"DEBUG={C.DEBUG}  •  MAX_ARMY_SIZE={state.max_army_size}  •  "
            f"REINFORCEMENTS_BASE={state.reinforcements_per_round}",
            True,
            C.COLOR_TEXT_MUTED,
        )
        self.screen.blit(footer_surf, (x0, footer_y))

    def _draw_all_buttons(self) -> None:
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
