"""
Denizens — Sound Manager & `.ogg` Audio Subsystem (`sounds.py`)
===============================================================
Provides centralized `.ogg` audio loading, caching, and playback for:
  - UI feedback (`positiveclick.ogg`, `negativeclick.ogg`, `victory.ogg`, `defeat.ogg`)
  - Creature movement (`flight.ogg`, `footsteps.ogg`, `largestomp.ogg`, `softmove.ogg`)
  - Reptile attacks (`crunch.ogg`, `dragon.ogg`, `dragtort.ogg`, `hiss.ogg`,
                     `lizard.ogg`, `pecking.ogg`, `snap.ogg`, `snapper.ogg`)
  - Mammal attacks (`baboon.ogg`, `deepsnarl.ogg`, `eagle.ogg`, `growl.ogg`,
                    `hound.ogg`, `rat.ogg`, `roar.ogg`)

All sound files are stored in `.ogg` (Ogg Vorbis) format for Pygame and future
Pygbag WebAssembly compatibility.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pygame

from creatures import ALL_CREATURES, CreatureSpec


UI_POSITIVE_PATH = "assets/sounds/UI/positiveclick.ogg"
UI_NEGATIVE_PATH = "assets/sounds/UI/negativeclick.ogg"
UI_VICTORY_PATH = "assets/sounds/UI/victory.ogg"
UI_DEFEAT_PATH = "assets/sounds/UI/defeat.ogg"

ALL_UI_SOUND_PATHS = (
    UI_POSITIVE_PATH,
    UI_NEGATIVE_PATH,
    UI_VICTORY_PATH,
    UI_DEFEAT_PATH,
)

# Default per-category volume tuning so UI clicks are crisp without overpowering combat audio
VOLUME_UI_CLICK = 0.65
VOLUME_UI_FANFARE = 0.85
VOLUME_MOVE = 0.75
VOLUME_ATTACK = 0.90


class SoundManager:
    """
    Loads, caches, and plays `.ogg` sound effects for UI interactions, creature
    movements, creature attacks, and victory/defeat fanfares.

    Degrades gracefully when running in a headless environment without audio
    hardware while still validating `.ogg` file paths and recording playback
    history for unit tests.
    """

    def __init__(self, base_dir: str, enabled: bool = True) -> None:
        self.base_dir: str = base_dir
        self.enabled: bool = enabled
        self.mixer_available: bool = False
        self._cache: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.last_played_path: Optional[str] = None
        self.played_history: List[str] = []

        if self.enabled:
            self._init_mixer()

    def _init_mixer(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            # Ensure enough channels for overlapping move/attack/UI sounds
            pygame.mixer.set_num_channels(max(16, pygame.mixer.get_num_channels()))
            self.mixer_available = True
        except Exception:
            self.mixer_available = False

    def resolve_path(self, rel_path: str) -> str:
        return os.path.join(self.base_dir, rel_path)

    def _get_sound(self, rel_path: str) -> Optional[pygame.mixer.Sound]:
        if rel_path in self._cache:
            return self._cache[rel_path]

        full_path = self.resolve_path(rel_path)
        if not os.path.isfile(full_path):
            self._cache[rel_path] = None
            return None

        if not self.mixer_available:
            self._cache[rel_path] = None
            return None

        try:
            snd = pygame.mixer.Sound(full_path)
            self._cache[rel_path] = snd
            return snd
        except Exception:
            self._cache[rel_path] = None
            return None

    def preload_all(self) -> int:
        """
        Preloads all UI, movement, and creature attack `.ogg` files into memory.
        Returns the number of unique `.ogg` paths verified to exist on disk.
        """
        paths = set(ALL_UI_SOUND_PATHS)
        for spec in ALL_CREATURES:
            paths.add(spec.move_sound_path)
            paths.add(spec.attack_sound_path)

        verified_count = 0
        for rel_path in sorted(paths):
            full_path = self.resolve_path(rel_path)
            if os.path.isfile(full_path):
                verified_count += 1
                self._get_sound(rel_path)
        return verified_count

    def play_path(self, rel_path: str, volume: float = 1.0) -> bool:
        """
        Plays the `.ogg` sound at `rel_path` if enabled and the file exists on disk.
        Returns True if the sound file exists and was triggered (even in headless mode
        where `mixer_available` is False), or False if disabled or missing.
        """
        if not self.enabled:
            return False

        full_path = self.resolve_path(rel_path)
        if not os.path.isfile(full_path):
            return False

        self.last_played_path = rel_path
        self.played_history.append(rel_path)

        snd = self._get_sound(rel_path)
        if snd is not None:
            try:
                snd.set_volume(max(0.0, min(1.0, volume)))
                snd.play()
            except Exception:
                pass
        return True

    def play_ui_positive(self) -> bool:
        """Plays the positive UI click feedback sound (`positiveclick.ogg`)."""
        return self.play_path(UI_POSITIVE_PATH, volume=VOLUME_UI_CLICK)

    def play_ui_negative(self) -> bool:
        """Plays the negative/illegal UI click feedback sound (`negativeclick.ogg`)."""
        return self.play_path(UI_NEGATIVE_PATH, volume=VOLUME_UI_CLICK)

    def play_victory(self) -> bool:
        """Plays the battle victory fanfare (`victory.ogg`)."""
        return self.play_path(UI_VICTORY_PATH, volume=VOLUME_UI_FANFARE)

    def play_defeat(self) -> bool:
        """Plays the campaign defeat fanfare (`defeat.ogg`)."""
        return self.play_path(UI_DEFEAT_PATH, volume=VOLUME_UI_FANFARE)

    def play_move(self, spec: CreatureSpec) -> bool:
        """Plays the movement sound configured for `spec` (`spec.move_sound_path`)."""
        return self.play_path(spec.move_sound_path, volume=VOLUME_MOVE)

    def play_attack(self, spec: CreatureSpec) -> bool:
        """Plays the attack sound configured for `spec` (`spec.attack_sound_path`)."""
        return self.play_path(spec.attack_sound_path, volume=VOLUME_ATTACK)

    def clear_history(self) -> None:
        """Clears the recorded playback history."""
        self.last_played_path = None
        self.played_history.clear()
