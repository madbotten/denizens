# Denizens — Project Overview & Agent Guidelines

Keep the player-facing instructions file (`denizens.txt`) in sync whenever game rules, evolution stages, or creature mechanics change.

@[denizens.txt](denizens.txt)

---

## Architectural Implications for Code & Rules (`rules.py`)

When designing and implementing game logic for **Denizens**, ensure that the shared rules module (`rules.py`) models these mechanics cleanly so human UI and enemy/bot AI share the exact same validation and execution code:

1. **Relative Hex Directions (6 Facings)**:
   - Represent creature `facing` as an integer `0..5` corresponding to the 6 hex sides.
   - Compute relative directions (`FRONT = 0`, `FRONT_RIGHT = 1`, `FLANK_RIGHT = 2`, `REAR = 3`, `FLANK_LEFT = 4`, `FRONT_LEFT = 5`) via modular arithmetic `(target_dir - facing) % 6`.
   - Top-down creature sprites in `assets/` are drawn moving toward the **Top of the image (`North` / `0°`, which represents the `Front` hex facing)**, so rotating a sprite to match its hex facing is a clean multiple of `60°`.

2. **Strict Top-Down Sprite Camera & Orientation Rule (All `topdown/` Assets ONLY)**:
   - Every creature in a `topdown/` folder MUST appear to be moving toward the **top edge (12 o'clock)** of the image, which corresponds to the creature's **Front** facing when placed on a hex.
   - The camera is strictly **90° overhead (dorsal bird's-eye view)**: only the **top (dorsal side) of the creature's head(s) and body** may be visible.
   - Creature heads must look **forward along the ground plane toward the top of the image**, **NEVER** tilted upward to face/stare at the camera, and **NEVER** showing ventral/throat/underside neck scales as if rearing up at the viewer.
   - *(Note: `portraits/` assets are character stat portraits and SHOULD show the creature's head/upper body from the front or dramatic three-quarter angle inside the unified gothic border frame.)*

2. **Rule-Breaking Evolution Traits (Data-Driven Capabilities on `CreatureSpec`)**:
   - Rather than hardcoding movement or attack constraints, `rules.py` inspects per-creature attributes on `CreatureSpec`:
     - **`num_rotate_moves`**: Number of `(Rotate -> Move)` cycles per turn (default `1` for Stage 1; up to `4` for Stage 8 `Tian Long`).
     - **`max_rotate_per_move`**: Maximum `60°` hex faces (`0..3`) a creature can rotate away from its step-start facing in each cycle (default `1` for Stage 1; `3` for omnidirectional `180°` rotation).
     - **`legal_move_dirs`**: Allowed relative move directions (`FORWARD`, `FORWARD_LEFT`, `FORWARD_RIGHT`, `FLANK_LEFT`, `FLANK_RIGHT`, `REAR`) relative to instantaneous facing at the moment of the move (default `{FORWARD}` for Stage 1).
     - **`total_attacks_per_move` & Attack Junctures**: Total attacks allowed per turn (default `1`), taken at `BEFORE_ROTATE`, `AFTER_ROTATE`, or `AFTER_MOVE` junctures (at most 1 attack per juncture). Rotations, moves, and attacks are all optional.
     - **`can_fly`**: Winged creatures may fly over occupied hexes on intermediate move steps provided they can land on an empty hex before their moves end.
     - **Attack Arcs (`attack_arcs`) & Vulnerability Arcs (`vulnerable_arcs`)**: Relative offensive directions (default `{FORWARD}`, e.g. Hydra `{FORWARD, FORWARD_LEFT, FORWARD_RIGHT}`) and vulnerable incoming directions (default `{FLANK_LEFT, FLANK_RIGHT, REAR}`).

3. **Evolution Progression Formula**:
   - Each class has **8 stages** (`1..8`).
   - Advancing from stage $s$ to $s + 1$ requires surviving $s$ additional battles.
   - Total cumulative battles survived required to reach stage $s$:
     $$\text{total\_battles}(s) = \frac{(s - 1) \cdot s}{2}$$
     (`0, 1, 3, 6, 10, 15, 21, 28` for Stages `1..8`).

4. **Campaign Difficulty Modes & Genetic Bot Evolution (`bots.py` & `bot_stable.json`)**:
   - **`GAME_MODE_BASIC` (Play Basic)**: Classic Benchmark AI (`rules.py`) with adaptive wave budgets scaled to the player's surviving army points.
   - **`GAME_MODE_INTERMEDIATE` (Play Intermediate)**: Classic Benchmark AI (`rules.py`) with the fixed mathematical wave schedule ($4 + 3(R - 1) + (R - 1)^2$).
   - **`GAME_MODE_ADVANCED` (Play Advanced)**: Evolved Champion AI (`bots.GeneticBotController` loaded from `bot_stable.json`) with the fixed mathematical wave schedule ($4 + 3(R - 1) + (R - 1)^2$).
   - **UI-Free Evolutionary Arena (`bots.py`)**: Evolves `BotGenome` populations (`SetupGenes`, per-species `MammalTacticalGenes`, `TargetingGenes`, `LookaheadGenes` up to 4 plies, and `GlobalCoordinationGenes`) via paired North/South-swapped tournaments, direct champion mutations (`mutate_champion`), two-parent crossovers (`crossover_and_mutate`), and random injections (`create_random_genome`). Persists top champions to `bot_stable.json`.

---

## Asset Organization & Visual Standards

- **8 Stages per Player Class**:
  - **Serpent (Snakes)**: `01_adder`, `02_viper`, `03_serpent`, `04_jaculus`, `05_amphisbaena`, `06_wyrm`, `07_basilisk`, `08_hydra`
    - Top-down hex sprites (`1024×1024` transparent RGBA PNGs): `assets/monsters/serpent/topdown/01_adder.png` .. `08_hydra.png`
    - Stat portraits (`1024×1024` PNGs with unified Jaculus gothic iron-and-bronze frame): `assets/monsters/serpent/portraits/01_adder.png` .. `08_hydra.png`
  - **Lizards**: `01_newt`, `02_horned`, `03_monitor`, `04_komodo`, `05_wyrmling`, `06_drake`, `07_young_dragon`, `08_tian_long`
    - Top-down hex sprites (`1024×1024` transparent RGBA PNGs): `assets/monsters/lizard/topdown/01_newt.png` .. `08_tian_long.png`
    - Stat portraits (`1024×1024` PNGs with unified Jaculus gothic iron-and-bronze frame): `assets/monsters/lizard/portraits/01_newt.png` .. `08_tian_long.png`
  - **Turtles**: `01_pond_turtle`, `02_box_turtle`, `03_snapper`, `04_tortoise`, `05_giant_snapper`, `06_alligator_snapper`, `07_war_turtle`, `08_dragon_turtle`
    - Top-down hex sprites (`1024×1024` transparent RGBA PNGs): `assets/monsters/turtle/topdown/01_pond_turtle.png` .. `08_dragon_turtle.png`
    - Stat portraits (`1024×1024` PNGs with unified Jaculus gothic iron-and-bronze frame): `assets/monsters/turtle/portraits/01_pond_turtle.png` .. `08_dragon_turtle.png`
  - **Mammals (Enemies — 13 Non-Evolving Forms)**: `01_rat`, `02_wolf`, `03_bear`, `04_lion`, `05_griffon`, `06_owlbear`, `07_behemoth`, `08_displacer_beast`, `09_manticore`, `10_rajang`, `11_dire_wolf`, `12_sabertooth`, `13_hell_hound`
    - Top-down hex sprites (`1024×1024` transparent RGBA PNGs): `assets/monsters/mammal/topdown/01_rat.png` .. `13_hell_hound.png`
    - Stat portraits (`1024×1024` PNGs with unified Jaculus gothic iron-and-bronze frame): `assets/monsters/mammal/portraits/01_rat.png` .. `13_hell_hound.png`
- **Sound Effects (`assets/sounds/` — 23 `.ogg` Vorbis Audio Files, Managed by `sounds.py`)**:
  - **UI (`assets/sounds/UI/`)**: `positiveclick.ogg`, `negativeclick.ogg`, `victory.ogg`, `defeat.ogg`
  - **Movement (`assets/sounds/moves/`)**: `flight.ogg`, `footsteps.ogg`, `largestomp.ogg`, `softmove.ogg`
  - **Reptile Attacks (`assets/sounds/reptileattacks/`)**: `crunch.ogg`, `dragon.ogg`, `dragtort.ogg`, `hiss.ogg`, `lizard.ogg`, `pecking.ogg`, `snap.ogg`, `snapper.ogg`
  - **Mammal Attacks (`assets/sounds/mammalattacks/`)**: `baboon.ogg`, `deepsnarl.ogg`, `eagle.ogg`, `growl.ogg`, `hound.ogg`, `rat.ogg`, `roar.ogg`
- **Asset Processing Tool**:
  - `tools/make_transparent.m` — Native macOS CoreGraphics/ImageIO utility for extracting transparent RGBA top-down sprites from `#181a1f` slate backgrounds and compositing the master gothic portrait frame (`--composite-frame`).

---

## Pygbag WebAssembly (WASM) Build & Deployment

`main.py` uses an `async def main()` loop (`DenizensApp.run_async`) with `await asyncio.sleep(0)` after `pygame.display.flip()` and seeds `random` from `os.urandom(16)` (`rules.default_rng()`), making the game ready to run in-browser via **Pygbag**.

1. **One-Time SSL Certificate Setup (macOS Terminal session)**:
   ```bash
   pip install pygbag certifi --user
   export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
   ```
2. **Build & Run Local Browser Dev Server (`http://localhost:8000`)**:
   Run from the parent directory (`/Users/madbot/PygameProjects`):
   ```bash
   cd /Users/madbot/PygameProjects
   python3 -m pygbag --disable-sound-format-error Denizens
   ```
3. **Production Web Build Only (no dev server)**:
   ```bash
   cd /Users/madbot/PygameProjects
   python3 -m pygbag --disable-sound-format-error --build Denizens
   ```
   Output files are written to `/Users/madbot/PygameProjects/Denizens/build/web/`.
