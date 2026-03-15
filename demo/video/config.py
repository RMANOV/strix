"""STRIX Demo Video — Global configuration."""

# ── Video ────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION = 240  # seconds
TOTAL_FRAMES = FPS * DURATION  # 7200
CRF = 18
OUTPUT_PATH = "/home/rmanov/strix/demo/video/strix_demo.mp4"

# ── Colors (RGBA 0.0-1.0 for cairo) ─────────────
BG = (0.039, 0.039, 0.071, 1.0)          # #0a0a12
CYAN = (0.0, 0.831, 1.0, 1.0)            # #00d4ff
GREEN = (0.0, 1.0, 0.533, 1.0)           # #00ff88
RED = (1.0, 0.2, 0.267, 1.0)             # #ff3344
WHITE = (1.0, 1.0, 1.0, 1.0)
DIM_WHITE = (0.7, 0.7, 0.8, 1.0)
AMBER = (1.0, 0.75, 0.0, 1.0)            # #ffbf00
YELLOW = (1.0, 0.9, 0.0, 1.0)
BLUE = (0.2, 0.4, 1.0, 1.0)

# Dimmed variants (for backgrounds, zones)
CYAN_DIM = (*CYAN[:3], 0.15)
GREEN_DIM = (*GREEN[:3], 0.12)
RED_DIM = (*RED[:3], 0.15)

# ── Scene frame ranges (start, end exclusive) ───
SCENES = {
    1: (0, 900),       # 0:00–0:30  Radio Command
    2: (900, 2700),    # 0:30–1:30  Reconnaissance
    3: (2700, 4500),   # 1:30–2:30  Loss & Recovery
    4: (4500, 6300),   # 2:30–3:30  Adversarial Prediction
    5: (6300, 7200),   # 3:30–4:00  Scale + Closing
}
CROSSFADE_FRAMES = 30  # 1-second transition

# ── Typography ───────────────────────────────────
FONT_FAMILY = "monospace"
FONT_SIZE_LARGE = 36
FONT_SIZE_MEDIUM = 24
FONT_SIZE_SMALL = 16
FONT_SIZE_TINY = 12

# ── Tactical map ─────────────────────────────────
MAP_WORLD_W, MAP_WORLD_H = 4000.0, 3000.0
MAP_GRID_SPACING = 500  # meters

# SAM corridor bounds (world coords)
SAM_ZONE = (1600, 800, 600, 1400)  # x, y, w, h

# Base location
BASE_POS = (200, 2600)

# ── Drones ───────────────────────────────────────
DRONE_ICON_SIZE = 12
NUM_RECON_DRONES = 4
MAX_PARTICLES = 3000

# ── Audio ────────────────────────────────────────
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2

# ── Bid formula weights (Scene 3) ────────────────
BID_WEIGHTS = {
    "urgency": 10,
    "capability": 3,
    "proximity": 5,
    "energy": 2,
    "risk": -4,
}
