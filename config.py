"""
config.py — Central configuration for AirMouse hand-gesture controller.

All tunable constants live here so every other module can
    from config import *
and pick up the same values.
"""

import pyautogui

# ── Screen ────────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_INDEX = 0

# ── Cursor smoothing ─────────────────────────────────────────────────────────
# Exponential moving average factor (0–1).
# Lower  → smoother but laggier.
# Higher → more responsive but jittery.
SMOOTHING = 0.7

# ── Gesture thresholds ───────────────────────────────────────────────────────
# Max normalised distance between thumb-tip (#4) and index-tip (#8)
# to register as a "pinch" → left click.
CLICK_DIST_THRESHOLD = 0.045

# Max normalised distance between thumb-tip (#4) and middle-tip (#12)
# to register as a right-click pinch.
RIGHT_CLICK_DIST_THRESHOLD = 0.045

# Minimum seconds between repeated click actions.
CLICK_COOLDOWN_SEC = 0.4

# How many pixels of vertical index-finger movement map to one scroll tick.
SCROLL_SENSITIVITY = 100

# Ignore hand movements smaller than this (normalised units) to kill jitter.
DEADZONE = 0.005

# ── Frame mapping ────────────────────────────────────────────────────────────
# Fraction of the camera frame to ignore on *each* edge (0–0.5).
# e.g. 0.15 means the usable area starts 15 % from the left/top and ends
# 15 % before the right/bottom, so you don't have to stretch your hand to
# the very edge of the webcam view.
FRAME_REDUCTION = 0.15

# ── Drag ─────────────────────────────────────────────────────────────────────
# How long (seconds) a pinch must be held before it becomes a drag.
DRAG_HOLD_SEC = 0.3

# ── Mode switching ───────────────────────────────────────────────────────────
# How long (seconds) both fists must be held to toggle mode.
MODE_SWITCH_HOLD_SEC = 3.0

# Cooldown after a mode switch to prevent rapid re-toggling.
MODE_SWITCH_COOLDOWN_SEC = 2.0

# ── Gesture mode ─────────────────────────────────────────────────────────────
# Circling motion for volume: angular threshold (radians) per volume step.
# ~5.0 rad ≈ 80 % of a full circle.  Lower = more sensitive.
VOLUME_CIRCLE_THRESHOLD = 5.0

# Max number of recent finger positions to track for circle detection.
VOLUME_CIRCLE_HISTORY = 40

# Horizontal hand movement (normalised) needed per brightness step.
GESTURE_BRIGHTNESS_SENSITIVITY = 0.03

# Brightness change (%) per step.
GESTURE_BRIGHTNESS_STEP = 5

# Cooldown between discrete gesture-mode actions (mute, play/pause).
GESTURE_COOLDOWN_SEC = 0.6
