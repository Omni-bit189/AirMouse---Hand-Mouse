"""
gestures.py — Pure-logic gesture detection for AirMouse.

No I/O, no side effects.  Takes a list of MediaPipe hand landmarks and
previous state, returns a GestureResult describing what the user is doing.

Modes
─────
  Mouse mode     Cursor control — move, click, drag, scroll.
  Gesture mode   System controls — volume, brightness, mute, play/pause.

Mode switching
──────────────
  Both hands visible + both closed into fists for 3 seconds → toggle mode.

Mouse-mode gestures
───────────────────
  Move        Index finger extended, others curled.
  Left Click  Thumb tip (#4) pinches index tip (#8).  Cursor freezes.
  Right Click Thumb tip (#4) pinches middle tip (#12). Cursor freezes.
  Drag        Pinch (thumb+index) held > DRAG_HOLD_SEC while moving.
  Scroll      Index + middle fingers extended together, hand moves up/down.

Gesture-mode gestures
─────────────────────
  Volume      Index finger extended, circling motion (CW = up, CCW = down).
  Brightness  Index + middle extended, move hand left/right.
  Mute        Fist (all fingers curled) held briefly.
  Play/Pause  Thumbs-up (thumb extended, others curled).
  Keyboard    Toggle  Index + middle + ring extended, pinky curled.
"""

import math
import time
from collections import namedtuple

from config import (
    CLICK_DIST_THRESHOLD,
    RIGHT_CLICK_DIST_THRESHOLD,
    DRAG_HOLD_SEC,
    MODE_SWITCH_HOLD_SEC,
    MODE_SWITCH_COOLDOWN_SEC,
    VOLUME_CIRCLE_THRESHOLD,
    VOLUME_CIRCLE_HISTORY,
    GESTURE_BRIGHTNESS_SENSITIVITY,
    GESTURE_COOLDOWN_SEC,
)


# ── Public data types ─────────────────────────────────────────────────────────

GestureResult = namedtuple(
    "GestureResult",
    ["gesture_name", "cursor_pos", "scroll_delta"],
)

# Landmark indices (MediaPipe Hand model — 21 landmarks)
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
WRIST            = 0
THUMB_CMC        = 1
THUMB_MCP        = 2
THUMB_IP         = 3
THUMB_TIP        = 4
INDEX_MCP        = 5
INDEX_PIP        = 6
INDEX_DIP        = 7
INDEX_TIP        = 8
MIDDLE_MCP       = 9
MIDDLE_PIP       = 10
MIDDLE_DIP       = 11
MIDDLE_TIP       = 12
RING_MCP         = 13
RING_PIP         = 14
RING_DIP         = 15
RING_TIP         = 16
PINKY_MCP        = 17
PINKY_PIP        = 18
PINKY_DIP        = 19
PINKY_TIP        = 20

# Finger definitions: (tip, dip, pip, mcp)
FINGER_LANDMARKS = {
    "index":  (INDEX_TIP,  INDEX_DIP,  INDEX_PIP,  INDEX_MCP),
    "middle": (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
    "ring":   (RING_TIP,   RING_DIP,   RING_PIP,   RING_MCP),
    "pinky":  (PINKY_TIP,  PINKY_DIP,  PINKY_PIP,  PINKY_MCP),
}


# ── Helper functions ──────────────────────────────────────────────────────────

def distance(lm_a, lm_b) -> float:
    """Euclidean distance between two landmarks (uses x, y only)."""
    return math.sqrt((lm_a.x - lm_b.x) ** 2 + (lm_a.y - lm_b.y) ** 2)


def is_finger_extended(landmarks, finger: str) -> bool:
    """
    Check whether a finger is extended (straight) vs curled.

    A finger is considered extended when its tip is farther from the wrist
    than its PIP joint (the middle knuckle).  This simple heuristic works
    well for front-facing hands.
    """
    tip, _dip, pip, _mcp = FINGER_LANDMARKS[finger]
    tip_dist = distance(landmarks[tip], landmarks[WRIST])
    pip_dist = distance(landmarks[pip], landmarks[WRIST])
    return tip_dist > pip_dist


def is_thumb_extended(landmarks) -> bool:
    """
    Thumb extension check — compares tip-to-wrist distance vs IP-to-wrist.
    """
    tip_dist = distance(landmarks[THUMB_TIP], landmarks[WRIST])
    ip_dist  = distance(landmarks[THUMB_IP],  landmarks[WRIST])
    return tip_dist > ip_dist


def is_fist(landmarks) -> bool:
    """
    Return True if the four fingers are curled (fist).

    The thumb is *not* checked because its position is unreliable in a
    natural fist — people tuck it in many different ways.
    """
    for finger in ("index", "middle", "ring", "pinky"):
        if is_finger_extended(landmarks, finger):
            return False
    return True


def is_open_hand(landmarks) -> bool:
    """Return True if all four fingers AND thumb are extended (open palm)."""
    for finger in ("index", "middle", "ring", "pinky"):
        if not is_finger_extended(landmarks, finger):
            return False
    if not is_thumb_extended(landmarks):
        return False
    return True


# ── Circle detection ─────────────────────────────────────────────────────────

def _detect_circle_direction(positions: list) -> tuple[str, float] | None:
    """
    Given a list of (x, y) positions, check if they trace a circle.

    Returns ``("cw", accumulated_angle)`` or ``("ccw", accumulated_angle)``
    if the absolute accumulated angle exceeds ``VOLUME_CIRCLE_THRESHOLD``.
    Returns ``None`` if no circle is detected yet.

    In screen coordinates (y-down), positive accumulated angle = clockwise.
    """
    if len(positions) < 8:
        return None

    # Centroid of recent points
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)

    total_angle = 0.0
    for i in range(1, len(positions)):
        a1 = math.atan2(positions[i - 1][1] - cy, positions[i - 1][0] - cx)
        a2 = math.atan2(positions[i][1] - cy, positions[i][0] - cx)
        delta = a2 - a1
        # Normalise to [-π, π]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        total_angle += delta

    if abs(total_angle) >= VOLUME_CIRCLE_THRESHOLD:
        # In screen coords (y-down): positive total = clockwise
        direction = "cw" if total_angle > 0 else "ccw"
        return (direction, total_angle)

    return None


# ── Mode switching ────────────────────────────────────────────────────────────

def detect_mode_switch(all_hand_landmarks: list, prev_state: dict) -> bool:
    """
    Detect a two-hand mode-switch gesture.

    Both hands must be closed fists and held for MODE_SWITCH_HOLD_SEC seconds.
    Returns True (once) when the switch triggers, then enforces a cooldown.

    Parameters
    ----------
    all_hand_landmarks : list
        List of landmark lists — one per detected hand.
    prev_state : dict
        Mutable state dict (shared with the rest of the gesture system).

    Returns
    -------
    bool
        True if the mode should toggle this frame.
    """
    now = time.time()

    if len(all_hand_landmarks) < 2:
        prev_state.pop("both_fists_start", None)
        return False

    hand_a = all_hand_landmarks[0]
    hand_b = all_hand_landmarks[1]

    # Both hands must be fists
    if is_fist(hand_a) and is_fist(hand_b):
        fist_start = prev_state.get("both_fists_start")
        if fist_start is None:
            prev_state["both_fists_start"] = now
            return False

        held_duration = now - fist_start
        if held_duration >= MODE_SWITCH_HOLD_SEC:
            # Check cooldown
            last_switch = prev_state.get("last_mode_switch", 0.0)
            if now - last_switch > MODE_SWITCH_COOLDOWN_SEC:
                prev_state["last_mode_switch"] = now
                prev_state.pop("both_fists_start", None)
                return True
            # In cooldown — reset timer so it doesn't fire again
            prev_state.pop("both_fists_start", None)
            return False

        return False  # still holding, not long enough yet
    else:
        # Not both fists — reset
        prev_state.pop("both_fists_start", None)
        return False


def get_mode_switch_progress(prev_state: dict) -> float:
    """
    Return 0.0–1.0 indicating how close the user is to triggering a
    mode switch.  Used by the HUD to show a progress indicator.
    """
    fist_start = prev_state.get("both_fists_start")
    if fist_start is None:
        return 0.0
    elapsed = time.time() - fist_start
    return min(1.0, elapsed / MODE_SWITCH_HOLD_SEC)


# ── Mouse-mode detection ─────────────────────────────────────────────────────

def detect_gesture(landmarks, prev_state: dict) -> GestureResult:
    """
    Mouse-mode gesture detection.

    When a click or right-click is detected, the returned ``cursor_pos`` is
    the **pre-pinch** position (the last known "move" position), so the
    cursor doesn't drift during the pinch motion.

    Parameters
    ----------
    landmarks : list
        21 MediaPipe NormalizedLandmark objects.
    prev_state : dict
        Mutable state dict.  Initialise with ``{}`` on the first frame.

    Returns
    -------
    GestureResult
        ``(gesture_name, cursor_pos, scroll_delta)``
    """

    cursor_x = landmarks[INDEX_TIP].x
    cursor_y = landmarks[INDEX_TIP].y

    pinch_dist     = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    pinky_pinch    = distance(landmarks[THUMB_TIP], landmarks[PINKY_TIP])

    index_up  = is_finger_extended(landmarks, "index")
    middle_up = is_finger_extended(landmarks, "middle")
    ring_up   = is_finger_extended(landmarks, "ring")
    pinky_up  = is_finger_extended(landmarks, "pinky")

    now = time.time()

    # ── Keyboard Toggle: index + middle + ring extended, pinky curled ────
    if index_up and middle_up and ring_up and not pinky_up:
        last_kb = prev_state.get("last_keyboard", 0.0)
        if now - last_kb > GESTURE_COOLDOWN_SEC:
            prev_state["last_keyboard"] = now
            return GestureResult("toggle_keyboard", (cursor_x, cursor_y), 0.0)
        
        # Idle state with frozen cursor so it doesn't jump
        frozen_x = prev_state.get("last_move_x", cursor_x)
        frozen_y = prev_state.get("last_move_y", cursor_y)
        return GestureResult("idle", (frozen_x, frozen_y), 0.0)

    # ── Scroll: index + middle extended, others curled ────────────────────
    if index_up and middle_up and not ring_up and not pinky_up:
        mid_y = (landmarks[INDEX_TIP].y + landmarks[MIDDLE_TIP].y) / 2.0
        scroll_ref = prev_state.get("scroll_ref_y")
        if scroll_ref is None:
            prev_state["scroll_ref_y"] = mid_y
            scroll_delta = 0.0
        else:
            scroll_delta = scroll_ref - mid_y
            prev_state["scroll_ref_y"] = mid_y

        prev_state.pop("pinch_start", None)
        prev_state["was_dragging"] = False

        # Update the "safe" cursor position during scroll
        prev_state["last_move_x"] = cursor_x
        prev_state["last_move_y"] = cursor_y

        return GestureResult("scroll", (cursor_x, cursor_y), scroll_delta)

    prev_state.pop("scroll_ref_y", None)

    # ── Freeze cursor position for clicks ────────────────────────────────
    # Use the last known "move" position so the pinch doesn't shift the cursor.
    frozen_x = prev_state.get("last_move_x", cursor_x)
    frozen_y = prev_state.get("last_move_y", cursor_y)

    # ── Right-click: thumb pinches pinky finger ───────────────────────────
    # The pinky is rarely extended during normal usage, making this a very
    # deliberate and reliable gesture that won't false-trigger.
    if pinky_pinch < RIGHT_CLICK_DIST_THRESHOLD:
        if not prev_state.get("was_dragging", False):
            prev_state.pop("pinch_start", None)
            return GestureResult("right_click", (frozen_x, frozen_y), 0.0)

    # ── Left-click / Drag: thumb pinches index finger ────────────────────
    if pinch_dist < CLICK_DIST_THRESHOLD:
        pinch_start = prev_state.get("pinch_start")
        if pinch_start is None:
            prev_state["pinch_start"] = now
            return GestureResult("left_click", (frozen_x, frozen_y), 0.0)

        hold_duration = now - pinch_start
        if hold_duration >= DRAG_HOLD_SEC:
            if not prev_state.get("was_dragging", False):
                prev_state["was_dragging"] = True
                return GestureResult("drag_start", (frozen_x, frozen_y), 0.0)
            # During drag, DO track the finger so the user can drag things
            return GestureResult("dragging", (cursor_x, cursor_y), 0.0)

        return GestureResult("left_click", (frozen_x, frozen_y), 0.0)

    # Pinch released
    if prev_state.get("was_dragging", False):
        prev_state["was_dragging"] = False
        prev_state.pop("pinch_start", None)
        return GestureResult("drag_end", (cursor_x, cursor_y), 0.0)

    prev_state.pop("pinch_start", None)

    # ── Move: index finger up, rest curled ───────────────────────────────
    if index_up:
        # Save this as the "safe" position for click-freeze
        prev_state["last_move_x"] = cursor_x
        prev_state["last_move_y"] = cursor_y
        return GestureResult("move", (cursor_x, cursor_y), 0.0)

    # ── Idle ─────────────────────────────────────────────────────────────
    return GestureResult("idle", (cursor_x, cursor_y), 0.0)


# ── Gesture-mode detection ───────────────────────────────────────────────────

def detect_gesture_mode(landmarks, prev_state: dict) -> GestureResult:
    """
    Gesture-mode detection for system controls.

    Gestures
    --------
    - **Volume**: index finger only → circling motion (CW = up, CCW = down).
    - **Brightness**: index + middle → track horizontal movement.
    - **Mute**: fist held briefly.
    - **Play/Pause**: thumbs-up (thumb extended, others curled).

    Returns
    -------
    GestureResult
        gesture_name is one of: "volume_up", "volume_down",
        "brightness_up", "brightness_down", "mute", "play_pause", "idle"
    """

    cursor_x = landmarks[INDEX_TIP].x
    cursor_y = landmarks[INDEX_TIP].y

    index_up  = is_finger_extended(landmarks, "index")
    middle_up = is_finger_extended(landmarks, "middle")
    ring_up   = is_finger_extended(landmarks, "ring")
    pinky_up  = is_finger_extended(landmarks, "pinky")
    thumb_up  = is_thumb_extended(landmarks)

    now = time.time()

    # ── Thumbs-up: thumb extended, all others curled → play/pause ────────
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        last_pp = prev_state.get("last_play_pause", 0.0)
        if now - last_pp > GESTURE_COOLDOWN_SEC:
            prev_state["last_play_pause"] = now
            prev_state.pop("circle_positions", None)
            prev_state.pop("gm_bri_ref_x", None)
            return GestureResult("play_pause", (cursor_x, cursor_y), 0.0)
        return GestureResult("idle", (cursor_x, cursor_y), 0.0)


    # ── Fist: all fingers curled → mute ──────────────────────────────────
    if is_fist(landmarks):
        last_mute = prev_state.get("last_mute", 0.0)
        if now - last_mute > GESTURE_COOLDOWN_SEC:
            prev_state["last_mute"] = now
            prev_state.pop("circle_positions", None)
            prev_state.pop("gm_bri_ref_x", None)
            return GestureResult("mute", (cursor_x, cursor_y), 0.0)
        return GestureResult("idle", (cursor_x, cursor_y), 0.0)

    # ── Brightness: index + middle extended, others curled ───────────────
    if index_up and middle_up and not ring_up and not pinky_up:
        mid_x = (landmarks[INDEX_TIP].x + landmarks[MIDDLE_TIP].x) / 2.0
        bri_ref = prev_state.get("gm_bri_ref_x")
        if bri_ref is None:
            prev_state["gm_bri_ref_x"] = mid_x
            prev_state.pop("circle_positions", None)
            return GestureResult("idle", (cursor_x, cursor_y), 0.0)

        delta_x = mid_x - bri_ref
        if abs(delta_x) >= GESTURE_BRIGHTNESS_SENSITIVITY:
            prev_state["gm_bri_ref_x"] = mid_x
            if delta_x > 0:
                return GestureResult("brightness_up", (cursor_x, cursor_y), delta_x)
            else:
                return GestureResult("brightness_down", (cursor_x, cursor_y), delta_x)

        return GestureResult("idle", (cursor_x, cursor_y), 0.0)

    prev_state.pop("gm_bri_ref_x", None)

    # ── Volume: index finger only → circling motion ─────────────────────
    if index_up and not middle_up and not ring_up and not pinky_up:
        # Track index fingertip positions
        positions = prev_state.get("circle_positions", [])
        positions.append((landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y))

        # Trim to max history
        if len(positions) > VOLUME_CIRCLE_HISTORY:
            positions = positions[-VOLUME_CIRCLE_HISTORY:]
        prev_state["circle_positions"] = positions

        # Check for circle
        result = _detect_circle_direction(positions)
        if result is not None:
            direction, _angle = result
            # Reset positions after detecting a circle
            prev_state["circle_positions"] = []
            if direction == "cw":
                return GestureResult("volume_up", (cursor_x, cursor_y), 0.0)
            else:
                return GestureResult("volume_down", (cursor_x, cursor_y), 0.0)

        return GestureResult("idle", (cursor_x, cursor_y), 0.0)

    # Not doing volume — clear circle history
    prev_state.pop("circle_positions", None)

    # ── Idle ─────────────────────────────────────────────────────────────
    return GestureResult("idle", (cursor_x, cursor_y), 0.0)
