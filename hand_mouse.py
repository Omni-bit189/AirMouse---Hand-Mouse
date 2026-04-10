"""
hand_mouse.py — AirMouse: control your desktop with hand gestures.

Usage
─────
    python hand_mouse.py                   # default camera, default settings
    python hand_mouse.py --cam 1           # use a different camera
    python hand_mouse.py --smoothing 0.5   # override cursor smoothing
    python hand_mouse.py --no-preview      # hide the OpenCV preview window
    python hand_mouse.py --debug           # show FPS, distances, landmark IDs

Press  Q  in the live preview window (or Ctrl+C in the terminal) to quit.

Modes
─────
  Mouse Mode   (default, green)  — move, click, drag, scroll
  Gesture Mode (purple)          — volume, brightness, mute, play/pause

Toggle modes: show both hands, close one into a fist.

Pipeline
────────
  Webcam  →  MediaPipe Hand Landmarker  →  Gesture Detection  →  Actions
     │           (tracks 2 hands)                                    │
     └──────────────────── HUD overlay ◄─────────────────────────────┘
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import (
    CAM_INDEX,
    CLICK_COOLDOWN_SEC,
    DEADZONE,
    FRAME_REDUCTION,
    SCREEN_H,
    SCREEN_W,
    SCROLL_SENSITIVITY,
    SMOOTHING,
)
from gestures import (
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    detect_gesture,
    detect_gesture_mode,
    detect_mode_switch,
    get_mode_switch_progress,
    distance,
)
from system_controls import (
    brightness_down,
    brightness_up,
    get_brightness,
    play_pause,
    toggle_mute,
    volume_down,
    volume_up,
    toggle_keyboard,
)


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"

# MediaPipe hand-skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

# Mode colors
COLOR_MOUSE   = (0, 255, 0)      # green
COLOR_GESTURE = (200, 100, 255)   # purple
COLOR_SWITCH  = (0, 255, 255)     # yellow flash

# PyAutoGUI fail-safe: moving cursor to corner raises exception (safety net).
# Disabled to avoid crashing while dragging near the edge of the screen.
pyautogui.FAILSAFE = False
# Disable the built-in pause so gestures feel instant.
pyautogui.PAUSE = 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AirMouse — hand-gesture desktop controller",
    )
    p.add_argument("--cam", type=int, default=CAM_INDEX,
                   help="Camera device index (default: %(default)s)")
    p.add_argument("--smoothing", type=float, default=SMOOTHING,
                   help="EMA smoothing factor 0–1 (default: %(default)s)")
    p.add_argument("--no-preview", action="store_true",
                   help="Hide the OpenCV preview window (lower CPU)")
    p.add_argument("--debug", action="store_true",
                   help="Show extra HUD info (FPS, distances, landmark IDs)")
    return p.parse_args()


# ── Coordinate mapping ───────────────────────────────────────────────────────

def map_to_screen(norm_x: float, norm_y: float) -> tuple[float, float]:
    """
    Map a normalised landmark coordinate (0–1, within camera frame) to
    screen pixels, accounting for the configurable frame-reduction margin.
    """
    lo = FRAME_REDUCTION
    hi = 1.0 - FRAME_REDUCTION

    x = (np.clip(norm_x, lo, hi) - lo) / (hi - lo)
    y = (np.clip(norm_y, lo, hi) - lo) / (hi - lo)

    return float(x * SCREEN_W), float(y * SCREEN_H)


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_hand_skeleton(frame, landmarks, mode: str, debug: bool = False):
    """Draw the 21-point hand skeleton onto *frame*."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Connection color based on mode
    conn_color = (150, 255, 150) if mode == "mouse" else (200, 160, 255)

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], conn_color, 2, cv2.LINE_AA)

    for i, (px, py) in enumerate(pts):
        color = (0, 255, 255)  # default: cyan
        if i == THUMB_TIP:
            color = (0, 140, 255)   # orange
        elif i == INDEX_TIP:
            color = (0, 255, 0)     # green
        elif i == MIDDLE_TIP:
            color = (255, 100, 0)   # blue-ish

        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

        if debug:
            cv2.putText(frame, str(i), (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)


def draw_mode_badge(frame, mode: str):
    """Draw a mode indicator badge in the top-right corner."""
    h, w = frame.shape[:2]
    label = "MOUSE" if mode == "mouse" else "GESTURE"
    color = COLOR_MOUSE if mode == "mouse" else COLOR_GESTURE
    bg_color = (20, 80, 20) if mode == "mouse" else (80, 30, 80)

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    pad = 10
    x1 = w - text_size[0] - pad * 2 - 10
    y1 = 6
    x2 = w - 10
    y2 = y1 + text_size[1] + pad * 2

    # Background with rounded feel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    cv2.putText(frame, label, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_gesture_mode_bars(frame, gesture_name: str, brightness: int):
    """Draw volume/brightness status info in gesture mode."""
    h, w = frame.shape[:2]
    y_base = h - 60

    # Show what gesture is active
    if gesture_name.startswith("volume"):
        icon = "VOL"
        icon_color = (0, 220, 255)
    elif gesture_name.startswith("brightness"):
        icon = "BRI"
        icon_color = (0, 255, 255)
    elif gesture_name == "mute":
        icon = "MUTE"
        icon_color = (0, 0, 255)
    elif gesture_name == "play_pause":
        icon = "PLAY/PAUSE"
        icon_color = (255, 200, 0)
    elif gesture_name == "toggle_keyboard":
        icon = "OSK"
        icon_color = (255, 0, 255)
    else:
        icon = ""
        icon_color = (180, 180, 180)

    if icon:
        # Action flash
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_base - 10), (200, y_base + 30), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, icon, (20, y_base + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, icon_color, 2, cv2.LINE_AA)

    # Brightness bar if available
    if brightness >= 0:
        bar_x = w - 50
        bar_h = 120
        bar_y = h - bar_h - 20
        bar_w = 20

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        # Fill
        fill_h = int(bar_h * brightness / 100)
        cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h),
                      (bar_x + bar_w, bar_y + bar_h),
                      (0, 255, 255), -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (200, 200, 200), 1)
        # Label
        cv2.putText(frame, f"{brightness}%", (bar_x - 10, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, "BRI", (bar_x - 2, bar_y + bar_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                    cv2.LINE_AA)


def draw_hud(frame, gesture_name: str, mode: str, fps: float, debug: bool,
             brightness: int = -1,
             pinch_dist: float = 0.0, right_dist: float = 0.0):
    """Draw the heads-up display."""
    h, w = frame.shape[:2]

    # Gesture-name colours
    mouse_colors = {
        "move":        (0, 255, 0),
        "left_click":  (0, 200, 255),
        "right_click": (0, 100, 255),
        "drag_start":  (255, 100, 0),
        "dragging":    (255, 50, 50),
        "drag_end":    (200, 200, 0),
        "scroll":      (255, 255, 0),
        "idle":        (180, 180, 180),
    }
    gesture_colors = {
        "volume_up":       (0, 220, 255),
        "volume_down":     (0, 180, 255),
        "brightness_up":   (0, 255, 255),
        "brightness_down": (0, 200, 200),
        "mute":            (0, 0, 255),
        "play_pause":      (255, 200, 0),
        "toggle_keyboard": (255, 0, 255),
        "idle":            (180, 180, 180),
    }
    colors = mouse_colors if mode == "mouse" else gesture_colors
    color = colors.get(gesture_name, (255, 255, 255))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    nice_name = gesture_name.replace("_", " ").upper()
    cv2.putText(frame, f"Gesture: {nice_name}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # Mode badge
    draw_mode_badge(frame, mode)

    # Gesture-mode bars
    if mode == "gesture":
        draw_gesture_mode_bars(frame, gesture_name, brightness)

    if debug:
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)

        cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (12, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f"R-Pinch: {right_dist:.3f}", (12, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)


def draw_cursor_dot(frame, norm_x: float, norm_y: float):
    """Draw a bright dot where the cursor is mapped on the preview."""
    h, w = frame.shape[:2]
    cx, cy = int(norm_x * w), int(norm_y * h)
    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)


def draw_mode_switch_flash(frame):
    """Brief yellow border flash to confirm mode switch."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_SWITCH, 4, cv2.LINE_AA)
    cv2.putText(frame, "MODE SWITCHED!", (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_SWITCH, 3, cv2.LINE_AA)


def draw_mode_switch_progress(frame, progress: float):
    """Draw a progress bar showing how close the user is to switching modes."""
    if progress <= 0.01:
        return
    h, w = frame.shape[:2]
    bar_w = 300
    bar_h = 20
    x = (w - bar_w) // 2
    y = h - 50

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 25), (x + bar_w + 5, y + bar_h + 5),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Label
    cv2.putText(frame, "Switching mode...", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Track
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                  (80, 80, 80), -1)
    # Fill
    fill_w = int(bar_w * progress)
    color = (
        (0, 255, 255) if progress < 0.7
        else (0, 200, 255) if progress < 0.9
        else (0, 255, 0)
    )
    cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                  (200, 200, 200), 1, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate model
    if not os.path.exists(MODEL_PATH):
        print(
            f"[ERROR] Hand landmarker model not found at '{MODEL_PATH}'.\n"
            "Download it with:\n"
            "  curl -o hand_landmarker.task -L "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task",
            file=sys.stderr,
        )
        sys.exit(1)

    # MediaPipe setup — track 2 hands for mode switching
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Open camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*56}")
    print(f"  AirMouse — Hand Gesture Desktop Controller")
    print(f"{'='*56}")
    print(f"  Camera        : {args.cam}")
    print(f"  Screen        : {SCREEN_W}×{SCREEN_H}")
    print(f"  Smoothing     : {args.smoothing}")
    print(f"  Preview       : {'OFF' if args.no_preview else 'ON'}")
    print(f"  Debug HUD     : {'ON' if args.debug else 'OFF'}")
    print(f"  Mode switch   : Both fists held 3 seconds")
    print(f"  Press Q to quit")
    print(f"{'='*56}\n")

    # ── State ─────────────────────────────────────────────────────────────
    current_mode = "mouse"          # "mouse" or "gesture"
    smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
    prev_state: dict = {}           # mutable state for gesture detector
    last_click_time = 0.0
    last_right_click_time = 0.0
    frame_timestamp = 0
    fps = 0.0
    fps_timer = time.time()
    frame_count = 0
    mode_switch_flash = 0           # frames remaining for flash animation
    current_brightness = get_brightness()

    alpha = args.smoothing  # EMA factor

    with mp_vision.HandLandmarker.create_from_options(hand_opts) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror
            frame_timestamp += int(1000 / 30)

            # Run hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_timestamp)

            display = frame.copy()
            gesture_name = "idle"
            pinch_dist = 0.0
            right_dist = 0.0
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

            # ── Mode switch check (needs 2 hands, both fists, 3s hold) ──
            if num_hands >= 2:
                if detect_mode_switch(result.hand_landmarks, prev_state):
                    current_mode = "gesture" if current_mode == "mouse" else "mouse"
                    mode_switch_flash = 15  # ~0.5s at 30fps
                    # Reset gesture-specific state on mode switch
                    for key in list(prev_state.keys()):
                        if key not in ("both_fists_start", "last_mode_switch"):
                            prev_state.pop(key, None)
                    print(f"[MODE] Switched to {current_mode.upper()} mode")

            # Mode switch progress (for HUD)
            mode_switch_progress = get_mode_switch_progress(prev_state)

            # ── Use the first detected hand for control ──────────────────
            if num_hands >= 1:
                landmarks = result.hand_landmarks[0]

                if current_mode == "mouse":
                    # ── MOUSE MODE ───────────────────────────────────────
                    gr = detect_gesture(landmarks, prev_state)
                    gesture_name = gr.gesture_name
                    raw_x, raw_y = gr.cursor_pos

                    # Map to screen
                    screen_x, screen_y = map_to_screen(raw_x, raw_y)

                    # Apply dead-zone
                    dx = abs(screen_x - smooth_x) / SCREEN_W
                    dy = abs(screen_y - smooth_y) / SCREEN_H
                    if dx > DEADZONE or dy > DEADZONE:
                        smooth_x = alpha * screen_x + (1 - alpha) * smooth_x
                        smooth_y = alpha * screen_y + (1 - alpha) * smooth_y

                    # Clamp
                    smooth_x = max(0, min(SCREEN_W - 1, smooth_x))
                    smooth_y = max(0, min(SCREEN_H - 1, smooth_y))

                    now = time.time()

                    # Execute mouse action
                    if gesture_name == "move":
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))

                    elif gesture_name == "left_click":
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                        if now - last_click_time > CLICK_COOLDOWN_SEC:
                            pyautogui.click()
                            last_click_time = now

                    elif gesture_name == "right_click":
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                        if now - last_right_click_time > CLICK_COOLDOWN_SEC:
                            pyautogui.rightClick()
                            last_right_click_time = now

                    elif gesture_name == "drag_start":
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                        pyautogui.mouseDown()

                    elif gesture_name == "dragging":
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))

                    elif gesture_name == "drag_end":
                        pyautogui.mouseUp()

                    elif gesture_name == "scroll":
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                        ticks = int(gr.scroll_delta * SCROLL_SENSITIVITY)
                        if ticks != 0:
                            pyautogui.scroll(ticks)

                    # Debug distances
                    pinch_dist = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
                    right_dist = distance(landmarks[THUMB_TIP], landmarks[MIDDLE_TIP])

                    # Draw on preview
                    if not args.no_preview:
                        draw_hand_skeleton(display, landmarks, "mouse",
                                           debug=args.debug)
                        draw_cursor_dot(display, raw_x, raw_y)

                else:
                    # ── GESTURE MODE ─────────────────────────────────────
                    gr = detect_gesture_mode(landmarks, prev_state)
                    gesture_name = gr.gesture_name

                    # Execute system action
                    if gesture_name == "volume_up":
                        volume_up()
                    elif gesture_name == "volume_down":
                        volume_down()
                    elif gesture_name == "brightness_up":
                        current_brightness = brightness_up()
                    elif gesture_name == "brightness_down":
                        current_brightness = brightness_down()
                    elif gesture_name == "mute":
                        toggle_mute()
                    elif gesture_name == "play_pause":
                        play_pause()
                    elif gesture_name == "toggle_keyboard":
                        toggle_keyboard()

                    # Draw on preview
                    if not args.no_preview:
                        draw_hand_skeleton(display, landmarks, "gesture",
                                           debug=args.debug)

            else:
                # No hand detected — reset drag state
                if prev_state.get("was_dragging", False):
                    pyautogui.mouseUp()
                    prev_state["was_dragging"] = False
                # Keep mode switch state, clear gesture state
                mode_keys = ("both_fists_start", "last_mode_switch")
                saved = {k: prev_state[k] for k in mode_keys if k in prev_state}
                prev_state.clear()
                prev_state.update(saved)

            # Draw all detected hands in skeleton (draw 2nd hand if present)
            if not args.no_preview and num_hands >= 2:
                draw_hand_skeleton(display, result.hand_landmarks[1],
                                   current_mode, debug=args.debug)

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Mode switch progress bar
            if not args.no_preview and mode_switch_progress > 0.01:
                draw_mode_switch_progress(display, mode_switch_progress)

            # Mode switch flash animation
            if mode_switch_flash > 0:
                draw_mode_switch_flash(display)
                mode_switch_flash -= 1

            # Show preview
            if not args.no_preview:
                draw_hud(display, gesture_name, current_mode, fps, args.debug,
                         brightness=current_brightness,
                         pinch_dist=pinch_dist, right_dist=right_dist)
                cv2.imshow("AirMouse", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # Cleanup — make sure mouse is released
    if prev_state.get("was_dragging", False):
        pyautogui.mouseUp()

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] AirMouse stopped. Goodbye!")


if __name__ == "__main__":
    main()
