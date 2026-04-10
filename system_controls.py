"""
system_controls.py — OS-level system controls for AirMouse gesture mode.

Provides functions for volume, brightness, and media playback control.
All platform-specific logic is isolated here.
"""

import pyautogui

try:
    import screen_brightness_control as sbc
    _HAS_SBC = True
except ImportError:
    _HAS_SBC = False

from config import GESTURE_BRIGHTNESS_STEP


# ── Volume ───────────────────────────────────────────────────────────────────

def volume_up(steps: int = 1) -> None:
    """Increase system volume by *steps* increments."""
    pyautogui.press("volumeup", presses=max(1, steps))


def volume_down(steps: int = 1) -> None:
    """Decrease system volume by *steps* increments."""
    pyautogui.press("volumedown", presses=max(1, steps))


def toggle_mute() -> None:
    """Toggle mute / unmute."""
    pyautogui.press("volumemute")


# ── Brightness ───────────────────────────────────────────────────────────────

def get_brightness() -> int:
    """Return current display brightness (0–100). Returns -1 if unavailable."""
    if not _HAS_SBC:
        return -1
    try:
        val = sbc.get_brightness(display=0)
        # sbc.get_brightness may return a list or int depending on version
        if isinstance(val, list):
            return val[0]
        return val
    except Exception:
        return -1


def brightness_up() -> int:
    """Increase brightness by GESTURE_BRIGHTNESS_STEP %. Returns new level."""
    if not _HAS_SBC:
        return -1
    try:
        current = get_brightness()
        if current < 0:
            return -1
        new_val = min(100, current + GESTURE_BRIGHTNESS_STEP)
        sbc.set_brightness(new_val, display=0)
        return new_val
    except Exception:
        return -1


def brightness_down() -> int:
    """Decrease brightness by GESTURE_BRIGHTNESS_STEP %. Returns new level."""
    if not _HAS_SBC:
        return -1
    try:
        current = get_brightness()
        if current < 0:
            return -1
        new_val = max(0, current - GESTURE_BRIGHTNESS_STEP)
        sbc.set_brightness(new_val, display=0)
        return new_val
    except Exception:
        return -1


# ── Media ────────────────────────────────────────────────────────────────────

def play_pause() -> None:
    """Toggle media play / pause."""
    pyautogui.press("playpause")


# ── Keyboard ─────────────────────────────────────────────────────────────────

import subprocess

def toggle_keyboard() -> None:
    """Toggle the Windows On-Screen Keyboard (osk.exe)."""
    try:
        output = subprocess.check_output('tasklist', shell=True).decode()
        if 'osk.exe' in output:
            subprocess.run(['taskkill', '/F', '/IM', 'osk.exe'], capture_output=True)
        else:
            # We use shell=True and start because directly popping osk sometimes fails on 64-bit python
            subprocess.Popen('start osk.exe', shell=True)
    except Exception as e:
        print(f"[OSK] Could not toggle keyboard: {e}")
