# AirMouse — Hand Gesture Desktop Controller

AirMouse is a desktop automation tool that transforms your webcam into a dual-mode smart controller. It uses **MediaPipe** and **OpenCV** to track up to two hands in real time, letting you control your mouse cursor, clicking, dragging, and scrolling, as well as managing system events like volume, screen brightness, and media playback entirely through hand gestures!

## 🚀 Features
* **Dual-Mode Architecture**: Seamlessly switch between **Mouse Mode** (cursor control) and **Gesture Mode** (system controls) using a simple two-hand gesture.
* **Smart Filtering**: Built-in exponential moving average (EMA) smoothing and deadzones to eliminate cursor jitter.
* **Real-Time HUD**: On-screen heads-up display showing active mode, active gesture, and progress bars.
* **No Extra Hardware required**: Just an ordinary webcam.

---

## 🛠️ Setup & Installation

### 1. Requirements
Ensure you have Python 3.10+ installed.

Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download the MediaPipe Model
The app relies on Google's MediaPipe Hand Landmarker model. It must be present in the project root:
```bash
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. Run
```bash
python hand_mouse.py
```
*(Optional flags available: `--cam 1` for an external webcam, `--smoothing 0.5` to adjust cursor smoothing, `--debug` for more stats, `--no-preview` to hide the camera view).*

---

## 👐 Gestures Guide

### ↔️ Mode Switching
This toggle works universally, no matter what mode you are currently in.
* **Switch Modes**: Hold **Both hands in a fist** for 3 seconds. A progress bar will fill up on screen, and a yellow flash will confirm the toggle.

---

### 🟢 Mouse Mode (Green Badge)
*Default mode. Designed to replace your physical mouse trackpad/mouse.*

| Intended Action | Hand Shape / Gesture | How it works |
| :--- | :--- | :--- |
| **Move Cursor** | ☝️ **Index finger extended** (others curled) | The cursor tracks your index fingertip. Move your hand naturally around the webcam frame. |
| **Left Click** | 🤏 **Index + Thumb pinch** | Pinch your thumb tip to your index fingertip. The cursor will freeze in place so it doesn't drift when you pinch. |
| **Drag & Drop** | 🤌 **Index + Thumb pinch (Hold)** | Pinch and hold for 0.3 seconds. You can then move your hand to drag the item. Release the pinch to drop. |
| **Right Click** | 🤙 **Pinky + Thumb pinch** (Others kept up or relaxed) | Very deliberate gesture — touch your thumb tip to your pinky tip. |
| **Scroll** | ✌️ **Index + Middle extended** (Others curled) | Move your entire hand **Up or Down** in the air to scroll the page. |
| **Toggle Keyboard** | 🖖 **Three Fingers Up** (Index, Middle, Ring extended) | Closes the Windows On-Screen Keyboard if open, or launches it if closed. |

---

### 🟣 Gesture Mode (Purple Badge)
*System controls mode. The cursor is ignored here while it detects media controls.*

| Intended Action | Hand Shape / Gesture | How it works |
| :--- | :--- | :--- |
| **Volume Control** | ☝️ **Index finger extended** (draw a circle) | Trace a circle in the air. **Clockwise** turns volume up. **Counter-clockwise** turns volume down. |
| **Brightness**| ✌️ **Index + Middle extended** (move horizontally) | Move your hand **Left** to lower screen brightness, and **Right** to increase it. A brightness bar will display on the screen. |
| **Mute / Unmute** | ✊ **Single Fist** | Briefly hold up a fist. It acts as a toggle. |
| **Play / Pause** | 👍 **Thumbs-Up** (Thumb out, others curled) | Instantly plays or pauses whatever media/video is currently active on your computer. |

---

## ⚙️ Configuration
All thresholds, sensitivities, and configurations are exposed in `config.py`. You can adjust them perfectly to your specific room lighting, camera width, and personal preference.

---
**Disclaimer**: Requires a fairly well-lit room for flawless 60fps tracking. Brightness control module requires a Windows environment.
