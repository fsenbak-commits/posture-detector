# Posture Detector 

A real-time posture monitor that watches you through your webcam while you work and sends a native notification when you've been slouching for too long.



## How it works

1. **Calibration** — when you start the app, sit up straight for 5 seconds. It records your personal "good posture" baseline.
2. **Live monitoring** — using [MediaPipe Pose](https://developers.google.com/mediapipe) to track body landmarks in real time, it continuously measures the vertical gap between your ear and shoulders, normalized by shoulder width.
3. **Debounced alerts** — if that ratio drops below your calibrated threshold and stays there for more than 5 seconds (to avoid false alarms from brief movements), you get a native macOS notification reminding you to sit up straight.

## Tech stack

- Python
- [OpenCV](https://opencv.org/) — webcam capture and display
- [MediaPipe](https://developers.google.com/mediapipe) — pretrained pose-landmark detection
- `osascript` — native macOS notifications

## Setup

```bash
git clone https://github.com/fsenbak-commits/posture-detector.git
cd posture-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 posture_detectors.py
```

On first run, macOS will ask for camera permission — click Allow.

## Usage

- Sit normally in front of your webcam when the app starts.
- Stay still and upright during the 5-second calibration.
- Keep working as usual — you'll get a notification if you slouch for more than 5 seconds continuously.
- Press `q` with the video window focused to quit.

## How the posture metric works

The core signal is a single number per frame:

```
ratio = (average_shoulder_height - ear_height) / shoulder_width
```

- Higher ratio → more upright (ear is well above shoulders)
- Lower ratio → more slouched (ear drops toward shoulder height, as happens with forward head posture)

Shoulder width is used to normalize the measurement so it stays consistent regardless of how close or far you are from the camera.

## Known limitations

- **Only detects forward head posture** (leaning toward the screen). It does not currently detect shoulder slump where the head stays in place but shoulders round forward/down — a possible future improvement.
- Accuracy depends on webcam angle and lighting; a front-facing, well-lit setup works best.
- Calibration is per-session — if you change seating position significantly, consider restarting the app to recalibrate.

## Possible future improvements

- Track shoulder-slump posture in addition to forward head posture
- Menu bar icon instead of a visible video window
- Posture history logging/dashboard
- Adjustable sensitivity via a settings file or UI

