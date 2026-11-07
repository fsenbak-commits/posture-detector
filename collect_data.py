import cv2
import os
import time
from posture_detectors import pose, get_posture_ratio  # reuse your code

GOOD_LEVEL = 0.92    # above 92% of baseline = clearly good
SLOUCH_LEVEL = 0.80  # below 80% of baseline = clearly slouch
SAVE_EVERY = 0.5     # save at most 2 photos per second
IMG_SIZE = 96        # ESP32 size

os.makedirs("dataset/good", exist_ok=True)
os.makedirs("dataset/slouch", exist_ok=True)

cap = cv2.VideoCapture(0)

# --- 1. Calibrate: sit straight for 5 s ---
print("Sit up straight for 5 seconds...")
values = []
start = time.time()
while time.time() - start < 5:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    r = get_posture_ratio(results, frame)
    if r is not None:
        values.append(r)
    cv2.imshow("Collect", frame)
    cv2.waitKey(1)

if not values:
    print("No pose found. Check the camera.")
    exit()
baseline = sum(values) / len(values)
print("baseline:", baseline)

# --- 2. Collect ---
counts = {"good": 0, "slouch": 0}
last_save = 0

mode=None
mode_start=0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ratio = get_posture_ratio(results, frame)

    # teacher decides the label (None = dead zone, skip)
    label = None
    settled = time.time() - mode_start > 1.0   # 1 s after key press
    if mode is not None and settled and ratio is not None:
        if mode == "good" and ratio > baseline * GOOD_LEVEL:
            label = "good"      # you say good AND the ratio agrees
        elif mode == "slouch":
            label = "slouch"    # trust you

    # save BEFORE drawing anything on the frame
    if label and time.time() - last_save > SAVE_EVERY:
        h, w = frame.shape[:2]
        x0 = (w - h) // 2               # left edge of the centre square
        square = frame[:, x0:x0 + h]    # all rows, middle h columns
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        name = f"dataset/{label}/{label}_{int(time.time() * 1000)}.png"
        cv2.imwrite(name, small)
        counts[label] += 1
        last_save = time.time()

    # now it's safe to draw
    text = f"mode:{mode or 'paused'}  good:{counts['good']}  slouch:{counts['slouch']}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Collect", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("g"):
        mode = "good"
        mode_start = time.time()
    elif key == ord("s"):
        mode = "slouch"
        mode_start = time.time()
    elif key == ord("p"):
        mode = None          # pause, so you can move freely
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Saved:", counts)