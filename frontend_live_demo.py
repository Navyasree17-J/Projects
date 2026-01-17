"""
import cv2
import os
import time
import winsound
from ultralytics import YOLO

# Paths
MODEL_PATH = "models/weapon_last.pt"
ALARM_PATH = "alarms/alarm.wav"
SCREENSHOT_DIR = "screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

WEAPON_CLASSES = ["Gun", "Knife", "Pistol", "Grenade"]

last_alarm_time = 0
ALARM_COOLDOWN = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    boxes = results[0].boxes

    weapon_detected = False
    normal_object_detected = False

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # ❌ Ignore humans completely
            if label.lower() in ["person", "human"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in WEAPON_CLASSES:
                weapon_detected = True

                # 🔴 Weapon box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            else:
                normal_object_detected = True

    # 🔊 Alarm + Screenshot (WEAPON ONLY)
    if weapon_detected:
        current_time = time.time()

        if current_time - last_alarm_time > ALARM_COOLDOWN:
            winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            last_alarm_time = current_time

            # 📸 Screenshot
            screenshot_name = f"{SCREENSHOT_DIR}/weapon_{int(current_time)}.jpg"
            cv2.imwrite(screenshot_name, frame)

    # 🟢 Normal Object box (ONLY if object present & no weapon)
    if normal_object_detected and not weapon_detected:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (80, 80), (w - 80, h - 80), (0, 255, 0), 3)
        cv2.putText(
            frame,
            "Normal Object",
            (90, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow("Abnormal Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""


import cv2
import os
import time
import winsound
from ultralytics import YOLO

# ---------------- PATHS ----------------
WEAPON_MODEL_PATH = "models/weapon_last.pt"
GENERAL_MODEL_PATH = "models/yolov8n.pt"
ALARM_PATH = "alarms/alarm.wav"
SCREENSHOT_DIR = "screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
weapon_model = YOLO(WEAPON_MODEL_PATH)
general_model = YOLO(GENERAL_MODEL_PATH)

cap = cv2.VideoCapture(0)

WEAPON_CLASSES = ["gun", "knife", "pistol", "grenade"]

last_alarm_time = 0
ALARM_COOLDOWN = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    weapon_detected = False
    normal_object_detected = False

    # ---------------- WEAPON DETECTION (UNCHANGED) ----------------
    weapon_results = weapon_model(frame, conf=0.5, verbose=False)
    weapon_boxes = weapon_results[0].boxes

    if weapon_boxes is not None:
        for box in weapon_boxes:
            cls_id = int(box.cls[0])
            label = weapon_model.names[cls_id].lower()
            conf = float(box.conf[0])

            if label in WEAPON_CLASSES:
                weapon_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label.upper()} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # 🔊 Alarm + Screenshot (WEAPON ONLY)
    if weapon_detected:
        current_time = time.time()
        if current_time - last_alarm_time > ALARM_COOLDOWN:
            winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            last_alarm_time = current_time
            cv2.imwrite(f"{SCREENSHOT_DIR}/weapon_{int(current_time)}.jpg", frame)

    # ---------------- NORMAL OBJECT DETECTION ----------------
    if not weapon_detected:
        general_results = general_model(frame, conf=0.4, verbose=False)
        general_boxes = general_results[0].boxes

        if general_boxes is not None:
            for box in general_boxes:
                cls_id = int(box.cls[0])
                label = general_model.names[cls_id].lower()

                # ❌ Ignore humans
                if label == "person":
                    continue

                normal_object_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "NORMAL OBJECT",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Abnormal Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
