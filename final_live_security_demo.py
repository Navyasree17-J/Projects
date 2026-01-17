import cv2
import time
import winsound
import os
from ultralytics import YOLO
from collections import deque

# ---------------- MODELS ----------------
weapon_model = YOLO("models/weapon_last.pt")
person_model = YOLO("yolov8n.pt")

# ---------------- PATHS ----------------
WEAPON_ALARM = "alarms/alarm.wav"
CROWD_ALARM = "alarms/crowd_alarm.wav"

# ---------------- SETTINGS ----------------
WEAPON_CLASSES = ["knife", "gun", "pistol", "grenade"]
CROWD_THRESHOLD = 5
FRAME_WINDOW = 10
ALARM_COOLDOWN = 5  # seconds

people_history = deque(maxlen=FRAME_WINDOW)
last_weapon_alarm = 0
last_crowd_alarm = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ---------- WEAPON DETECTION ----------
    weapon_results = weapon_model(frame, conf=0.5)
    weapon_found = False

    for box in weapon_results[0].boxes or []:
        cls = int(box.cls[0])
        label = weapon_model.names[cls].lower()

        if label in WEAPON_CLASSES:
            weapon_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label.upper(),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

    # 🔊 Weapon alarm (independent)
    if weapon_found and (current_time - last_weapon_alarm > ALARM_COOLDOWN):
        winsound.PlaySound(WEAPON_ALARM, winsound.SND_FILENAME | winsound.SND_ASYNC)
        last_weapon_alarm = current_time

    # ---------- CROWD DETECTION ----------
    person_results = person_model(frame, conf=0.4, verbose=False)
    count = 0

    for r in person_results:
        for box in r.boxes:
            if person_model.names[int(box.cls[0])] == "person":
                count += 1

    people_history.append(count)
    max_people = max(people_history)

    # 🔊 Crowd alarm (independent)
    crowd_anomaly = max_people > CROWD_THRESHOLD
    if crowd_anomaly and (current_time - last_crowd_alarm > ALARM_COOLDOWN):
        winsound.PlaySound(CROWD_ALARM, winsound.SND_FILENAME | winsound.SND_ASYNC)
        last_crowd_alarm = current_time

    # ---------- STATUS DISPLAY ----------
    if weapon_found:
        cv2.putText(frame, "ABNORMAL OBJECT DETECTED",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    elif crowd_anomaly:
        cv2.putText(frame, "CROWD ANOMALY DETECTED",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    else:
        cv2.putText(frame, "STATUS: NORMAL",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

    cv2.imshow("Final Live Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
