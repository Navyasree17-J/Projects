"""
import cv2
import time
import os
import winsound
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
WEAPON_MODEL_PATH = "models/weapon_last.pt"
PERSON_MODEL_PATH = "models/yolov8n.pt"

WEAPON_ALARM = "alarms/weapon_alarm.wav"
CROWD_ALARM = "alarms/crowd_alarm.wav"

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
weapon_model = YOLO(WEAPON_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

# ---------------- SETTINGS ----------------
WEAPON_CLASSES = ["knife", "gun", "pistol", "grenade", "weapon"]
CONF_WEAPON = 0.5
CONF_PERSON = 0.4

CROWD_THRESHOLD = 5
FRAME_WINDOW = 10
ALARM_COOLDOWN = 5

# ---------------- STORAGE ----------------
people_history = deque(maxlen=FRAME_WINDOW)
frame_count = 0
graph_x, graph_y = [], []

last_weapon_alarm = 0
last_crowd_alarm = 0

# ---------------- GRAPH ----------------
plt.ion()
fig, ax = plt.subplots()

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("✅ Final Security System Started")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1

    # ======================================================
    # 🔴 WEAPON + NORMAL OBJECT DETECTION
    # ======================================================
    weapon_found = False
    normal_object_found = False

    weapon_results = weapon_model(frame, conf=CONF_WEAPON, verbose=False)

    for box in weapon_results[0].boxes or []:
        cls = int(box.cls[0])
        label = weapon_model.names[cls].lower().strip()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Ignore humans if present
        if label == "person":
            continue

        # 🔴 Abnormal object
        if label in WEAPON_CLASSES:
            weapon_found = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label.upper(),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

            if current_time - last_weapon_alarm > ALARM_COOLDOWN:
                winsound.PlaySound(WEAPON_ALARM, winsound.SND_FILENAME | winsound.SND_ASYNC)
                cv2.imwrite(f"{SCREENSHOT_DIR}/weapon_{int(current_time)}.jpg", frame)
                last_weapon_alarm = current_time

        # 🟢 Normal object
        else:
            normal_object_found = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "NORMAL OBJECT",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    # ======================================================
    # 👥 CROWD COUNT & ANOMALY
    # ======================================================
    person_count = 0
    person_results = person_model(frame, conf=CONF_PERSON, verbose=False)

    for r in person_results:
        for box in r.boxes:
            if person_model.names[int(box.cls[0])] == "person":
                person_count += 1

    people_history.append(person_count)
    max_people = max(people_history)

    # Graph update
    graph_x.append(frame_count)
    graph_y.append(person_count)

    ax.clear()
    ax.plot(graph_x, graph_y)
    ax.axhline(y=CROWD_THRESHOLD, color="r", linestyle="--")
    ax.set_title("Live Crowd Count Graph")
    ax.set_xlabel("Frame")
    ax.set_ylabel("People Count")
    plt.pause(0.001)

    # Crowd anomaly
    crowd_anomaly = max_people > CROWD_THRESHOLD
    if crowd_anomaly and (current_time - last_crowd_alarm > ALARM_COOLDOWN):
        winsound.PlaySound(CROWD_ALARM, winsound.SND_FILENAME | winsound.SND_ASYNC)
        cv2.imwrite(f"{SCREENSHOT_DIR}/crowd_{int(current_time)}.jpg", frame)
        last_crowd_alarm = current_time

    # ======================================================
    # 🖥️ STATUS DISPLAY
    # ======================================================
    if weapon_found:
        status = "ABNORMAL OBJECT DETECTED"
        color = (0, 0, 255)
    elif crowd_anomaly:
        status = "CROWD ANOMALY DETECTED"
        color = (0, 0, 255)
    else:
        status = "STATUS: NORMAL"
        color = (0, 255, 0)

    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(frame, f"People Count: {person_count}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 0), 2)

    cv2.imshow("Final Live Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()

print("🛑 System stopped safely")
"""



import cv2
import time
import os
import winsound
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
WEAPON_MODEL_PATH = "models/weapon_last.pt"
PERSON_MODEL_PATH = "models/yolov8n.pt"

WEAPON_ALARM = "alarms/weapon_alarm.wav"
CROWD_ALARM = "alarms/crowd_alarm.wav"

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
weapon_model = YOLO(WEAPON_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

# ---------------- SETTINGS ----------------
WEAPON_CLASSES = ["knife", "gun", "pistol", "grenade", "weapon"]
CONF_WEAPON = 0.5
CONF_PERSON = 0.4

CROWD_THRESHOLD = 5
FRAME_WINDOW = 10
ALARM_COOLDOWN = 5  # seconds

# ---------------- STORAGE ----------------
people_history = deque(maxlen=FRAME_WINDOW)
frame_count = 0
graph_x, graph_y = [], []

last_weapon_alarm = 0
last_crowd_alarm = 0

# ---------------- GRAPH SETUP ----------------
plt.ion()
fig, ax = plt.subplots()

# ---------------- MOTION (NORMAL OBJECT) ----------------
prev_gray = None
MOTION_THRESHOLD = 2000  # pixels

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print(" Final Security System Started")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1

    # ======================================================
    # 🔴 WEAPON DETECTION
    # ======================================================
    weapon_found = False
    normal_object_detected = False

    weapon_results = weapon_model(frame, conf=CONF_WEAPON, verbose=False)

    for box in weapon_results[0].boxes or []:
        cls = int(box.cls[0])
        label = weapon_model.names[cls].lower().strip()

        # Ignore humans if present in weapon model
        if label == "person":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in WEAPON_CLASSES:
            weapon_found = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label.upper(),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

            if current_time - last_weapon_alarm > ALARM_COOLDOWN:
                winsound.PlaySound(
                    WEAPON_ALARM,
                    winsound.SND_FILENAME | winsound.SND_ASYNC
                )
                cv2.imwrite(
                    f"{SCREENSHOT_DIR}/weapon_{int(current_time)}.jpg",
                    frame
                )
                last_weapon_alarm = current_time

    # ======================================================
    # 🟢 NORMAL OBJECT (MOTION BASED)
    # ======================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if not weapon_found:
        if prev_gray is not None:
            frame_diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            motion_pixels = cv2.countNonZero(thresh)

            if motion_pixels > MOTION_THRESHOLD:
                normal_object_detected = True

        prev_gray = gray

    # ======================================================
    # 👥 CROWD COUNT & ANOMALY
    # ======================================================
    person_count = 0
    person_results = person_model(frame, conf=CONF_PERSON, verbose=False)

    for r in person_results:
        for box in r.boxes:
            if person_model.names[int(box.cls[0])] == "person":
                person_count += 1

    people_history.append(person_count)
    max_people = max(people_history)

    # ---------------- GRAPH UPDATE ----------------
    graph_x.append(frame_count)
    graph_y.append(person_count)

    ax.clear()
    ax.plot(graph_x, graph_y)
    ax.axhline(y=CROWD_THRESHOLD, color='r', linestyle='--')
    ax.set_title("Live Crowd Count Graph")
    ax.set_xlabel("Frame")
    ax.set_ylabel("People Count")
    plt.pause(0.001)

    # Crowd anomaly alarm
    crowd_anomaly = max_people > CROWD_THRESHOLD
    if crowd_anomaly and (current_time - last_crowd_alarm > ALARM_COOLDOWN):
        winsound.PlaySound(
            CROWD_ALARM,
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )
        cv2.imwrite(
            f"{SCREENSHOT_DIR}/crowd_{int(current_time)}.jpg",
            frame
        )
        last_crowd_alarm = current_time

    # ======================================================
    # 🖥️ STATUS DISPLAY
    # ======================================================
    if weapon_found:
        status_text = "ABNORMAL OBJECT DETECTED"
        color = (0, 0, 255)

    elif crowd_anomaly:
        status_text = "CROWD ANOMALY DETECTED"
        color = (0, 0, 255)

    elif normal_object_detected:
        status_text = "NORMAL OBJECT"
        color = (0, 255, 0)

        h, w, _ = frame.shape
        cv2.rectangle(frame, (100, 100), (w - 100, h - 100), color, 3)

    else:
        status_text = "STATUS: NORMAL"
        color = (0, 255, 0)

    cv2.putText(frame, status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    cv2.putText(frame, f"People Count: {person_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 0), 2)

    cv2.imshow("Final Live Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()

print("🛑 System stopped safely")
