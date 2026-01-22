from flask import Flask, render_template, Response
import cv2
import time
import os
import winsound
from ultralytics import YOLO
from collections import deque

# ---- Matplotlib for graph (Flask safe) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------------- PATHS ----------------
WEAPON_MODEL_PATH = "models/weapon_last.pt"
PERSON_MODEL_PATH = "models/yolov8n.pt"

WEAPON_ALARM = "alarms/weapon_alarm.wav"
CROWD_ALARM = "alarms/crowd_alarm.wav"

SCREENSHOT_DIR = "screenshots"
GRAPH_PATH = "static/crowd_graph.png"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------- LOAD MODELS ----------------
weapon_model = YOLO(WEAPON_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

# ---------------- SETTINGS ----------------
WEAPON_CLASSES = ["knife", "gun", "pistol", "grenade", "weapon"]
CONF_WEAPON = 0.5
CONF_PERSON = 0.4

CROWD_THRESHOLD = 5
FRAME_WINDOW = 15
ALARM_COOLDOWN = 5

# ---------------- STORAGE ----------------
people_history = deque(maxlen=FRAME_WINDOW)
graph_x, graph_y = [], []
frame_count = 0

last_weapon_alarm = 0
last_crowd_alarm = 0

# ---------------- GRAPH FUNCTION ----------------
def save_crowd_graph(x, y):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color='blue', linewidth=2)
    plt.axhline(y=CROWD_THRESHOLD, color='red', linestyle='--')
    plt.xlabel("Frame")
    plt.ylabel("People Count")
    plt.title("Crowd Count Over Time")
    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce lag

def generate_frames():
    global frame_count, last_weapon_alarm, last_crowd_alarm

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_time = time.time()

        weapon_found = False

        # ---------------- WEAPON DETECTION ----------------
        weapon_results = weapon_model(frame, conf=CONF_WEAPON, verbose=False)

        for box in weapon_results[0].boxes or []:
            cls = int(box.cls[0])
            label = weapon_model.names[cls].lower()

            if label in WEAPON_CLASSES:
                weapon_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label.upper(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if current_time - last_weapon_alarm > ALARM_COOLDOWN:
                    winsound.PlaySound(WEAPON_ALARM, winsound.SND_ASYNC)
                    cv2.imwrite(f"{SCREENSHOT_DIR}/weapon_{int(current_time)}.jpg", frame)
                    last_weapon_alarm = current_time

        # ---------------- PERSON COUNT ----------------
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

        if frame_count % 15 == 0:
            save_crowd_graph(graph_x, graph_y)

        # ---------------- CROWD ANOMALY ----------------
        crowd_anomaly = max_people > CROWD_THRESHOLD

        if crowd_anomaly and current_time - last_crowd_alarm > ALARM_COOLDOWN:
            winsound.PlaySound(CROWD_ALARM, winsound.SND_ASYNC)
            cv2.imwrite(f"{SCREENSHOT_DIR}/crowd_{int(current_time)}.jpg", frame)
            last_crowd_alarm = current_time

        # ---------------- STATUS TEXT ----------------
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

        cv2.putText(frame, f"People Count: {person_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
