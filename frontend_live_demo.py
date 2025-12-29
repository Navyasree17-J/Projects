import cv2
import os
import time
import winsound
from ultralytics import YOLO

# Paths
MODEL_PATH = "models/weapon_last.pt"   # your trained YOLO model
ALARM_PATH = "alarms/alarm.wav"
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Webcam
cap = cv2.VideoCapture(0)
last_alarm_time = 0
ALARM_COOLDOWN = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    detections = results[0].boxes
    abnormal_detected = False

    if detections is not None and len(detections) > 0:
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            # ✅ Skip humans
            if label.lower() in ['person', 'human']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🔴 Red for abnormal (weapons), 🟢 Green for normal
            if label.lower() in ['knife', 'gun', 'pistol']:  # adjust according to your model
                color = (0, 0, 255)
                abnormal_detected = True
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 🔊 Alarm + Screenshot if abnormal detected
        if abnormal_detected:
            current_time = time.time()
            if current_time - last_alarm_time > ALARM_COOLDOWN:
                winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
                last_alarm_time = current_time
                filename = f"{SCREENSHOT_DIR}/abnormal_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)

    cv2.imshow("Object Detection (No Humans)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
