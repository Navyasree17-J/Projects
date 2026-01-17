import cv2
import time
import os
from ultralytics import YOLO
from collections import deque

# Load model
model = YOLO("yolov8n.pt")

# Settings
CROWD_THRESHOLD = 5
FRAME_WINDOW = 10
COOLDOWN = 5

people_history = deque(maxlen=FRAME_WINDOW)
last_save_time = 0

os.makedirs("screenshots", exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                count += 1

    people_history.append(count)
    max_people = max(people_history)

    if max_people > CROWD_THRESHOLD:
        cv2.putText(frame, "CROWD ANOMALY!",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

        now = time.time()
        if now - last_save_time > COOLDOWN:
            filename = f"screenshots/crowd_{int(now)}.jpg"
            cv2.imwrite(filename, frame)
            last_save_time = now
    else:
        cv2.putText(frame, "CROWD NORMAL",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    cv2.imshow("Crowd Anomaly Screenshot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
