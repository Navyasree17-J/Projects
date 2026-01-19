import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("models/weapon_best.pt")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    weapon_count = 0
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label.lower() in ["gun", "grenade", "knife"]:
                weapon_count += 1
            if label.lower() == "person":
                person_count += 1

    # UI Overlay
    cv2.rectangle(frame, (0, 0), (420, 130), (0, 0, 0), -1)

    cv2.putText(frame, f"People Count: {person_count}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Weapon Detected: {weapon_count}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if weapon_count > 0:
        cv2.putText(frame, "ALERT: ABNORMAL OBJECT DETECTED!",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Crowd Anomaly Monitoring Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
