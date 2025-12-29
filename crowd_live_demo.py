import cv2
from ultralytics import YOLO

# ----- Load pretrained YOLO model -----
# Make sure YOLOv8 or YOLOv5 is installed: pip install ultralytics
model = YOLO("yolov8n.pt")  # small model for speed; detects 'person' class

# ----- Open webcam -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----- Run person detection -----
    results = model(frame)

    person_count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label.lower() == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                person_count += 1

    # ----- Display total people count -----
    cv2.putText(frame, f"People Count: {person_count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # ----- Show frame -----
    cv2.imshow("Crowd Counting Demo", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
