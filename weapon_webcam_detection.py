import cv2
from ultralytics import YOLO

model = YOLO("models/weapon_last.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Weapon Detection - Abnormal Object", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
