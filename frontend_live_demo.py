import cv2

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 🟢 GREEN BOX for NORMAL OBJECT
    cv2.rectangle(
        frame,
        (50, 50),
        (w - 50, h - 50),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Normal Object",
        (60, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    cv2.imshow("Normal Object Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
