import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import deque

# Load model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

# Store last N frame counts
MAX_POINTS = 50
people_counts = deque(maxlen=MAX_POINTS)

plt.ion()  # interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_ylim(0, 15)
ax.set_title("Live Crowd Count Graph")
ax.set_xlabel("Frames")
ax.set_ylabel("People Count")

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

    people_counts.append(count)

    # Update graph
    line.set_xdata(range(len(people_counts)))
    line.set_ydata(people_counts)
    ax.set_xlim(0, MAX_POINTS)
    plt.draw()
    plt.pause(0.01)

    cv2.putText(frame, f"People Count: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.imshow("Crowd Graph Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
