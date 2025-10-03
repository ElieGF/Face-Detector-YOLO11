import cv2
import ultralytics

# Load the trained model
model = ultralytics.YOLO("runs/detect/train/weights/best.pt")

# Get live video feed from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Run inference
    results = model.predict(
        frame,
        conf=0.5,
        imgsz=640,
        half=True,     # FP16 (Float16) to speed up inference
        device=0
        )


    # Get annotated frame and display
    annot = results[0].plot()
    cv2.imshow("Live Webcam", annot)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Release webcam and close visualization window
cap.release()
cv2.destroyAllWindows()
