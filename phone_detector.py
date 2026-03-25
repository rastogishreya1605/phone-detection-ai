from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_phone(frame):

    results = model(frame, conf=0.35)  # confidence threshold

    phone_detected = False 

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":

                phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

                cv2.putText(frame,
                            "PHONE DETECTED",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

    return phone_detected