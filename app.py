import cv2
from ultralytics import YOLO
from pygame import mixer
import os

# --- 1. Sound Setup ---
mixer.init()
SOUND_FILE = "alarm.wav"
if os.path.exists(SOUND_FILE):
    mixer.music.load(SOUND_FILE)
    print("✅ Sound Loaded!")

# --- 2. Model Load ---
model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)
is_playing = False

print("\n🚀 Scanner Start! Phone dikhao...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Detection (Confidence 0.4 rakha hai)
    results = model(frame, conf=0.4, stream=True)
    phone_found = False

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            
            if label == "cell phone":
                phone_found = True
                # Bounding Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 🟢 1. Green Rectangle draw karna
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
                # 📝 2. "PHONE DETECTED" Text likhna (Screen par)
                # Font scale 1.2 aur Thickness 3 rakha hai taaki bada dikhe
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # --- 3. Sound Control ---
    if phone_found:
        if not is_playing:
            mixer.music.play(-1)
            is_playing = True
            print("🚨 Phone Detected! Sound ON")
    else:
        if is_playing:
            mixer.music.stop()
            is_playing = False
            print("✅ Phone Removed! Sound OFF")

    # Display Window
    cv2.imshow("Real-time AI Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mixer.quit()