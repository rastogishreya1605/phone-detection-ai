import cv2
from ultralytics import YOLO
from pygame import mixer
import time
import os

# --- 1. Sound Setup ---
mixer.init()
SOUND_FILE = "alarm.wav" # Check kijiye ki aapki file ka naam yahi hai

if os.path.exists(SOUND_FILE):
    try:
        alert_sound = mixer.Sound(SOUND_FILE)
        print(f"✅ SUCCESS: {SOUND_FILE} load ho gayi hai!")
    except Exception as e:
        print(f"❌ ERROR: Sound file load nahi hui: {e}")
        alert_sound = None
else:
    print(f"❌ ERROR: '{SOUND_FILE}' folder mein nahi mili!")
    alert_sound = None

# --- 2. Model Setup ---
model = YOLO("yolov8n.pt") 

# --- 3. Camera Setup ---
cap = cv2.VideoCapture(0)
last_alert_time = 0 
alert_cooldown = 2 # 2 second ka gap sound ke beech mein

print("\n🚀 System Start! 'q' dabayein band karne ke liye.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Real-time Phone Detection
    results = model(frame, conf=0.4, stream=True)
    phone_detected = False

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            
            if label == "cell phone":
                phone_detected = True
                # Bounding Box Draw karna
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- 4. Automatic Sound Logic ---
    if phone_detected:
        current_time = time.time()
        if alert_sound and (current_time - last_alert_time > alert_cooldown):
            alert_sound.play()
            last_alert_time = current_time
            print("🚨 Phone Dikha! Sound baj raha hai...")

    # Output dikhayyein
    cv2.imshow("Phone Detection Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mixer.quit()