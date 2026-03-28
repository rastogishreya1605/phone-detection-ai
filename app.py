import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- PYGAME CLOUD BYPASS ---
try:
    import pygame
    pygame.mixer.init()
    pygame_available = True
    print("Sound System Ready!")
except Exception as e:
    pygame_available = False
    print("Sound disabled (Cloud Mode)")

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Phone Detector", layout="wide")
st.title("📱 Phone Distraction Detection")

# 1. Video aur Alerts ke liye Placeholders
alert_placeholder = st.empty()  # Ye warning box ko control karega
frame_placeholder = st.empty()  # Ye video feed ko control karega
stop_button = st.button("Stop Camera")

# YOLO Model Load karo
model = YOLO("yolov8n.pt") 

# Camera Start
cap = cv2.VideoCapture(0)

# Status Variables
alarm_playing = False

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera nahi mil raha!")
        break

    # YOLO Detection
    results = model(frame, conf=0.5, verbose=False)
    phone_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Agar 'cell phone' detect hua
            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw Red Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # --- ALERT LOGIC 
    if phone_detected:
        # Warning Box sirf ek baar dikhayega
        alert_placeholder.warning("⚠️ PHONE DETECTED! FOCUS ON YOUR WORK!")
        
        if pygame_available and not alarm_playing:
            try:
                pygame.mixer.music.load("alarm.wav")
                pygame.mixer.music.play(-1) # Loop play
                alarm_playing = True
            except:
                pass
    else:
        #JAise hi phone hatega, warning box khali (empty) ho jayega
        alert_placeholder.empty()
        
        if pygame_available and alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    # --- DISPLAY ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
alert_placeholder.empty() 
st.success("Camera Closed Successfully.")