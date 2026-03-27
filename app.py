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
    print("✅ Sound System Ready!")
except Exception as e:
    pygame_available = False
    print("⚠️ Sound disabled (Cloud Mode)")

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Phone Detector", layout="wide")
st.title("📱 Real-time Phone Distraction Detection")
st.write("Ye AI camera se phone detect karta hai aur alarm bajata hai.")

# YOLO Model Load karo
model = YOLO("yolov8n.pt") 

# Streamlit placeholder for video
frame_placeholder = st.empty()
stop_button = st.button("Stop Camera")

# Camera Start
cap = cv2.VideoCapture(0)

# Alarm Status
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

    # Alarm Logic
    if phone_detected:
        if pygame_available and not alarm_playing:
            try:
                pygame.mixer.music.load("alarm.wav")
                pygame.mixer.music.play(-1) # Loop play
                alarm_playing = True
            except:
                pass
        # Web Alert
        st.warning("⚠️ PHONE DETECTED! FOCUS ON YOUR WORK!")
    else:
        if pygame_available and alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    # --- CRITICAL FIX: STREAMLIT DISPLAY ---
    # OpenCV BGR format use karta hai, Streamlit RGB mangta hai
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Thoda gap taaki CPU blast na ho
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
st.success("Camera Closed Successfully.")