import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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

# YOLO Model Load karo
model = YOLO("yolov8n.pt") 

# --- VIDEO PROCESSOR (Aapka logic yahan hai) ---
class MyVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alarm_playing = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # YOLO Detection (Aapka original logic)
        results = model(img, conf=0.5, verbose=False)
        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Agar 'cell phone' detect hua
                if label == "cell phone":
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw Red Box (Aapka style)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, "PHONE DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # --- ALERT/SOUND LOGIC (Aapka logic) ---
        if phone_detected:
            if pygame_available and not self.alarm_playing:
                try:
                    pygame.mixer.music.load("alarm.wav")
                    pygame.mixer.music.play(-1) # Loop play
                    self.alarm_playing = True
                except:
                    pass
        else:
            if pygame_available and self.alarm_playing:
                pygame.mixer.music.stop()
                self.alarm_playing = False

        return frame.from_ndarray(img, format="bgr24")

# --- DEPLOYMENT ---
webrtc_streamer(
    key="phone-detection",
    video_processor_factory=MyVideoProcessor, # Sirf ye change kiya hai deploy ke liye
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.write("Click **Start** to begin tracking.")