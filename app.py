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

# YOLO Model Load (Caching use kar rahe hain taaki RAM kam kharch ho)
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# --- VIDEO PROCESSOR CLASS (Aapka logic yahan hai) ---
class PhoneDetector(VideoProcessorBase):
    def __init__(self):
        self.alarm_playing = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO Detection (Aapka Logic)
        results = model(img, conf=0.5, verbose=False)
        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                if label == "cell phone":
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, "PHONE DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Alert/Sound Logic (Render/Cloud Compatibility)
        if phone_detected:
            if pygame_available and not self.alarm_playing:
                try:
                    pygame.mixer.music.load("alarm.wav")
                    pygame.mixer.music.play(-1)
                    self.alarm_playing = True
                except: pass
        else:
            if pygame_available and self.alarm_playing:
                pygame.mixer.music.stop()
                self.alarm_playing = False

        return frame.from_ndarray(img, format="bgr24")

# --- DISPLAY ---
webrtc_streamer(
    key="phone-detection",
    video_processor_factory=PhoneDetector, # Yahan use hua hai processor
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.info("Note: Click 'Start' to begin. Phone detection will trigger visual and sound alerts.")