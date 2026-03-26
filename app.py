import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import base64
import os 
import numpy as np
def get_audio_html(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    return f"""
        <audio id="alarm-audio" loop>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        <script>
            var audio = document.getElementById('alarm-audio');
            window.parent.document.addEventListener('phone_detected', function(e) {{
                if (e.detail) {{ audio.play(); }}
                else {{ audio.pause(); audio.currentTime = 0; }}
            }});
        </script>
    """

st.title("📱 Live Phone Detector with Sound")

# Model Load
model = YOLO("yolov8n.pt")

# HTML Audio Inject karna
if os.path.exists("alarm.wav"):
    st.components.v1.html(get_audio_html("alarm.wav"), height=0)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.phone_in_frame = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.4)
        
        current_phone_detected = False
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label == "cell phone":
                    current_phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(img, "PHONE DETECTED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # JavaScript ko signal bhejna sound ke liye
        if current_phone_detected != self.phone_in_frame:
            self.phone_in_frame = current_phone_detected
            val = "true" if current_phone_detected else "false"
            st.components.v1.html(f"""
                <script>
                    var event = new CustomEvent('phone_detected', {{ detail: {val} }});
                    window.parent.document.dispatchEvent(event);
                </script>
            """, height=0)

        return img

webrtc_streamer(key="phone-check", video_transformer_factory=VideoTransformer)