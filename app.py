import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from ultralytics import YOLO

# YOLO Model Load
model = YOLO("yolov8n.pt")

class PhoneDetector(VideoTransformerBase):
    def transform(self, frame):
        # Frame ko array mein badalna
        img = frame.to_ndarray(format="bgr24")
        
        # YOLO Detection 
        results = model(img, conf=0.5, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                # Agar "cell phone" detect hua (COCO dataset mein index 67 hota hai)
                if model.names[cls] == "cell phone":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Red color ka rectangle draw 
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # Label likhna
                    cv2.putText(img, "PHONE DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return img

# Streamlit UI
st.set_page_config(page_title="AI Phone Detector", layout="wide")
st.title("📱 Real-Time Phone Detection System")
st.write("Click **'Start'** below and allow camera access to begin detection.")

# WebRTC Streamer (Browser camera ke liye)
webrtc_streamer(
    key="phone-detection",
    video_transformer_factory=PhoneDetector,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.info("Note: Sound is disabled in Cloud Mode for stability.")