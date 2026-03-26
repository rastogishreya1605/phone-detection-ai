import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model
model = YOLO("yolov8n.pt")

st.title("📱 Phone Detection AI - Pro App")

# ------------------ DETECTION FUNCTION ------------------
def detect(img):
    results = model(img)
    phone_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            conf = float(box.conf[0])
            label_text = f"{label} ({conf:.2f})"

            if label == "cell phone":
                phone_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, label_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return img, phone_count


# ------------------ MODE SELECT ------------------
option = st.radio("Choose Mode", ["Upload Image", "Capture Photo", "Live Webcam"])

# ------------------ UPLOAD ------------------
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)

        img, count = detect(img)
        st.image(img, channels="BGR")
        st.subheader(f"📱 Phones detected: {count}")

# ------------------ CAMERA CAPTURE ------------------
elif option == "Capture Photo":
    cam = st.camera_input("Take Photo")

    if cam:
        bytes_data = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)

        img, count = detect(img)
        st.image(img, channels="BGR")
        st.subheader(f"📱 Phones detected: {count}")

# ------------------ LIVE WEBCAM ------------------
elif option == "Live Webcam":

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img, _ = detect(img)
            return img

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)