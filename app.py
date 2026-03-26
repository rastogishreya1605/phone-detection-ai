import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Better model (more accurate)
model = YOLO("yolov8s.pt")

st.title("📱 Phone Detection AI - Final Clean App")

# ------------------ DETECTION FUNCTION ------------------
def detect(img):
    results = model(img)
    phone_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # ✅ Correct detection
            if label == "cell phone" and conf > 0.4:
                phone_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_text = f"{label} ({conf:.2f})"

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, label_text, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return img, phone_count


# ------------------ MODE ------------------
option = st.radio("Choose Mode", ["Upload Image", "Capture Photo", "Live Webcam"])

# ------------------ UPLOAD ------------------
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        img, count = detect(img)

        st.image(img, channels="BGR")
        st.subheader(f"📱 Phones detected: {count}")

        if count > 0:
            st.success("Phone detected ✅")
            st.audio("alarm.wav", autoplay=True)
        else:
            st.warning("No phone detected ❌")


# ------------------ CAMERA ------------------
elif option == "Capture Photo":
    cam = st.camera_input("Take Photo")

    if cam:
        img = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), 1)

        img, count = detect(img)

        st.image(img, channels="BGR")
        st.subheader(f"📱 Phones detected: {count}")

        if count > 0:
            st.success("Phone detected ✅")
            st.audio("alarm.wav", autoplay=True)
        else:
            st.warning("No phone detected ❌")


# ------------------ LIVE ------------------
elif option == "Live Webcam":

    st.write("🎥 Live detection running... Click capture")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_frame = None

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img, _ = detect(img)

            self.last_frame = img
            return img

    ctx = webrtc_streamer(
        key="webcam",
        video_transformer_factory=VideoTransformer
    )

    if ctx.video_transformer:
        if st.button("📸 Capture Result"):
            frame = ctx.video_transformer.last_frame

            if frame is not None:
                img, count = detect(frame)

                st.image(img, channels="BGR")
                st.subheader(f"📱 Phones detected: {count}")

                if count > 0:
                    st.success("Phone detected ✅")
                    st.audio("alarm.wav", autoplay=True)
                else:
                    st.warning("No phone detected ❌")