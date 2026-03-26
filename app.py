import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load better model
model = YOLO("yolov8s.pt")

st.title("📱 Phone Detection AI - Final App")

# ------------------ DETECTION FUNCTION ------------------
def detect(img):
    results = model(img)
    phone_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

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


# ------------------ UPLOAD IMAGE ------------------
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


# ------------------ CAMERA CAPTURE ------------------
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


# ------------------ LIVE WEBCAM ------------------
elif option == "Live Webcam":

    st.write("🎥 Live detection running...")

    if "phone_detected" not in st.session_state:
        st.session_state.phone_detected = False

    if "alert_played" not in st.session_state:
        st.session_state.alert_played = False

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)

            phone_found = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])

                    if label == "cell phone" and conf > 0.4:
                        phone_found = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label_text = f"{label} ({conf:.2f})"

                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(img, label_text, (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            st.session_state.phone_detected = phone_found
            return img

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=VideoTransformer
    )

    # ✅ CLEAN RESULT DISPLAY
    if st.session_state.phone_detected:
        st.success("📱 Phone detected!")

        # 🔊 play only once
        if not st.session_state.alert_played:
            st.audio("alarm.wav", autoplay=True)
            st.session_state.alert_played = True
    else:
        st.session_state.alert_played = False