import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

st.title("📱 Phone Detection AI")

# Mode selection
option = st.radio("Choose Input Type", ["Upload Image", "Use Camera"])

# ------------------ IMAGE UPLOAD ------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

# ------------------ CAMERA INPUT ------------------
elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    else:
        img = None

# ------------------ DETECTION ------------------
if 'img' in locals() and img is not None:
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

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(img, channels="BGR")

    st.subheader(f"📱 Phones detected: {phone_count}")

    if phone_count == 0:
        st.warning("No phone detected")
    else:
        st.success("Phone detected")