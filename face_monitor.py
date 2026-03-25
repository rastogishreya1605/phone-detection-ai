import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

def check_attention(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        landmarks = result.multi_face_landmarks[0]

        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]

        if abs(left_eye.y - right_eye.y) < 0.01:
            return "sleeping"

        nose = landmarks.landmark[1]

        if nose.x < 0.3 or nose.x > 0.7:
            return "not looking"

        return "focused"

    return "no face"