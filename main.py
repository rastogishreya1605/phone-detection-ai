import cv2
from phone_detector import detect_phone
from alarm import play_alarm

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

alarm_on = False   # alarm state

while True:

    ret, frame = cap.read()

    if not ret:
        break

    phone_detected = detect_phone(frame)

    if phone_detected:

        cv2.putText(frame,
                    "PHONE DETECTED",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

        # Play alarm only once
        if not alarm_on:
            play_alarm()
            alarm_on = True

    else:

        cv2.putText(frame,
                    "NO PHONE",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        # Reset alarm state
        alarm_on = False

    cv2.imshow("AI Study Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

