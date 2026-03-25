import winsound
import threading

def play_alarm():
    def sound():
        winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)

    threading.Thread(target=sound).start()