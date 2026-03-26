import streamlit as st
import base64

def play_alarm():
    with open("alarm.wav", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

    md = f"""
    <audio autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)