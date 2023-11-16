import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Define RTC configuration for ICE servers (STUN/TURN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.title("WebRTC Test")

    # Create a webrtc streamer
    webrtc_streamer(key="example", rtc_configuration=RTC_CONFIGURATION)

if __name__ == "__main__":
    main()
