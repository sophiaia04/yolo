import streamlit as st
import webbrowser
import cv2
import numpy as np
from PIL import Image

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
# MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to get the face bounding boxes using OpenCV DNN
def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect

# Streamlit UI elements and configuration
st.image("logo.png")  # Display a logo image
st.title("Play with AI Models")
st.write("Play with some AI models that leverage GPU computation, all running on the below server!")

# Button to start Age and Gender Estimation
if st.button("Age and Gender Estimation", use_container_width=True):
    st.title("Webcam Live Feed")
    run = True


from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
import cv2
import numpy as np

# Define RTC configuration to ensure compatibility with different network conditions
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}  # STUN server
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Apply face detection and age, gender estimation
        # Insert the user's existing logic here
        # ...

        # Then return the result frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI element to start the webcam and display the processed video
if st.button("Age and Gender Estimation"):
    # Use webrtc_streamer to process video stream with the VideoTransformer
    webrtc_streamer(key="example", video_processor_factory=VideoTransformer, rtc_configuration=RTC_CONFIGURATION)

else:
    st.write('Stopped')

# Button to stop the webcam feed
if st.button("Stop", use_container_width=True):
    run = False


# More Info section with buttons
st.write("More Info")
col1, col2 = st.columns(2)
with col1:
    if st.button("Check out our website!", use_container_width=True):
        webbrowser.open_new_tab("https://lac2.org")
with col2:
    if st.button("Book an appointment with our AI Hub Manager!", use_container_width=True):
        webbrowser.open_new_tab("https://www.typecalendar.com/wp-content/uploads/2022/12/December-2023-Calendar.jpg")
