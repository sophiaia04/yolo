import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
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

# Define RTC configuration for ICE servers (STUN/TURN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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


# Paths to pre-trained models
face_txt_path = "opencv_face_detector.pbtxt"
face_model_path = "opencv_face_detector_uint8.pb"
age_txt_path = "age_deploy.prototxt"
age_model_path = "age_net.caffemodel"
gender_txt_path = "gender_deploy.prototxt"
gender_model_path = "gender_net.caffemodel"

# Constants and class labels for age and gender
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_classes = ['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
            'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60']
gender_classes = ['Gender: Male', 'Gender: Female']

# Load pre-trained models
face_net = cv2.dnn.readNetFromTensorflow(face_model_path, face_txt_path)
age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)


# Define the VideoTransformer class
class VideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face detection
        processed_frame, b_boxes = get_face_box(face_net, img)

        for bbox in b_boxes:
            # Extracting face ROI
            face = processed_frame[max(0, bbox[1]):min(bbox[3], processed_frame.shape[0]-1), 
                                   max(0, bbox[0]):min(bbox[2], processed_frame.shape[1]-1)]

            # Prepare the face ROI for age and gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Age prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_classes[age_preds[0].argmax()]

            # Gender prediction
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_classes[gender_preds[0].argmax()]

            # Annotate the processed frame with age and gender information
            label = f"{gender}, {age}"
            cv2.putText(processed_frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Convert the color from BGR to RGB
        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        return av.VideoFrame.from_ndarray(img_rgb, format="bgr24")



# Button to start Age and Gender Estimation
if st.button("Age and Gender Estimation", use_container_width=True):
    # Use webrtc_streamer to process video stream
    webrtc_streamer(key="age_gender_estimation", 
                    video_processor_factory=VideoTransformer, 
                    rtc_configuration=RTC_CONFIGURATION)


# More Info section with buttons
st.write("More Info")
col1, col2 = st.columns(2)
with col1:
    if st.button("Check out our website!", use_container_width=True):
        webbrowser.open_new_tab("https://lac2.org")
with col2:
    if st.button("Book an appointment with our AI Hub Manager!", use_container_width=True):
        webbrowser.open_new_tab("https://www.typecalendar.com/wp-content/uploads/2022/12/December-2023-Calendar.jpg")
