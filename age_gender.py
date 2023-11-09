import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
import cv2
import numpy as np

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
# MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define RTC configuration to ensure compatibility with different network conditions
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Function to get the face bounding boxes using OpenCV DNN
def get_face_box(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame, b_boxes_detect

# Paths to pre-trained models
face_txt_path = "opencv_face_detector.pbtxt"
face_model_path = "opencv_face_detector_uint8.pb"
age_txt_path = "age_deploy.prototxt"
age_model_path = "age_net.caffemodel"
gender_txt_path = "gender_deploy.prototxt"
gender_model_path = "gender_net.caffemodel"

# Load pre-trained models
age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

# Constants and class labels for age and gender
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classes = ['Male', 'Female']

# Streamlit UI elements and configuration
st.image("logo.png")  # Display a logo image
st.title("Play with AI Models")
st.write("Play with some AI models that leverage GPU computation, all running on the below server!")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Apply face detection
        img, b_boxes = get_face_box(face_net, img)
        
        # Apply age and gender estimation
        for bbox in b_boxes:
            face = img[max(0, bbox[1]):min(bbox[3], img.shape[0]-1), 
                       max(0, bbox[0]):min(bbox[2], img.shape[1]-1)]
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_classes[gender_preds[0].argmax()]
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_classes[age_preds[0].argmax()]
            
            label = f"{gender}, {age}"
            cv2.putText(img, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Using webrtc_streamer to process video stream
webrtc_streamer
