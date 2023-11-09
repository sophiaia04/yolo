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
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    # Check if the webcam is working
    if not camera.isOpened():
        st.error("Error: Webcam not found or not accessible. Please make sure the webcam is connected and permissions are granted.")
        run = False  # Stop if the webcam is not accessible
    else:
        st.write('Webcam worked')

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
    age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    while run:
        print('run')
        _, frame = camera.read()
        frame, b_boxes = get_face_box(face_net, frame)
        if not b_boxes:
            st.write("No face Detected, Checking the next frame")
        else:
            st.write(f"Detected {len(b_boxes)} face(s)")

        for bbox in b_boxes:
            face = frame[max(0, bbox[1]): min(bbox[3], frame.shape[0] - 1),
                         max(0, bbox[0]): min(bbox[2], frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_pred_list = gender_net.forward()
            gender = gender_classes[gender_pred_list[0].argmax()]
            st.write(f"Gender: {gender}, confidence = {gender_pred_list[0].max() * 100}%")

            age_net.setInput(blob)
            age_pred_list = age_net.forward()
            age = age_classes[age_pred_list[0].argmax()]
            st.write(f"Age: {age}, confidence = {age_pred_list[0].max() * 100}%")

            label = f"{gender}, {age}"
            cv2.putText(
                frame,
                label,
                (bbox[0],
                 bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,
                 255,
                 255),
                2,
                cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        break
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
