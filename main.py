import streamlit as st
from PIL import Image
import yolo_opencv  # Make sure this is correctly pointing to your modified script


st.write('Image Upload')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Call the object detection function from yolo_opencv
    detected_image = yolo_opencv.detect_objects("temp.jpg")
    
    # Display the detected image
    st.image(detected_image, caption='Processed Image', use_column_width=True)
