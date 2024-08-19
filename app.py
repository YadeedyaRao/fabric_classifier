import streamlit as st

from PIL import Image
import torch
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model

@st.cache_resource
def load_yolo_model(file_path):
    return YOLO(file_path)
@st.cache_resource
def load_cloth_model(file_path):
    return load_model(file_path)

#load models
cloth_model = load_cloth_model('fabric_7_25_model.keras')
yolo_model = load_yolo_model('best.pt')

# some constant vars
slct_cloth = "Picture of piece of cloth (More accurate)"
slct_pers = "Picture of a person wearing cloth (Comparitively less accurate)"
class_names = ['Artificial_fur', 'Artificial_leather', 'Chenille', 'Corduroy','Cotton','Silk', 'Woolen']


def clear_uploaded_file():
    if 'uploaded_file' in st.session_state:
        del st.session_state['uploaded_file']

def draw_label_with_background(image, text, position, font, font_scale, color, thickness):
    # Calculate the text size and position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position # position of top-left of rectangle
    x = x - thickness # padding of thickness
    y = y-thickness # padding of thickness
    
    text_color = tuple([255 - x for x in color])  # text color giving -> (contrast)
    padding = 5  # Padding around text
    cv2.rectangle(image, (x, y - text_height - padding), 
                      (x + text_width + padding, y+thickness), 
                      color, 
                      thickness=cv2.FILLED)

    # Draw the text on top of the rectangle
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def make_prediction(img):
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    score2 = tf.nn.softmax(cloth_model.predict(img[None,:,:]))
    return [class_names[np.argmax(score2)], np.max(score2)*100]

st.title('Cloth Fabric Classifier') 

if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

option = st.selectbox("Select type of picture you are uploading", [slct_cloth, slct_pers])

# Check if the selected option has changed
if option != st.session_state['selected_option']:
    # Update the session state
    st.session_state['selected_option'] = option

    # Clear the uploaded file
    if 'uploaded_file' in st.session_state:
        del st.session_state['uploaded_file']

# File uploader logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
    if option == slct_cloth:
        pil_image = Image.open(uploaded_file).convert('RGB')
        
        # Convert PIL Image to NumPy array
        image_np = np.array(pil_image)
        cls_name, conf = make_prediction(image_np)

        #display the prediction
        st.write(f"Predicted as {cls_name} with {conf:.2f}% confidence")
        st.image(image_np, "uploaded image")

    if option == slct_pers:
        pil_image = Image.open(uploaded_file).convert('RGB')
        
        # Convert PIL Image to NumPy array
        image_np = np.array(pil_image)
        # resizing into shape divisible by 32 as YOLO model's stride requirements
        resize_image_np = cv2.resize(image_np, (640, 640))
        
        # Convert resized image to tensor
        image_tensor = torch.from_numpy(resize_image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Perform inference
        results = yolo_model(image_tensor)
        class_names = ['Artificial_fur', 'Artificial_leather', 'Chenille', 'Corduroy','Cotton','Silk', 'Woolen']
        # Display results
        st.image(image_np, caption='Uploaded Image', use_column_width=True)
        pred_image = results[0].plot(conf = False)
        cat = [int(x) for x in results[0].boxes.cls]
        names = results[0].names
        display_image = pred_image
        col1, col2 = st.columns([3, 1])  # Adjust column ratio as needed
        with col1:
            image_placeholder = st.image(display_image, caption='Segmented Image', use_column_width=True)
        with col2:
            st.subheader("Classify")
            dress = False
            top = False
            outer = False
            cbxs = []
            for cl in cat:
                if not outer and cl == 4 :
                    outer = True
                    cbxs.append([cl, st.checkbox(names[cl])])
                elif not dress and cl == 5 : 
                    dress = True
                    cbxs.append([cl, st.checkbox(names[cl])])
                elif not top and cl == 8 :
                    top = True
                    cbxs.append([cl, st.checkbox(names[cl])])
            
        # Checkboxes for different filters
        active = []
        for cbx in cbxs:
            if cbx[1]:
                active.append(cbx[0])
            if not cbx[1]:
                if cbx[0] in active:
                    active.remove(cbx[0])
        if len(active):
            display_image = resize_image_np.copy()
            for box, class_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if class_id in active:
                    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                    x_min, y_min, x_max, y_max = map(int, box)
                    center = [(x_min + x_max) //2, (y_min + y_max) //2]
                    x1, y1 = [center[0] - 56, center[1] - 56]
                    x2, y2 = [center[0] + 56, center[1] + 56]
                    # Crop the image using the bounding box coordinates
                    cropped_image = resize_image_np[y1:y2, x1:x2]
                    st.write("Fabric classified using the below extracted piece")
                    st.image(cropped_image)

                    # Convert BGR to RGB for displaying with Streamlit
                    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    [cl_name, conf] = make_prediction(cropped_image_rgb)
                    # Draw the bounding box on the image
                    cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), color, 2)
                    # Add a label to the bounding box
                    label_text = f"{cl_name}, {conf:.2f}%"
                    draw_label_with_background(display_image, label_text, (x_min, y_min), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            image_placeholder.image(display_image)