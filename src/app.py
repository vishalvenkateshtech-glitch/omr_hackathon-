# app.py
import streamlit as st
import cv2
import numpy as np
from detect_score import detect_answers  # we'll adapt detect_score to a function

st.title("Automated OMR Evaluation System")

st.write("Upload one or more OMR sheet images to detect answers and calculate scores.")

uploaded_files = st.file_uploader("Upload OMR Sheets (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Hardcoded answer key
answer_key = {1: "A", 2: "C", 3: "B", 4: "D", 5: "E"}

fill_ratio_threshold = 0.3  # adjust if needed

# Function to convert uploaded file to OpenCV image
def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img

# Fixed grid coordinates (adjust per your sheet)
x_start, y_start = 100, 200
x_end, y_end = 600, 800
num_questions = 5
num_options = 5

if uploaded_files:
    for file in uploaded_files:
        img = load_image(file)
        answer_area = img[y_start:y_end, x_start:x_end]
        cell_height = answer_area.shape[0] // num_questions
        cell_width = answer_area.shape[1] // num_options
        detected_answers = []

        # Detect answers
        for q in range(num_questions):
            row = answer_area[q*cell_height:(q+1)*cell_height, :]
            fill_ratios = []
            for o in range(num_options):
                cell = row[:, o*cell_width:(o+1)*cell_width]
                total_pixels = cell.size
                filled_pixels = cv2.countNonZero(cell)
                fill_ratio = filled_pixels / total_pixels
                fill_ratios.append(fill_ratio)
            max_idx = np.argmax(fill_ratios)
            selected = chr(65 + max_idx) if fill_ratios[max_idx] > fill_ratio_threshold else "None"
            detected_answers.append(selected)

        score = sum([detected_answers[i] == answer_key[i+1] for i in range(len(detected_answers))])

        st.subheader(f"Sheet: {file.name}")
        st.write("Detected Answers:", detected_answers)
        st.write(f"Score: {score}/{num_questions}")

        # Show uploaded image
        st.image(img, channels="GRAY", caption="Uploaded Sheet", use_column_width=True)
