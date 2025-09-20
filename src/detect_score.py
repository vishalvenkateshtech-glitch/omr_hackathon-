import cv2
import numpy as np
import pandas as pd
import os
import csv
import re

# ---------------- Find Excel File Automatically ----------------
def find_excel_file(data_folder, keyword="key"):
    for f in os.listdir(data_folder):
        if f.lower().endswith(('.xlsx', '.xls')) and keyword.lower() in f.lower():
            return os.path.join(data_folder, f)
    raise FileNotFoundError(f"No Excel file containing '{keyword}' found in {data_folder}")

# ---------------- Load Answer Key ----------------
def load_answer_key_from_excel(data_folder):
    excel_path = find_excel_file(data_folder, keyword="key")
    print(f"Using Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    option_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    answer_key = {}
    
    for col in df.columns:
        col_answers = []
        for ans in df[col].dropna():
            try:
                ans_str = str(ans).strip()
                # Split by '-' or '.'
                if '-' in ans_str:
                    parts = ans_str.split("-")[1].split(",")
                elif '.' in ans_str:
                    parts = ans_str.split(".")[1].split(",")
                else:
                    parts = [ans_str]  # fallback if only option letter
                indices = [option_map[p.strip().lower()] for p in parts]
                col_answers.append(indices)
            except Exception as e:
                print(f"Error parsing cell '{ans}' in column '{col}': {e}")
                col_answers.append([])
        answer_key[col] = col_answers
    
    return answer_key

# ---------------- Bubble Detection & Scoring ----------------
def detect_bubbles_and_score(image_path, answer_key, options_per_question=4, fill_threshold=0.2):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = thresh.shape
    subjects = list(answer_key.keys())
    questions_per_subject = len(answer_key[subjects[0]])
    
    row_height = h // questions_per_subject
    col_width = w // (len(subjects) * options_per_question)
    
    total_score = 0
    scores = {}
    
    for s_idx, subject in enumerate(subjects):
        correct = 0
        for q_idx in range(questions_per_subject):
            y_start = q_idx * row_height
            y_end = y_start + row_height
            max_fill = 0
            marked_option = None
            
            for o_idx in range(options_per_question):
                x_start = (s_idx * options_per_question + o_idx) * col_width
                x_end = x_start + col_width
                bubble = thresh[y_start:y_end, x_start:x_end]
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bubble)
                if num_labels > 1:
                    largest_area = max(stats[1:, cv2.CC_STAT_AREA])
                    fill_ratio = largest_area / (bubble.shape[0]*bubble.shape[1])
                else:
                    fill_ratio = 0
                
                if fill_ratio > max_fill:
                    max_fill = fill_ratio
                    marked_option = o_idx
            
            if max_fill > fill_threshold:
                if marked_option in answer_key[subject][q_idx]:
                    correct += 1
        scores[subject] = correct
        total_score += correct
    
    scores['total'] = total_score
    return scores

# ----------------- NUMERIC SORT FUNCTION -----------------
def numeric_sort_key(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

# ----------------- MAIN DRIVER -----------------
if __name__ == "__main__":
    preprocessed_folder = "./preprocessed"
    data_folder = "../data"
    output_file = "./results.csv"
    
    # Load answer key
    answer_key = load_answer_key_from_excel(data_folder)
    print("Answer key loaded successfully.")
    
    # Process preprocessed images
    image_files = [f for f in os.listdir(preprocessed_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    image_files.sort(key=numeric_sort_key)
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(preprocessed_folder, img_file)
        try:
            scores = detect_bubbles_and_score(img_path, answer_key)
            row = {"filename": img_file}
            row.update(scores)
            results.append(row)
            print(f"Processed {img_file}: {scores}")
        except Exception as e:
            print(f"Failed {img_file}: {e}")
    
    if results:
        fieldnames = ["filename"] + list(answer_key.keys()) + ["total"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {output_file}")
