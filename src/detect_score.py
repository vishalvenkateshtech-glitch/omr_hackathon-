import cv2
import numpy as np
import os

# Answer key
ANSWER_KEY = {
    0: "C",
    1: "A",
    2: "E",
    3: "D",
    4: "B"
}

def detect_answers(image_path, debug=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)

    h, w = img.shape

    # ✅ Adjusted crop (slightly lower + right)
    grid = img[int(0.30*h):int(0.86*h), int(0.22*w):int(0.86*w)]
    color_grid = color_img[int(0.30*h):int(0.86*h), int(0.22*w):int(0.86*w)]

    # Resize to clean 500x500 so it's perfectly divisible
    grid = cv2.resize(grid, (500, 500))
    color_grid = cv2.resize(color_grid, (500, 500))

    # Threshold
    _, thresh = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Split into 5 rows (questions)
    rows = np.vsplit(thresh, 5)
    detected_answers = []

    for q_idx, row in enumerate(rows):
        cols = np.hsplit(row, 5)  # 5 options per question
        filled = [cv2.countNonZero(c) for c in cols]

        # Pick the darkest bubble
        marked = np.argmax(filled)
        max_val = filled[marked]

        if max_val > 500:  # threshold to ignore noise
            detected_answers.append(chr(ord("A") + marked))
        else:
            detected_answers.append(None)

        # Debug: Draw grid boxes
        if debug:
            for c_idx in range(5):
                x = c_idx * 100
                y = q_idx * 100
                cv2.rectangle(color_grid, (x, y), (x+100, y+100), (0, 255, 0), 2)
                if marked == c_idx:
                    cv2.putText(color_grid, "X", (x+30, y+60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if debug:
        debug_path = image_path.replace(".jpg", "_debug.jpg")
        cv2.imwrite(debug_path, color_grid)
        print(f"Debug image saved at {debug_path}")

    return detected_answers

if __name__ == "__main__":
    data_dir = "../data"
    sheets = [f for f in os.listdir(data_dir) if f.startswith("preprocessed")]

    for sheet in sheets:
        sheet_path = os.path.join(data_dir, sheet)
        detected = detect_answers(sheet_path, debug=True)

        # Scoring
        score = sum(1 for i, ans in enumerate(detected) if ans == ANSWER_KEY.get(i))

        print(f"\nSheet: {sheet}")
        print("Detected answers:", detected)
        print(f"Score: {score}/5")
