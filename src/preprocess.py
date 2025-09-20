import cv2
import numpy as np
import os

def deskew_image(thresh):
    """
    Detect skew angle using Hough lines and rotate image to correct orientation.
    Returns rotated image and angle.
    """
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0
    if lines is not None:
        angles = []
        for rho, theta in lines[:,0]:
            deg = (theta * 180 / np.pi) - 90
            if -45 < deg < 45:  # ignore vertical lines
                angles.append(deg)
        if len(angles) > 0:
            angle = np.median(angles)
            # rotate image
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image_path, standard_width=1000, margin=10):
    """
    Preprocess OMR sheet image:
    - Resize if too small
    - Adaptive thresholding + morphological cleaning
    - Deskew (auto-rotate)
    - Crop only the answer area using projection profiles
    Returns: cropped color image, thresholded binary image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Resize if too small
    h, w = image.shape[:2]
    if w < standard_width:
        ratio = standard_width / w
        image = cv2.resize(image, (standard_width, int(h*ratio)))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Deskew using Hough lines
    angle = deskew_image(cleaned)
    if abs(angle) > 0.5:  # ignore tiny angles
        print(f"Rotating {image_path} by {angle:.2f} degrees")
        image = rotate_image(image, angle)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Horizontal projection to detect rows of bubbles
    hor_proj = np.sum(cleaned, axis=1)
    y_indices = np.where(hor_proj > 0.15 * np.max(hor_proj))[0]
    if len(y_indices) == 0:
        raise ValueError("No answer rows detected.")
    y_start, y_end = max(0, y_indices[0]-margin), min(cleaned.shape[0], y_indices[-1]+margin)
    
    # Vertical projection to detect columns of bubbles
    ver_proj = np.sum(cleaned, axis=0)
    x_indices = np.where(ver_proj > 0.15 * np.max(ver_proj))[0]
    if len(x_indices) == 0:
        raise ValueError("No answer columns detected.")
    x_start, x_end = max(0, x_indices[0]-margin), min(cleaned.shape[1], x_indices[-1]+margin)
    
    # Crop the answer area
    cropped_color = image[y_start:y_end, x_start:x_end]
    cropped_gray = gray[y_start:y_end, x_start:x_end]
    
    # Threshold for detection
    cropped_thresh = cv2.adaptiveThreshold(
        cropped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    return cropped_color, cropped_thresh

# ----------------- MAIN DRIVER -----------------
if __name__ == "__main__":
    data_folder = "../data"  # Relative path to your images
    output_folder = "./preprocessed"
    os.makedirs(output_folder, exist_ok=True)
    
    # Clear folder before preprocessing
    for f in os.listdir(output_folder):
        file_path = os.path.join(output_folder, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleared folder: {output_folder}")
    
    supported_ext = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(data_folder) if f.lower().endswith(supported_ext)]
    
    if not images:
        print(f"No image files found in {data_folder}")
    
    for img_file in images:
        img_path = os.path.join(data_folder, img_file)
        print(f"Processing {img_file}...")
        try:
            cropped_color, cropped_thresh = preprocess_image(img_path)
            output_path = os.path.join(output_folder, img_file)
            cv2.imwrite(output_path, cropped_color)
            print(f"Saved preprocessed image: {output_path}")
        except Exception as e:
            print(f"Failed to process {img_file}: {e}")
