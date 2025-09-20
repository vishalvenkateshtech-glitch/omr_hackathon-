import cv2
import numpy as np
import os

def preprocess_image(image_path, standard_width=1000):
    """
    Preprocess OMR sheet image:
    - Resize if too small
    - Grayscale + blur
    - Adaptive thresholding
    - Morphological cleaning
    - Robust contour detection with Canny
    - Perspective correction
    Returns: warped color image, thresholded binary image
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
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection
    edges = cv2.Canny(cleaned, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError("No contours found in image.")
    
    sheet_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(sheet_contour, True)
    approx = cv2.approxPolyDP(sheet_contour, 0.02 * peri, True)
    
    # Fallback: bounding rectangle if approx is not 4 points
    if len(approx) == 4:
        rect = order_points(approx.reshape(4,2))
    else:
        print(f"Warning: Sheet contour not 4 points, using bounding rectangle for {image_path}")
        x, y, w, h = cv2.boundingRect(sheet_contour)
        rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype="float32")
    
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Thresholded binary for detection
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    return warped, warped_thresh

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ----------------- MAIN DRIVER -----------------
if __name__ == "__main__":
    data_folder = "../data"  # Relative path to your images
    output_folder = "./preprocessed"
    os.makedirs(output_folder, exist_ok=True)
    
    supported_ext = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(data_folder) if f.lower().endswith(supported_ext)]
    
    if not images:
        print(f"No image files found in {data_folder}")
    
    for img_file in images:
        img_path = os.path.join(data_folder, img_file)
        print(f"Processing {img_file}...")
        try:
            warped, warped_thresh = preprocess_image(img_path)
            output_path = os.path.join(output_folder, img_file)
            cv2.imwrite(output_path, warped)
            print(f"Saved preprocessed image: {output_path}")
        except Exception as e:
            print(f"Failed to process {img_file}: {e}")
