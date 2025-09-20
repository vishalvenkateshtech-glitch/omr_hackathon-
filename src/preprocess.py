import cv2
import os

images = [
    "MCQPaper.jpg",
    "MCQPaper (1).jpg",
    "MCQPaper (2).jpg"
]

for img_name in images:
    image_path = os.path.join("..", "data", img_name)


    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow(f"Preprocessed - {img_name}", thresh)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    preprocessed_path = os.path.join("..", "data", f"preprocessed_{img_name}")
    cv2.imwrite(preprocessed_path, thresh)
    print(f"Preprocessed image saved at {preprocessed_path}")