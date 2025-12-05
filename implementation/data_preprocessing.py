import cv2
import os

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

if __name__ == "__main__":
    print("Preprocessing example image...")
