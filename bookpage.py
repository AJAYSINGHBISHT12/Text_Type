import cv2
import pytesseract
from gtts import gTTS
import os

def preprocess_image(image):
    # Apply preprocessing techniques like resizing, noise reduction, etc.
    # Return the preprocessed image
    return image

def extract_text(image):
    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(image)
    return text

def read_text(text):
    # Split the text into lines and print each line
    for line in text.split('\n'):
        print(line)

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Extract text from each page
    pages_text = []
    # Assuming only one page in this example
    text = extract_text(processed_image)
    pages_text.append(text)

    # Read the text of each page
    for text in pages_text:
        read_text(text)

if __name__ == "__main__":
    image_path = "Book2.jpg"  # Provide the path to your book image
    main(image_path)
