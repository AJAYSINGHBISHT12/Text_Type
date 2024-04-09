import cv2
import pytesseract
from langdetect import detect  # Import language detection library
from gtts import gTTS
import os

def preprocess_image(image):
    # Apply preprocessing techniques like resizing, noise reduction, etc.
    # Return the preprocessed image
    return image

def extract_text(image, lang):
    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def read_text(text):
    # Split the text into lines and print each line
    for line in text.split('\n'):
        print(line)

def extract_and_read_text(image, lang):
    print(f"Extracting text for language: {lang}")
    text = extract_text(image, lang)
    print("Extracted text:")
    read_text(text)
    print("\n")

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Perform text recognition using pytesseract
    text = pytesseract.image_to_string(processed_image, lang='hin+eng+mar')

    # Detect language of the recognized text
    try:
        language = detect(text)
    except Exception as e:
        language = "Unknown"
    
    # Print the detected language and recognized text
    print("Detected Language:", language)
    # cc

    # Extract and read text based on detected language
    if language == 'en':
        extract_and_read_text(processed_image, 'eng')
    elif language == 'hi':
        extract_and_read_text(processed_image, 'hin')
    elif language == 'mr':
        extract_and_read_text(processed_image, 'mar')
    else:
        print("Language not supported.")

if __name__ == "__main__":
    image_path = "Book2.jpg"  # Provide the path to your book image
    main(image_path)
