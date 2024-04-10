import cv2
import pytesseract
from langdetect import detect  # Import language detection library
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
    # Split the text into paragraphs and print each paragraph
    paragraphs = text.split('\n\n')  # Split based on empty lines
    paragraph_number = 1
    for paragraph in paragraphs:
        if paragraph.strip():  # Check if the paragraph is not empty
            print(f"Paragraph {paragraph_number}: {paragraph}")
            paragraph_number += 1

def extract_and_read_text(image, lang):
    print(f"Extracting text for language: {lang}")
    text = extract_text(image, lang)
    print("Extracted text:")
    read_text(text)
    print("\n")

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 for the default webcam, change accordingly if you have multiple cameras

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        # Check for a key press
        key = cv2.waitKey(1)
        
        # If the 'c' key is pressed, capture the frame and perform OCR
        if key & 0xFF == ord('c'):
            # Convert the frame to grayscale for preprocessing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Preprocess the frame
            processed_frame = preprocess_image(gray_frame)

            # Perform text recognition using pytesseract
            text = pytesseract.image_to_string(processed_frame, lang='hin+eng+mar')

            # Detect language of the recognized text
            try:
                language = detect(text)
            except Exception as e:
                language = "Unknown"
            
            # Print the detected language and recognized text
            print("Detected Language:", language)

            # Extract and read text based on detected language
            if language == 'en':
                extract_and_read_text(processed_frame, 'eng')
            elif language == 'hi':
                extract_and_read_text(processed_frame, 'hin')
            elif language == 'mr':
                extract_and_read_text(processed_frame, 'mar')
            else:
                print("Language not supported.")

        # Break the loop if 'q' is pressed
        if key & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
