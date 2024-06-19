import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from gtts import gTTS
import os

# Function to convert PDF page to image
def pdf_page_to_image(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    image = Image.open(io.BytesIO(pix.tobytes()))
    return image

# Function to extract text from image using OCR
def ocr_image(image, lang='hin'):
    text = pytesseract.image_to_string(image, lang=lang)
    return text

# Function to extract text from each page of a PDF
def extract_text_from_pdf(pdf_path, lang='hin'):
    pdf_document = fitz.open(pdf_path)
    num_pages = pdf_document.page_count
    full_text = ""

    for page_number in range(num_pages):
        image = pdf_page_to_image(pdf_path, page_number)
        text = ocr_image(image, lang=lang)
        full_text += text + "\n"
    
    return full_text

# Function to convert text to speech using gTTS
def text_to_speech(text, lang='hi', save_path='output.mp3'):
    tts = gTTS(text=text, lang=lang)
    tts.save(save_path)
    return save_path

# Example usage
pdf_path = 'Hindi.pdf'
text = extract_text_from_pdf(pdf_path, lang='hin')

# Print the extracted text
print(text)

# Save the extracted text to a file
output_text_file = 'output.txt'
with open(output_text_file, 'w', encoding='utf-8') as file:
    file.write(text)

print(f"The text has been saved to {output_text_file}")

# Convert text to speech
output_audio_file = 'output.mp3'
text_to_speech(text, lang='hi', save_path=output_audio_file)

print(f"The text has been converted to audio and saved to {output_audio_file}")