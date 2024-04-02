import cv2
import pytesseract

# Load the image
image = cv2.imread('C:\\Users\\91781\\Desktop\\tc10.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or other preprocessing if needed
# For example:
# threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(gray_image)

# Print the extracted text
print(extracted_text)

# Optionally, you can save the extracted text to a file
with open('extracted_text.txt', 'w') as f:
    f.write(extracted_text)