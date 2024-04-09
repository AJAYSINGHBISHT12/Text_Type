import cv2
import numpy as np

def scan_document(image):
  """Scans a document and returns the scanned image."""

  # Convert the image to grayscale.
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply a threshold to the image.
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

  # Find the contours of the document.
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  # Find the largest contour.
  largest_contour = max(contours, key=cv2.contourArea)

  # Get the bounding box of the largest contour.
  x, y, w, h = cv2.boundingRect(largest_contour)

  # Crop the image to the bounding box.
  scanned_image = image[y:y+h, x:x+w]

  # Return the scanned image.
  return scanned_image

# Load the image.
image = cv2.imread("C:\\Users\\91781\\Desktop\\PMC5055614_00001.jpg")

# Scan the document.
scanned_image = scan_document(image)

# Display the scanned image.
cv2.imshow("Scanned Image", scanned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()