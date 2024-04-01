import cv2
from skimage import io

def scan_document(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours of the document.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to get the document contour.
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Get the rotation angle of the document
    rect = cv2.minAreaRect(max_contour)
    angle = rect[-1]

    # Rotate the image to correct orientation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Convert rotated image to grayscale
    rotated_image_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(rotated_image_gray)

    # Thresholding again within the document contour
    _, thresh = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x, y, w, h = cv2.boundingRect(max_contour)
    thresh = thresh[y:y+h, x:x+w]

    # Find contours again after enhancement
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    # Save the output image
    io.imsave("C:\\Users\\91781\\Desktop\\output.jpg", output_image)

# Call the function with the input image path
scan_document("C:\\Users\\91781\\Desktop\\snapshot.jpg")
