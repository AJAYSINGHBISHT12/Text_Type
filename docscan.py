import cv2
import numpy as np
from skimage import io

def scan_document(image):
    # Preprocess the image.
    image = cv2.resize(image, (600, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization to enhance contrast.
    equalized = cv2.equalizeHist(gray)

    # Thresholding
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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

    # Get the four corners of the document.
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

    # Order the points to be consistent with the perspective transformation.
    rect_points = np.float32([point[0] for point in approx])

    # Ensure correct orientation of the document.
    rect_points = rect_points[np.argsort(rect_points[:, 1])]  # Sort points based on y-coordinate

    # Reorder the points to ensure top-left, top-right, bottom-right, bottom-left order.
    if rect_points[0][0] > rect_points[1][0]:
        rect_points[[0, 1]] = rect_points[[1, 0]]
    if rect_points[2][0] < rect_points[3][0]:
        rect_points[[2, 3]] = rect_points[[3, 2]]

    # Perspective transform the document.
    dst = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect_points, dst)
    warped = cv2.warpPerspective(image, M, (600, 600))

    # Return the scanned document.
    return warped

image = io.imread("C:\\Users\\91781\\Desktop\\ocr.png")
scanned_document = scan_document(image)

# Convert to uint8 for saving the image
scanned_document = cv2.convertScaleAbs(scanned_document)

io.imsave("C:\\Users\\91781\\Desktop\\output.png", scanned_document)
