import cv2
import numpy as np
from skimage import io

def scan_document(image):
    # Preprocess the image.
    image = cv2.resize(image, (600, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours of the document.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and shape to get the document contour.
    max_area = 0
    max_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # Check if contour is quadrilateral
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

    # Check if a valid contour was found
    if max_contour is None:
        print("No document contour found.")
        return None

    # Get the four corners of the document.
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

    # Order the points to ensure top-left, top-right, bottom-right, bottom-left order.
    rect_points = np.float32([point[0] for point in approx])
    rect_points = rect_points[np.argsort(rect_points[:, 1])]  # Sort points based on y-coordinate

    if rect_points[0][0] > rect_points[1][0]:
        rect_points[[0, 1]] = rect_points[[1, 0]]
    if rect_points[2][0] < rect_points[3][0]:
        rect_points[[2, 3]] = rect_points[[3, 2]]

    # Check if the number of points is 4
    if len(rect_points) != 4:
        print("Could not detect four corners of the document.")
        return None

    # Perspective transform the document.
    dst = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect_points, dst)
    warped = cv2.warpPerspective(image, M, (600, 600))

    # Return the scanned document and contour points.
    return warped, rect_points

def enhance_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Display the original and enhanced images
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = io.imread("test3.jpg")
    scanned_document, contour_points = scan_document(image)
    
    if scanned_document is not None:
        # Convert to uint8 for saving the image
        scanned_document = cv2.convertScaleAbs(scanned_document)
        io.imsave("output.jpg", scanned_document)
        
        image_path = "test2.jpg"  # Change to the path of your image
        enhance_image(image_path)
