import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

count = 0
scale = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold

def scan_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 3)

def center_text(image, text):
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, 2, (255, 0, 255), 5, cv2.LINE_AA)

image_path = "test1.jpg"
frame = cv2.imread(image_path)

frame_copy = frame.copy()
scan_detection(frame_copy)

cv2.imshow("input", cv2.resize(frame, (int(scale * frame.shape[1]), int(scale * frame.shape[0]))))

warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
cv2.imshow("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))

processed = image_processing(warped)
cv2.imshow("Processed", cv2.resize(processed, (int(scale * processed.shape[1]), int(scale * processed.shape[0]))))

cv2.waitKey(0)
cv2.destroyAllWindows()
