import cv2 as cv
import numpy as np

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def highPassFilter(kSize):
    global img
    
    print("Applying high pass filter")
    
    if not kSize % 2:
        kSize += 1
        
    kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
    
    filtered = cv.filter2D(img, -1, kernel)
    
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127 * np.ones(img.shape, np.uint8)
    
    filtered = filtered.astype('uint8')
    
    img = filtered

def blackPointSelect():
    global img
    
    print("Adjusting black point for final output ...")
    
    img = img.astype('int32')
    img = map(img, blackPoint, 255, 0, 255)

    _, img = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)

    img = img.astype('uint8')

def whitePointSelect():
    global img
    
    print("White point selection running ...")

    _, img = cv.threshold(img, whitePoint, 255, cv.THRESH_TRUNC)

    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')

def blackAndWhite():
    global img
    
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    (l, a, b) = cv.split(lab)

    img = cv.add(cv.subtract(l, b), cv.subtract(l, a))

def scan_document(image):
    image = cv.resize(image, (600, 600))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

    if max_contour is None:
        print("No document contour found.")
        return None

    peri = cv.arcLength(max_contour, True)
    approx = cv.approxPolyDP(max_contour, 0.02 * peri, True)

    rect_points = np.float32([point[0] for point in approx])
    rect_points = rect_points[np.argsort(rect_points[:, 1])]

    if rect_points[0][0] > rect_points[1][0]:
        rect_points[[0, 1]] = rect_points[[1, 0]]
    if rect_points[2][0] < rect_points[3][0]:
        rect_points[[2, 3]] = rect_points[[3, 2]]

    if len(rect_points) != 4:
        print("Could not detect four corners of the document.")
        return None

    dst = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
    M = cv.getPerspectiveTransform(rect_points, dst)
    warped = cv.warpPerspective(image, M, (600, 600))

    return warped

if __name__ == "__main__":
    img = cv.imread("test12.jpg")

    scanned_document = scan_document(img)

    if scanned_document is not None:
        img = scanned_document

        blackPoint = 66
        whitePoint = 160

        mode = "GCMODE"

        if mode == "GCMODE":
            highPassFilter(kSize=51)
            whitePoint = 127
            whitePointSelect()
            blackPointSelect()
        elif mode == "RMODE":
            blackPointSelect()
            whitePointSelect()
        elif mode == "SMODE":
            blackPointSelect()
            whitePointSelect()
            blackAndWhite()

        print("\nDone.")

        cv.imwrite('output2.jpg', img)

        cv.imshow("Final", cv.resize(img, None, fx=0.125, fy=0.125, interpolation=cv.INTER_CUBIC))
        cv.waitKey(0)
    else:
        print("No document contour found.")
