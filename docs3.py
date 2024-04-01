import cv2
import easyocr

# Function to remove extra space around text in an image
def remove_extra_space(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR to detect text regions
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)

    # Extract bounding boxes of text regions
    bounding_boxes = [result[0] for result in results]

    # Calculate the bounding box of all text regions
    min_x = min(box[0][0] for box in bounding_boxes)
    min_y = min(box[0][1] for box in bounding_boxes)
    max_x = max(box[2][0] for box in bounding_boxes)
    max_y = max(box[2][1] for box in bounding_boxes)

    # Crop the image to include only the text regions
    cropped_image = image[min_y:max_y, min_x:max_x]

    # Save the cropped image
    cv2.imwrite("C:\\Users\\91781\\Desktop\\output.jpg", cropped_image)

# Example usage
remove_extra_space("C:\\Users\\91781\\Desktop\\snapshot.jpg")
