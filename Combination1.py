import random
import cv2
from ultralytics import YOLO
import easyocr

# Load YOLO model and class list
model = YOLO("weights/yolov8n.pt", "v8")

# Load class list for object dpipetection
my_file = open("utils/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Create EasyOCR reader for text recognitio
reader = easyocr.Reader(['en'])

# Vals to resize video frames
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or change the index for a different camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize frame for object detection
        frame_resized = cv2.resize(frame, (frame_wid, frame_hyt))

        # Perform object detection
        detect_params = model.predict(source=[frame_resized], conf=0.45, save=False)

        # Convert tensor array to numpy
        DP = detect_params[0].numpy()

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

        # Perform text recognition
        result = reader.readtext(frame)

        for detection in result:
            points = detection[0]
            text = detection[1]

            # Extract top-left and bottom-right coordinates
            top_left = tuple(map(int, points[0]))
            bottom_right = tuple(map(int, points[2]))

            # Draw rectangle and text
            frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
            frame = cv2.putText(frame, text, top_left, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame using OpenCV's imshow
        cv2.imshow("ObjectDetectionAndTextRecognition", frame)

        if cv2.waitKey(1) == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()