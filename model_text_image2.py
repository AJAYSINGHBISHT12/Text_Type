import os
import sys
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms
import argparse

import cv2
import numpy as np
from skimage import io

seed = 1234
random.seed(seed)
torch.manual_seed(seed)

CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}
SAVE_PATH = "C:\\Users\\91781\\Desktop"
MODEL_PATH = "C:\\Users\\91781\\Desktop\\model_196000.pth"


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


def show(img, name="disp", width=1000):
    """
    name: name of window, should be name of img
    img: source of img, should in type ndarray
    """
    cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(name, width, 1000)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def overlay_mask(image, mask, alpha=0.5):
    c = (np.random.random((1, 3)) * 153 + 102).tolist()[0]

    mask = np.dstack([mask.astype(np.uint8)] * 3)
    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    inv_mask = 255 - mask

    overlay = image.copy()
    overlay = np.minimum(overlay, inv_mask)

    color_mask = (mask.astype(np.bool) * c).astype(np.uint8)
    overlay = np.maximum(overlay, color_mask).astype(np.uint8)

    image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
    return image


def overlay_ann(image, mask, box, label, score, alpha=0.5):
    c = np.random.random((1, 3))
    mask_color = (c * 153 + 102).tolist()[0]
    text_color = (c * 183 + 72).tolist()[0]

    mask = np.dstack([mask.astype(np.uint8)] * 3)
    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    inv_mask = 255 - mask

    overlay = image.copy()
    overlay = np.minimum(overlay, inv_mask)

    color_mask = (mask.astype(np.bool_) * mask_color).astype(np.uint8)

    overlay = np.maximum(overlay, color_mask).astype(np.uint8)

    image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)

    cv2.rectangle(
        image,
        (box[0], box[1]),
        (box[2], box[3]),
        mask_color, 1
    )

    (label_size_width, label_size_height), base_line = \
        cv2.getTextSize(
            "{}".format(label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, 1
        )

    cv2.rectangle(
        image,
        (box[0], box[1] + 10),
        (box[0] + label_size_width, box[1] + 10 - label_size_height),
        (223, 128, 255),
        cv2.FILLED
    )

    cv2.putText(
        image,
        # "{}: {:.3f}".format(label, score),
        "{}".format(label),
        (box[0], box[1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3, (0, 0, 0), 1
    )

    return image


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

    # Return the scanned document.
    return warped


def enhance_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Display the original and enhanced images
    cv2.imshow('Original Image', gray)
    cv2.imshow('Enhanced Image', equalized)
    io.imsave("output2.jpg", equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    num_classes = 6
    model = get_instance_segmentation_model(num_classes)
    # No CUDA
    model.eval()

    if os.path.exists(MODEL_PATH):
        checkpoint_path = MODEL_PATH
    else:
        checkpoint_path = MODEL_PATH

    print(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    image_path = 'test12.jpg'
    print(image_path)
    assert os.path.exists(image_path)

    # Load the original image
    original_image = cv2.imread(image_path)

    # Scan the document
    scanned_document = scan_document(original_image)

    if scanned_document is not None:
        # Process the scanned document
        image = scanned_document
    else:
        print("Document scanning failed. Proceeding with original image.")
        image = original_image

    rat = 1000 / image.shape[0]
    image = cv2.resize(image, None, fx=rat, fy=rat)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = transform(image)

    with torch.no_grad():
        prediction = model([image])

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:
                continue

            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]

            score = pred["scores"][idx].item()

            image = overlay_ann(image, m, box, label, score)

    cv2.imwrite('/{}'.format(os.path.basename(image_path)), image)

    show(image)

    # Enhance the image
    enhance_image(image_path)


if __name__ == "__main__":
    main()
