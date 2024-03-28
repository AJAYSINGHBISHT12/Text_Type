import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms

# Constants
CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}
MODEL_PATH = "C:\\Users\\91781\\Desktop\\model_196000.pth"


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def overlay_ann(image, mask, box, label, score, alpha=0.5):
    # Function to overlay annotations on the image
    # Same as provided in your original code


def scan_document(image_path):
    # Function to scan the document
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)

    # Thresholding
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours of the document
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Get the four corners of the document
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

    # Order the points to be consistent with the perspective transformation
    rect_points = np.float32([point[0] for point in approx])

    # Ensure correct orientation of the document
    rect_points = rect_points[np.argsort(rect_points[:, 1])]  # Sort points based on y-coordinate

    # Reorder the points to ensure top-left, top-right, bottom-right, bottom-left order
    if rect_points[0][0] > rect_points[1][0]:
        rect_points[[0, 1]] = rect_points[[1, 0]]
    if rect_points[2][0] < rect_points[3][0]:
        rect_points[[2, 3]] = rect_points[[3, 2]]

    # Perspective transform the document
    dst = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect_points, dst)
    warped = cv2.warpPerspective(image, M, (600, 600))

    return warped


def main():
    # Load the text detection model
    num_classes = 6
    model = get_instance_segmentation_model(num_classes)
    model.eval()

    # Load the trained weights
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Input image path
    image_path = 'C:\\Users\\91781\\Desktop\\snapshot.jpg'
    assert os.path.exists(image_path)

    # Scan the document
    scanned_document = scan_document(image_path)

    # Convert the scanned document to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    scanned_document_tensor = transform(scanned_document)

    # Perform text detection on the scanned document
    with torch.no_grad():
        prediction = model([scanned_document_tensor])

    # Overlay annotations on the original image
    image = cv2.imread(image_path)
    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:
                continue
            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            score = pred["scores"][idx].item()
            image = overlay_ann(image, m, box, label, score)

    # Save and display the final image
    cv2.imwrite('/{}'.format(os.path.basename(image_path)), image)
    show(image)


if __name__ == "__main__":
    main()
