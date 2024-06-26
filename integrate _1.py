import os
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms

import cv2
import numpy as np
import pytesseract
from langdetect import detect

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
SAVE_PATH = r"."
MODEL_PATH = "model_196000.pth"

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

    # draw on color mask
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
        "{}".format(label),
        (box[0], box[1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3, (0, 0, 0), 1
    )

    return image

def preprocess_frame(frame):
    # Apply preprocessing steps such as denoising, contrast adjustment, etc.
    # Example: frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    return frame

def predict_from_image(image_path):
    num_classes = 6
    model = get_instance_segmentation_model(num_classes)
    model.eval()

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = transform(image)

    with torch.no_grad():
        prediction = model([image])

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            score = pred["scores"][idx].item()

            if score > 0.5:
                image = overlay_ann(image, m, box, label, score)

    return image

def main():
    num_classes = 6
    model = get_instance_segmentation_model(num_classes)
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

    cap = cv2.VideoCapture(0)  # Open webcam

    prev_predictions = []  # Previous frame's predictions for temporal smoothing
    smoothing_window_size = 5  # Number of frames to average predictions over

    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        frame = preprocess_frame(frame)  # Apply preprocessing

        rat = 1000 / frame.shape[0]
        frame = cv2.resize(frame, None, fx=rat, fy=rat)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        image = transform(frame)

        with torch.no_grad():
            prediction = model([image])

        image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

        avg_confidence = 0
        num_predictions = 0

        for pred in prediction:
            for idx, mask in enumerate(pred['masks']):
                m = mask[0].mul(255).byte().cpu().numpy()
                box = list(map(int, pred["boxes"][idx].tolist()))
                label = CATEGORIES2LABELS[pred["labels"][idx].item()]

                score = pred["scores"][idx].item()
                avg_confidence += score
                num_predictions += 1

                if score > 0.5:
                    image = overlay_ann(image, m, box, label, score)

        # Calculate average confidence score
        if num_predictions > 0:
            avg_confidence /= num_predictions

        # Update previous predictions for temporal smoothing
        prev_predictions.append(avg_confidence)
        if len(prev_predictions) > smoothing_window_size:
            prev_predictions.pop(0)

        # Calculate dynamic confidence threshold
        if len(prev_predictions) > 0:
            avg_confidence = sum(prev_predictions) / len(prev_predictions)
            confidence_threshold = max(0.5, avg_confidence)

        # Perform text recognition
                # Perform text recognition using pytesseract
        text = pytesseract.image_to_string(frame, lang='eng')

        # Detect language of the recognized text
        try:
            language = detect(text)
        except Exception as e:
            language = "Unknown"

        # If text is detected, save the frame and use it for prediction
        if text.strip() != "":
            cv2.imwrite(os.path.join(SAVE_PATH, "snapshot.jpg"), frame)
            prediction_result = predict_from_image(os.path.join(SAVE_PATH, "snapshot.jpg"))

            # Do something with the prediction result
            # For example, you can display it:
            cv2.imshow('Prediction Result', prediction_result)
        else:
            # Display the original frame if no text is detected
            cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
