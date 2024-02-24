import os
import cv2
import numpy as np
import onnxruntime
from torchvision.transforms import transforms

# Define the categories to labels mapping
CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}

# Function to overlay mask on the image
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

# Function to overlay annotations on the image
def overlay_ann(image, mask, box, label, score, alpha=0.5):
    c = np.random.random((1, 3))
    mask_color = (c * 153 + 102).tolist()[0]
    text_color = (c * 183 + 72).tolist()[0]
    mask = np.dstack([mask.astype(np.uint8)] * 3)
    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    inv_mask = 255 - mask
    overlay = image.copy()
    overlay = np.minimum(overlay, inv_mask)
    color_mask = (mask.astype(np.bool) * mask_color).astype(np.uint8)
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

# Path to the ONNX model folder
onnx_model_folder = r'C:\Users\91781\Desktop\model_196000'

# Load the ONNX model
onnx_model_path = os.path.join(onnx_model_folder, 'model_196000.onnx')
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Image path
image_path = r'C:\Users\91781\Desktop\Notice.jpg'
assert os.path.exists(image_path)

# Read and preprocess the image
image = cv2.imread(image_path)
image = cv2.resize(image, (1300, 1300))  # Resize the image to match the model's input shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
image = transform(image)

# Perform inference
input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]
image_np = image.numpy().astype(np.float32)

result = ort_session.run(output_names, {input_name: image_np})

# Flag to check if text is detected
text_detected = False

# Process predictions and overlay annotations
for pred in result:
    if isinstance(pred, np.ndarray):  # Check if pred is a NumPy array
        print("Shape of 'pred' array:", pred.shape)  # Debugging line
        continue

    for idx, label in enumerate(pred.get("labels", [])):
        # Check if the label corresponds to text
        if label.item() == 1:  # Assuming "text" label is index 1
            score = pred.get("scores", [])[idx].item()
            if score > 0.7:  # Set your confidence threshold here
                text_detected = True
                mask = pred.get("masks", [])[idx][0].mul(255).byte().cpu().numpy()
                box = pred.get("boxes", [])[idx].tolist()
                image = overlay_ann(image, mask, box, "Text", score)

# Ensure image is converted back to numpy array before saving
image = image.numpy().transpose((1, 2, 0)).astype(np.uint8)

# Save the annotated image
cv2.imwrite(r'C:\Users\91781\Desktop\annotated_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if text_detected:
    print("Text detected and annotated on the image.")
else:
    print("No text detected in the image.")
